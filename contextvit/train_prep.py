
import os
import torch
import yaml
from torch import nn
from torchvision import transforms
from timm.data import create_transform, Mixup
from pathlib import Path

from .model import OuterModel, PushGrad
from .config import MEAN, STD, get_args, CUDA_DEVICE
from .data import HFImageDataset
from .train_utils import init_model, OptScheduler
from .utils import plot_data, reset
from modules.utils import IdleMonitor


def load_data(args):
    train_transforms = create_transform(
        input_size=args.kw["img_size"],
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.kw["img_size"] * 1.15)),
            transforms.CenterCrop([args.kw["img_size"]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    train_dataset = HFImageDataset(args.data_dir, "train", train_transforms)
    val_dataset = HFImageDataset(args.data_dir, "val", val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
    )

    args.steps_p_epoch = len(train_loader)
    print(f"INFO: Steps per epoch: {args.steps_p_epoch}")
    if args.print_samples > 0:
        plot_data(train_loader, args.print_samples)

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.kw["label_smoothing"],
        num_classes=1000,
    )

    return train_loader, val_loader, mixup_fn

def load_model(args):
    models = nn.ModuleDict()
    optimizers = {}
    scalers = {}

    for name, kw in args.models.items():
        models[name] = m = OuterModel(args, name, kw).cuda()
        params = init_model(m, args)
        optimizers[name] = opt = torch.optim.AdamW([*params.values()], fused=True)
        scalers[name] = scaler = torch.amp.GradScaler("cuda")
        m.backward = PushGrad(opt, scaler, args)

    opt_scheduler = OptScheduler(optimizers, args, args.exp)
    if args.checkpoint_path:
        print("INFO: Loading from provided checkpoint")

    checkpoint_path = args.checkpoint_path or (
        args.exp_dir / "model.pth" if (args.exp_dir / "model.pth").is_file() else None
    )

    if checkpoint_path and not args.new_run:
        print(f"INFO: Loading model from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except:
            assert checkpoint_path == args.exp_dir / "model.pth", "Loading failed"
            checkpoint = torch.load(args.exp_dir / "model_prev.pth", map_location="cpu")

        for n in models:
            models[n].load_state_dict(checkpoint["model"][n])
            models[n].backward.optimizer.load_state_dict(checkpoint["optimizer"][n])
            models[n].backward.scaler.load_state_dict(checkpoint["scaler"][n])
        if checkpoint.get("opt_scheduler"):
            opt_scheduler.load_state_dict(checkpoint["opt_scheduler"])
    else:
        print("INFO: Initializing new model")

    if args.compile:
        print("INFO: Compiling model")
        for m in models.values():
            m.compile_model()

    return models, optimizers, scalers, opt_scheduler


def prep_training(dict_args, exp):
    reset(0)
    dict_args["exp_root"] = Path(dict_args["exp_root"])
    pref = dict_args["exp_root"].relative_to("/notebooks/runs")
    pref = pref.as_posix().replace("/", "-")
    exp.set_name(f"{pref}-{dict_args['exp_version']}")
    print(f"INFO: Setting up experiment: {exp.get_name()}, key: {exp.get_key()}")

    # Args
    args = get_args(dict_args, check_args=True)
    args.exp_dir = args.exp_root / args.exp_version
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir = Path(args.exp_dir)
    
    if args.use_idle_monitor:
        print("INFO: Activating Idle monitoring")
        args.idle_monitor = IdleMonitor()
        
    # Compiling cache
    if args.compile and args.exp_cache:
        assert CUDA_DEVICE in str(args.exp_cache)
        print(f"INFO: TORCHINDUCTOR_CACHE_DIR = {args.exp_cache}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.exp_cache
    else:
        cache_dir = args.exp_dir / Path("cache") / Path(CUDA_DEVICE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    # Set config
    if (args.exp_dir / "params.yaml").is_file() and not args.new_run:
        with open(args.exp_dir / "params.yaml", "r") as f:
            exp_args = yaml.safe_load(f)

        keys_to_ignore = args.update_args + [
            "checkpoint_path",
            "exp_dir",
            "schedulers",
            "num_workers",
            "compile",
        ]
        for key, value in exp_args.items():
            if key not in keys_to_ignore:
                setattr(args, key, value)

        args.new_run = False
        print(f"INFO: Loading config from file: {args.exp_dir / 'params.yaml'}")

    dict_args = {k: v for k, v in sorted(vars(args).items())}
    dict_args["exp_root"] = str(dict_args["exp_root"])
    dict_args["exp_dir"] = str(dict_args["exp_dir"])

    if not (args.exp_dir / "params.yaml").is_file() or args.new_run:
        if (args.exp_dir / "params.yaml").is_file():
            os.rename(args.exp_dir / "params.yaml", args.exp_dir / "params_prev.yaml")
        with open(args.exp_dir / "params.yaml", "w") as f:
            yaml.dump(dict_args, f)
    exp.log_parameters(dict_args)
    args.exp = exp
    print("INFO: Args:", dict_args)
    #print("INFO: Num Patches:", (args.kw["img_size"] // args.vkw["tmp"]["patch_size"]) ** 2)
    print("INFO: Peak lr:",  (args.opt["lr"][0] * args.batch_size) / args.opt["lr"][2])

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    data = load_data(args)
    model = load_model(args)
    return (*model, *data, args)