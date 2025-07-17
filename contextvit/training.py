import shutil
import math
import json
import os
import time
import random
from collections import defaultdict
from pathlib import Path
from math import inf
from tqdm.auto import tqdm as tqdm_nb

import torch
from torch import nn
import torch._inductor.config as config

import comet_ml

COMET_API_KEY = "hHeAbGuZehhIQkr1vLroWGbbT"
comet_ml.login(api_key=COMET_API_KEY)

from .config import AMP_DTYPE, cuda_device, get_args
from .data import load_data
from .model import OuterModel, PushGrad, init_model, accuracy
from .utils import t, reload
from modules.schedulers import SchedulerManager


class OptScheduler(nn.Module):
    def __init__(self, optimizers, args, exp=None, batch_to_step = True):
        super().__init__()
        self.optimizers = optimizers
        
        factor = args.batches_p_epoch if batch_to_step else 1
        self.wu_steps = args.opt["lr_wu"]["steps"] * factor 
        self.wu_start = args.opt["lr_wu"]["init"] 
        self.dec_steps = args.opt["dec_steps"] * factor
        self.lr_end = args.opt["lr"][1]
        self.wd_start = args.opt["wd"][0]
        self.wd_end = args.opt["wd"][1]
        self.curr_step = 1
        self.exp = exp
        print(f"INFO: wu_steps: {self.wu_steps}, dec_steps: {self.dec_steps}")

    def forward(self, step: int = None):
        """
        Call at each training step to update LRs.
        If `step` is provided, uses that instead of internal counter.
        """
        step = step if step is not None else self.curr_step
        if step <= self.wu_steps:
            lr_curr = self._set_warm_up(step)
            wd_curr = self.wd_start
        else:
            lr_curr = self._set_lr_cosine(step)
            wd_curr = self._set_wd_cosine(step)
        self.curr_step += 1

        if self.exp is not None:
            self.exp.log_metric("General/Val - LR", lr_curr, step=step)
            self.exp.log_metric("General/Val - WD", wd_curr, step=step)

    def _set_warm_up(self, step: int):
        """Linearly ramp LR from wu_start → lr_max over wu_steps."""
        curr = 0
        alpha = step / float(self.wu_steps)
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                lr_max = pg.get("lr_max")
                assert lr_max is not None, "param group missing `lr_max`"
                pg["lr"] = self.wu_start + alpha * (lr_max - self.wu_start)
                curr = max(curr, pg["lr"])
        return curr

    def _set_lr_cosine(self, step: int):
        """Cosine-decay LR from lr_max → lr_end over dec_steps."""
        curr = 0
        dec_step = step - self.wu_steps
        if dec_step >= self.dec_steps:
            for opt in self.optimizers.values():
                for pg in opt.param_groups:
                    pg["lr"] = self.lr_end
            return self.lr_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                lr_max = pg.get("lr_max")
                assert lr_max is not None, "param group missing `lr_max`"
                pg["lr"] = self.lr_end + (lr_max - self.lr_end) * cos_factor
                curr = max(curr, pg["lr"])
        return curr

    def _set_wd_cosine(self, step: int):
        """Cosine-decay LR from lr_max → lr_end over dec_steps."""
        dec_step = step - self.wu_steps
        if dec_step >= self.dec_steps:
            for opt in self.optimizers.values():
                for pg in opt.param_groups:
                    if pg["weight_decay"] != 0:
                        pg["weight_decay"] = self.wd_end
            return self.wd_end

        cos_factor = 0.5 * (1 + math.cos(math.pi * dec_step / float(self.dec_steps)))
        new_wd = self.wd_end + (self.wd_start - self.wd_end) * cos_factor
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                if pg["weight_decay"] == 0:
                    continue
                pg["weight_decay"] = new_wd
        return new_wd

    def state_dict(self):
        return {
            "wu_steps": self.wu_steps,
            "wu_start": self.wu_start,
            "dec_steps": self.dec_steps,
            "lr_end": self.lr_end,
            "curr_step": self.curr_step,
            "wd_start": self.wd_start,
            "wd_end": self.wd_end,
        }

    def load_state_dict(self, sd):
        for k, v in sd.items():
            setattr(self, k, v)

def save_model(modules, name):
    model, optimizer, scaler, sched, opt_sched, *_, args = modules
    save_path = args.exp_dir / (name + ".pth")
    if save_path.exists():
        shutil.copy(save_path, args.exp_dir / (name + "_prev.pth"))

    torch.save(
        {
            "model": {n: m.state_dict() for n, m in model.items()},
            "optimizer": {n: o.state_dict() for n, o in optimizer.items()},
            "scaler": {n: s.state_dict() for n, s in scaler.items()},
            "scheduler": sched.state_dict(),
            "opt_scheduler": opt_sched.state_dict(),
        },
        save_path,
    )

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
    schedulers = SchedulerManager()
    for target_object, scheduler_list in args.schedulers.items():
        for scheduler in scheduler_list:
            scheduler.target_object = locals()[target_object]
            scheduler_name = f"General/scheduler_{target_object}_{scheduler.param_name}"
            schedulers.add_scheduler(scheduler_name, scheduler)
            print("add scheduler:", target_object)

    if args.checkpoint_path:
        print("INFO: press enter to load from checkpoint")
        assert not input("press enter to load from checkpoint")

    checkpoint_path = args.checkpoint_path or (
        args.exp_dir / "model.pth" if (args.exp_dir / "model.pth").is_file() else None
    )

    if checkpoint_path and not args.new_run:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except:
            assert checkpoint_path == args.exp_dir / "model.pth", "Loading failed"
            checkpoint = torch.load(args.exp_dir / "model_prev.pth", map_location="cpu")

        for n in models:
            models[n].load_state_dict(checkpoint["model"][n])
            models[n].backward.optimizer.load_state_dict(checkpoint["optimizer"][n])
            models[n].backward.scaler.load_state_dict(checkpoint["scaler"][n])
        schedulers.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("opt_scheduler"):
            opt_scheduler.load_state_dict(checkpoint["opt_scheduler"])
    else:
        print("Initializing new model")

    if args.compile:
        print("INFO: Compiling model")
        for m in models.values():
            m.compile_model()

    return models, optimizers, scalers, schedulers, opt_scheduler


def prep_training(dict_args, exp):
    reload(0)
    pref = dict_args["exp_root"].relative_to("/notebooks/runs")
    pref = pref.as_posix().replace("/", "-")
    exp.set_name(f"{pref}-{dict_args['exp_version']}")
    print(f"Setting up experiment: {exp.get_name()}, key: {exp.get_key()}")

    # Args
    args = get_args(dict_args, check_args=True)
    args.exp_dir = args.exp_root / args.exp_version
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    # Compiling cache
    if args.exp_cache:
        assert cuda_device in str(args.exp_cache)
        print(f"INFO: TORCHINDUCTOR_CACHE_DIR = {args.exp_cache}")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.exp_cache
    else:
        cache_dir = args.exp_dir / Path("cache") / Path(cuda_device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    # Set config
    if (args.exp_dir / "params.json").is_file() and not args.new_run:
        with open(args.exp_dir / "params.json", "r") as f:
            exp_args = json.load(f)

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
        print(f"Loading config from file: {args.exp_dir / 'params.json'}")

    dict_args = {k: v for k, v in sorted(vars(args).items())}
    dict_args["exp_root"] = str(dict_args["exp_root"])
    dict_args["exp_dir"] = str(dict_args["exp_dir"])
    "schedulers" in dict_args and dict_args.pop("schedulers")

    if not (args.exp_dir / "params.json").is_file() or args.new_run:
        if (args.exp_dir / "params.json").is_file():
            os.rename(args.exp_dir / "params.json", args.exp_dir / "params_prev.json")

        with open(args.exp_dir / "params.json", "w") as f:
            json.dump(dict_args, f, indent=4)
    exp.log_parameters(dict_args)
    args.exp = exp
    print("Args:", dict_args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    data = load_data(args)
    model = load_model(args)
    return (*model, *data, args)


@torch.no_grad()
def validate(model, loader, name, curr_step, args, exp):
    model.eval()
    stats = defaultdict(list)
    curr_epoch = curr_step // args.batches_p_epoch
    args.pb.set_description_str(f"Validating {name} - Epoch: {curr_epoch}")

    for step, data in enumerate(loader):
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            imgs, labels = map(lambda d: d.cuda(non_blocking=True), data)
            model.forward(imgs, labels, stats)
        args.pb.set_postfix_str(f"Step: {step} / {len(loader)}")

    for k, v in stats.items():
        exp.log_metric(k, sum(v) / len(v), step=curr_step)

def start_training_loop(modules, exp):
    models, opt, _, sched, opt_sched, train_loader, val_loader, mixup_fn, args = modules
    args.pb = pb = tqdm_nb(range(args.epochs), total=args.epochs)
    stats = {name: defaultdict(list) for name in models}
    next_stats, init_run = sched.curr_step + args.freq["stats"] * 2, True

    for _ in pb:

        # -- Epoch Start --

        next_epoch = sched.curr_step + len(train_loader)
        pb.set_description_str(f"Epoch: {sched.curr_epoch} - Next @ {next_epoch}")
        epoch_time, curr_epoch = time.perf_counter(), sched.curr_epoch
        batch_time = stats_time = None

        models.train()
        for step, data in enumerate(train_loader, start=sched.curr_step):
            if batch_time is not None:
                exp.log_metric("General/Batch time", t(batch_time), step=step)

            opt_sched()
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                imgs, labels = map(lambda d: d.cuda(non_blocking=True), data)
                mixup = False
                if args.kw["mixup_p"] >= random.random():
                    mixup = True
                    imgs, labels = mixup_fn(imgs, labels)
                for name, model in models.items():
                    model.forward(imgs, labels, stats[name], mixup)

            if step and step % args.freq["stats"] == 0:
                if stats_time is not None:
                    for s in stats.values():
                        for k, v in s.items():
                            exp.log_metric(k, sum(v) / len(v), step=step)
                    exp.log_metric("General/Stat time", t(stats_time), step=step)
                    sched.curr_step += 1
                    save_model(modules, "model")
                del stats

                stats_time = time.perf_counter()
                stats = {name: defaultdict(list) for name in models}
                next_stats = sched.curr_step + args.freq["stats"]
            else:
                sched.curr_step += 1

            pb.set_postfix_str(f"Step: {step} - Next Stats: {next_stats}")
            batch_time = time.perf_counter()

        # -- Epoch End --

        sched.step(exp)
        sched.curr_epoch += 1

        if not init_run:
            exp.log_metric("General/Epoch time", t(epoch_time), step=curr_epoch)
        init_run = False

        if args.freq["save"] == 1 or (
            curr_epoch and curr_epoch % args.freq["save"] == 0
        ):
            save_model(modules, name=f"model_{curr_epoch}")

        if args.freq["eval"] == 1 or (
            curr_epoch and curr_epoch % args.freq["eval"] == 0
        ):
            val_time = time.perf_counter()
            for name, model in models.items():
                validate(model, val_loader, name, step, args, exp)
            exp.log_metric("General/Val time", t(val_time), step=curr_epoch)


def start_training(dict_args):
    exp = comet_ml.start(
        api_key=COMET_API_KEY,
        project_name=dict_args["project_name"],
        experiment_key=dict_args.get("exp_key", None),
    )
    try:
        modules = prep_training(dict_args, exp)
        start_training_loop(modules, exp)
    finally:
        exp.end()



def set_inductor_config():
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.capture_scalar_outputs = False
    # spend longer tuning for best Triton kernels
    config.max_autotune = True
    # fuse pointwise ops into matrix-kernel epilogues
    config.epilogue_fusion = True
    # pad sizes for better tensor-core alignment
    config.shape_padding = True
    # Allow fusing mul+add into a single FMA
    config.cpp.enable_floating_point_contract_flag = "fast"

    config.b2b_gemm_pass = True

    # Turn on unsafe-math for speed (be aware: may break strict IEEE)
    config.cpp.enable_unsafe_math_opt_flag = True

    # Increase horizontal fusion width if you have many small pointwise ops
    config.cpp.max_horizontal_fusion_size = 32
    config.cpp.fallback_scatter_reduce_sum = False
    config.cpp.gemm_max_k_slices = 4  # 2
    config.cpp.gemm_cache_blocking = "4,1,8"
    config.cpp.gemm_thread_factors = "4,4,2"

    # ──── 3) Tiling & Fusion ────────────────────────────────────────────────────────
    # allow up to 3D tiling (more parallelism)
    config.triton.max_tiles = 3
    # favor higher-dim tiles for cleaner index math
    config.triton.prefer_nd_tiling = True
    # let pointwise fuse through tiles
    config.triton.tiling_prevents_pointwise_fusion = False
    # allow reduction fusion after tiling
    config.triton.tiling_prevents_reduction_fusion = False

    # ──── 4) Reduction Strategies ───────────────────────────────────────────────────
    config.triton.persistent_reductions = True  # keep reduction state in shared memory
    config.triton.cooperative_reductions = True  # cross-block sync for small outputs
    config.triton.multi_kernel = 1  # enable multi-kernel reduction search

    # ──── 5) Numeric & Codegen Tweaks ──────────────────────────────────────────────
    config.triton.divisible_by_16 = True  # hint for vectorized loads/stores
    config.triton.spill_threshold = 16  # allow up to 16 register spills
    config.triton.codegen_upcast_to_fp32 = (
        True  # upcast FP16/BF16 math to FP32 in-kernel
    )

    # 2) Host‐compile optimizations
    config.cuda.compile_opt_level = "-O3"
    config.cuda.enable_cuda_lto = True
    config.cuda.use_fast_math = True

    # 3) CUTLASS autotune settings
    config.cuda.cutlass_max_profiling_configs = None  # tune _all_ kernels
    config.cuda.cutlass_backend_min_gemm_size = 32 * 32 * 32  # small GEMMs → Triton
    config.cuda.cutlass_op_denylist_regex = "pingpong"  # filter unstable kernels
    print("Config Set")

