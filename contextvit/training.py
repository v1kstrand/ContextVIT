
import time
import random
from collections import defaultdict
from tqdm.auto import tqdm as tqdm_nb
import torch

import comet_ml
COMET_API_KEY = "hHeAbGuZehhIQkr1vLroWGbbT"
comet_ml.login(api_key=COMET_API_KEY)

from .config import AMP_DTYPE
from .train_utils import save_model
from .train_prep import prep_training
from modules.utils import t

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

def train_loop(modules, exp):
    models, _, _, sched, opt_sched, train_loader, val_loader, mixup_fn, args = modules
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
        train_loop(modules, exp)
    finally:
        exp.end()


