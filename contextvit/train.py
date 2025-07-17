#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os

os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

import jupyter_black

jupyter_black.load()

import comet_ml

COMET_API_KEY = "hHeAbGuZehhIQkr1vLroWGbbT"
comet_ml.login(api_key=COMET_API_KEY)

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import functional as T
import torch._inductor.config as config
import torchvision
import torch._dynamo
torch._dynamo.config.cache_size_limit = 12

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

AMP_DTYPE = torch.float16
if "A100" in torch.cuda.get_device_name():
    AMP_DTYPE = torch.bfloat16
    cuda_device = "A100"
else:
    cuda_device = "A6000"

import matplotlib.pyplot as plt
import argparse
import shutil
from typing import Dict, Any
import json
import random
import gc
import math
import time
import psutil
from collections import defaultdict
from functools import partial
from pathlib import Path
from copy import deepcopy
from math import inf
from tqdm import tqdm as tqdm_nb
import pickle
import sys, subprocess

from PIL import Image
import numpy as np

def install_if_missing(package: str):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Ensure datasets is installed
install_if_missing("datasets")
from datasets import load_from_disk
install_if_missing("timm")
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy

# local
from modules.schedulers import SchedulerManager
from modules.context_vit_v3 import LinearContextViTv3
from modules.context_vit_v4 import LinearContextViTv4
from modules.dinov2 import DinoVisionTransformer as ViT
from modules.utils import delete_in_parallel


SEED = 4200
EPS = 1e-6
NUM_CLASSES = 1000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# assert torch.isfinite(g).all(), "g"
WORKERS = os.cpu_count() - 1

print(f"INFO: torch: {torch.__version__}, torchvision: {torchvision.__version__}")


# # Args
# In[ ]:


schedulers = {}

args = {
    #   --  ViT  --   #
    "vkw": {
        "tmp": {
            "path_drop": 0,
            "token_drop": 0.1,
            "patch_size": 16,
            "layerscale": False,
            "n_layers": 12,
            "d": 384,
            "n_heads": 6,
            "n_registers": 3,
            "attn_act": "nn.Identity",
        },
    },
    "models": {
        "ViT_n0_k0": {"arc": "vit", "n_ctx": -1, "k_ctx": -1, "vkw": "tmp"},
        "CiTv3_n64_k1": {"arc": "citv3", "n_ctx": 64, "k_ctx": 1, "vkw": "tmp"},
        "CiTv4_n64_k1": {"arc": "citv4", "n_ctx": 64, "k_ctx": 1, "vkw": "tmp"},
    },
    #   --  Optim  --   #
    "opt": {
        "lr": (1e-3, 1e-5),
        "wd": (0.05, 0.1),
        "dec_steps": 250, 
        "lr_wu": {"init": 1e-6, "steps": 8}, 
        "gc": 3,
        "ld": 0.95,
    },
    #   --  Run  --   #
    "kw": {"mixup_p": 0.95, "label_smoothing": 0.1, "img_size": 224},
    "freq": {"eval": 1, "save": inf, "stats": 150},
    #   --  DataLoader  --   #
    "num_workers": WORKERS,
    "prefetch_factor": 2,
    "batch_size": 1024,
    "compile": True,
    #   --  Exp  --   #
    "project_name": "vv.2_linearAtt_imgnet1k_224",
    "exp_cache": "/notebooks/runs/cache/IMGNET_CIT_224/A100",
    "exp_version": "V1",
    "exp_root": Path("/notebooks/runs/vv.2_linearAtt_imgnet1k_224/models"),
    "exp_info": "",
    "exp_key": "6856d8c4a59d4f019518dfcb630c4c0c",
    #"new_run": True,
    # "checkpoint_path": "/notebooks/runs/vv.1_SIE_vs_TIE_vs_EWP_3DIE_compOnline/models/V3/model-Copy1.pth",
    "update_args": ["freq"],
    "schedulers": schedulers,
    # "detect_anomaly": True,
}

# TODO explain_output = torch._dynamo.explain(bar)(torch.randn(10), torch.randn(10))
# TODO explore ctx reg
# TODO CuDA graphs


# # Utils

# In[ ]:


target = "/notebooks/.Trash-0/files/"
delete_in_parallel(target, num_threads=WORKERS)


# In[3]:


def compile_full(m):
    return torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)


def compile_dynamic(m):
    return torch.compile(m, backend="inductor", fullgraph=False, dynamic=True)


def t(t):
    return (time.perf_counter() - t) / 60


# # Dataset

# In[4]:


class HFImageDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.hf_ds = load_from_disk("/notebooks/data/imagenet_1k_resized_256")[mode]
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        example = self.hf_ds[idx]
        image = example["image"]  # a PIL.Image.Image
        label = example["label"]  # an integer
        image = self.transform(image)
        return image, label


# # Plot

# In[5]:


def denormalize_and_plot(img1, img2):
    def denormalize(img):
        if img.dim() == 4:
            img = img.squeeze(0)

        mean_tensor = torch.tensor(MEAN).view(3, 1, 1)
        std_tensor = torch.tensor(STD).view(3, 1, 1)
        img = img * std_tensor + mean_tensor

        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img

    n = img1.size(0)
    fig, axes = plt.subplots(math.ceil(n / 2), 4, figsize=(8 * 2, 10))
    axes = axes.flatten()

    for i in range(n):
        i1 = denormalize(img1[i])
        i2 = denormalize(img2[i])

        j = i * 2
        axes[j].imshow(i1)
        axes[j].axis("off")

        axes[j + 1].imshow(i2)
        axes[j + 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_data(data_loader, n):
    k = iter(data_loader)
    t = min(torch.randint(1, 5, (1,)).item(), len(data_loader) - 1)
    for _ in range(t):
        x1, l1 = next(k)
        x2, l2 = next(k)
    for _ in range(torch.randint(1, 100, (1,)).item()):
        idxs = random.sample(range(x1.size(0)), n)
    x1, x2 = x1[idxs], x2[idxs]
    l1, l2 = l1[idxs], l2[idxs]

    mixup_fn = Mixup(
        mixup_alpha=0.8,  # more mid-range mixes for a bit of hardness (λ∼Beta(0.5,0.5))
        cutmix_alpha=1.0,  # full-sized CutMix patches
        cutmix_minmax=None,  # keep Beta(1.0,1.0) sampling
        prob=1,  # apply mixup/CutMix on 50% of batches
        switch_prob=0.5,  # 50/50 chance Mixup vs. CutMix when applied
        mode="elem",  # per-sample mixing (so 'easy' and 'hard' examples interleave)
        label_smoothing=0.1,  # standard smoothing to prevent over-confidence
        num_classes=1000,  # ImageNet-1k
    )

    x1, _ = mixup_fn(x1, l1)
    x2, _ = mixup_fn(x2, l2)
    denormalize_and_plot(x1, x2)


@torch.no_grad()
def log_img(a, exp, name):
    a = a.detach().float().cpu().numpy()
    indices = np.arange(len(a))
    fig, ax = plt.subplots()
    ax.bar(indices, a)

    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()  # raw RGBA bytes
    w, h = canvas.get_width_height()  # width, height

    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    img_rgb = img_rgba[..., :3]
    exp.log_image(Image.fromarray(img_rgb), name=name)
    plt.close(fig)


# ## Debug

# In[6]:


def get_data_debug(get_args, load_data):
    args = get_args(check_args=False)
    args.print_samples = 0
    args.batch_size = 800
    args.prefetch_factor = 2
    args.num_workers = WORKERS
    args.kw["label_smoothing"] = 0.01
    args.kw["img_size"] = 128
    train_loader, val_loader, _ = load_data(args)
    return train_loader, val_loader


# In[7]:


# train_loader, val_loader = get_data_debug(get_args, load_data)


# In[8]:


# plot_data(train_loader, 10)


# In[9]:


# plot_data(val_loader, 10)


# # Model

# ## Metrics

# In[10]:


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res


# ## Init

# In[11]:


def init_model(model, args):
    regularized, not_regularized, reg_id = [], [], set()
    for n, param in model.named_parameters():
        if n.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
            reg_id.add(id(param))

    base_lr = (args.opt["lr"][0] * args.batch_size) / 512.0
    wd = args.opt["wd"][0]
    layer_decay = args.opt["ld"]
    n_layers = args.vkw[model.kw["vkw"]]["n_layers"]
    set_param_group = lambda x: {
        "params": [],
        "lr": x[0],
        "weight_decay": x[1],
        "lr_max": x[0],
    }

    # Blocks
    blocks = model.inner.model.m.blocks
    params = {}
    for i in range(len(blocks) - 1, -1, -1):
        lr = base_lr * (layer_decay ** (n_layers - i))
        params[f"reg_{i + 1}"] = set_param_group((lr, wd))
        params[f"no_reg_{i + 1}"] = set_param_group((lr, 0))
        for p in blocks[i].parameters():
            group = f"reg_{i + 1}" if id(p) in reg_id else f"no_reg_{i + 1}"
            params[group]["params"].append(p)

    # Patcher
    lr = base_lr * (layer_decay ** (n_layers + 1))
    params["reg_0"] = set_param_group((lr, wd))
    params["no_reg_0"] = set_param_group((lr, 0))
    for p in model.inner.model.m.patch_embed.parameters():
        group = "reg_0" if id(p) in reg_id else "no_reg_0"
        params[group]["params"].append(p)

    # Tokens
    params["no_reg_0"]["params"].append(model.inner.model.m.cls_token)
    params["no_reg_0"]["params"].append(model.inner.model.m.pos_embed)
    if hasattr(model.inner.model.m, "ctx_tokens"):
        params["no_reg_0"]["params"].append(model.inner.model.m.ctx_tokens)

    # Store all curr params
    seen = set()
    for g in params.values():
        for p in g["params"]:
            seen.add(id(p))

    # Inner
    params[f"reg_inner"] = set_param_group((base_lr, wd))
    params[f"no_reg_inner"] = set_param_group((base_lr, 0))
    for p in regularized + not_regularized:
        if id(p) not in seen:
            group = "reg_inner" if id(p) in reg_id else "no_reg_inner"
            params[group]["params"].append(p)

    return params


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ## Utils

# In[12]:


def get_mlp(layer_str):
    layer_dims, drop_out, norm, _ = layer_str.split(":")
    layer_dims = list(map(int, layer_dims.split("-")))
    input_dim = layer_dims[0]
    output_dim = layer_dims[-1]
    hidden_dims = layer_dims[1:-1]
    drop_out = float(drop_out)

    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim, bias=False))

        if norm != "0":
            assert norm in ("1", "2")
            curr_norm = nn.LayerNorm if norm == "1" else nn.BatchNorm1d
            layers.append(curr_norm(h_dim))
        layers.append(nn.GELU())

        if drop_out > 0:
            layers.append(nn.Dropout(drop_out))
        in_dim = h_dim

    if output_dim > 0:
        layers.append(nn.Linear(in_dim, output_dim))
    mlp = nn.Sequential(*layers)
    mlp.apply(init_weights)
    return mlp


class PushGrad(nn.Module):
    def __init__(self, optimizer, scaler, args):
        super().__init__()
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args
        self.gc = torch.tensor(self.args.opt["gc"])

    def forward(self, model, loss):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.gc)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.zero()

    def zero(self):
        self.optimizer.zero_grad(set_to_none=True)


class ModuleWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m(x)[:, 0, :]


def get_encoder(module, args, kw):
    vkw = kw["vkw"]
    return ModuleWrap(
        module(
            patch_size=args.vkw[vkw]["patch_size"],
            img_size=args.kw["img_size"],
            embed_dim=args.vkw[vkw]["d"],
            depth=args.vkw[vkw]["n_layers"],
            num_heads=args.vkw[vkw]["n_heads"],
            mlp_ratio=4,
            drop_path_uniform=True,
            drop_path_rate=args.vkw[vkw]["path_drop"],
            layerscale=args.vkw[vkw]["layerscale"],
            token_drop=args.vkw[vkw]["token_drop"],
            n_registers=args.vkw[vkw]["n_registers"],
            attn_act=eval(args.vkw[vkw]["attn_act"]),
            n_ctx=kw["n_ctx"],
            k_ctx=kw["k_ctx"],
        )
    )


# ## Modules

# ### Inner

# In[13]:


def get_context_vit(arc):
    return {"citv3" : LinearContextViTv3,
            "citv4" : LinearContextViTv4}[arc.lower()]

class InnerModel(nn.Module):
    def __init__(self, args, kw):
        super().__init__()
        arc = get_context_vit(kw["arc"]) if kw["arc"].lower() != "vit" else ViT
        self.model = get_encoder(arc, args, kw)
        self.clsf_out = nn.Linear(args.vkw["tmp"]["d"], NUM_CLASSES)
        self.criterion = SoftTargetCrossEntropy()
        self.ls = args.kw["label_smoothing"]

    def forward(self, x, labels, mixup=False):
        pred = self.clsf_out(self.model(x))
        if self.training and mixup:
            return self.criterion(pred, labels), None, None
        ce = F.cross_entropy(pred, labels, label_smoothing=self.ls)
        acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
        return ce, acc1, acc5


# ### Outer

# In[14]:


class OuterModel(nn.Module):
    def __init__(self, args, name, kw):
        super().__init__()
        self.args = args
        self.name = name
        self.kw = kw
        self.inner = InnerModel(args, kw)
        self.backward = None

    def compile_model(self):
        self.inner.compile(backend="inductor", fullgraph=True, dynamic=False)
        # self.backward = compile_dynamic(self.backward)

    def forward(self, imgs, labels, cum_stats, mixup=False):
        stats, start_time = {}, time.perf_counter()

        if self.training:
            self.backward.zero()
            ce, acc1, acc5 = self.inner(imgs, labels, mixup)
            self.backward(self.inner, ce)
            stats[f"Time/{self.name} forward pass"] = t(start_time)
        else:
            ce, acc1, acc5 = self.inner(imgs, labels)

        pref = "3 - Train Metrics" if self.training else "4 - Val Metrics"
        stats[f"{pref}/{self.name} CE "] = ce.item()
        if acc1 is not None:
            stats[f"{pref}/{self.name} Top-1"] = acc1.item()
            stats[f"{pref}/{self.name} Top-5"] = acc5.item()

        for k, v in stats.items():
            cum_stats[k].append(v)
        del stats


# # Parser

# In[15]:


def assertions_and_checks(args, dict_args):
    assert not args.new_run or args.exp_key is None

    for key, value in dict_args.items():
        if not hasattr(args, key):
            raise ValueError(f"{key} : {value} not found in args")
        setattr(args, key, value)

    assert not args.kw["img_size"] % args.vkw["tmp"]["patch_size"]
    print("Num Patches:", (args.kw["img_size"] // args.vkw["tmp"]["patch_size"]) ** 2)
    print("INFO: Peak lr:",  (args.opt["lr"][0] * args.batch_size) / 512.0)


# In[16]:


def get_args(dict_args=None, check_args=False):
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--vkw", type=dict, default={})
    parser.add_argument("--kw", type=dict, default={})
    parser.add_argument("--models", type=dict, default={})
    parser.add_argument("--opt", type=dict, default={})
    parser.add_argument("--schedulers", type=dict, default={})

    # Running
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    # Exp
    parser.add_argument("--skip_log_first_n", type=int, default=50)
    parser.add_argument("--freq", type=dict, default={})
    parser.add_argument("--exp_root", type=Path, default="")
    parser.add_argument("--exp_version", type=str, default="")
    parser.add_argument("--exp_run", type=str, default="")
    parser.add_argument("--exp_key", type=str, default=None)
    parser.add_argument("--exp_info", type=str, default="")
    parser.add_argument("--exp_cache", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--new_run", action="store_true")
    parser.add_argument("--print_samples", type=int, default=0)

    # Util
    parser.add_argument("--num_workers", type=int, default=WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--update_args", type=dict, default=[])

    args = parser.parse_known_args()[0]
    if check_args:
        assertions_and_checks(args, dict_args or {})
    return args


# # Train

# ## Setup

# ### Utils

# In[17]:


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


def reload(n=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(n)


# In[18]:
class OptScheduler(nn.Module):
    def __init__(self, optimizers: Dict, args: Any, exp=None, batch_to_step = True):
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
            lr_curr = self._set_lr_to_percent_of_max(0.15)
            #lr_curr = self._set_lr_cosine(step)
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
    
    def _set_lr_to_percent_of_max(self, factor: float):
        """Set learning rate to `percent` (0–100) of `lr_max` for all param groups."""
        assert 0.0 <= factor <= 1.0, "percent should be between 0 and 100"
        curr = 0
        for opt in self.optimizers.values():
            for pg in opt.param_groups:
                lr_max = pg.get("lr_max")
                assert lr_max is not None, "param group missing `lr_max`"
                pg["lr"] = lr_max * factor
                curr = max(curr, pg["lr"])
        return curr

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


# ### Load Data

# In[19]:


def load_data(args):
    train_transforms = create_transform(
        input_size=args.kw["img_size"],  # resize/crop to 224×224
        is_training=True,  # training augmentations
        color_jitter=0.3,  # standalone jitter if not using AA/RA
        auto_augment="rand-m9-mstd0.5-inc1",  # RandAugment policy
        interpolation="bicubic",  # resize interpolation
        re_prob=0.25,  # Random Erasing probability
        re_mode="pixel",  # Random Erasing mode
        re_count=1,  # how many erasing patches
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.kw["img_size"] * 1.15)),
            transforms.CenterCrop([args.kw["img_size"]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    train_dataset = HFImageDataset("train", train_transforms)
    val_dataset = HFImageDataset("val", val_transforms)

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

    args.batches_p_epoch = len(train_loader)
    print(f"Batches per epoch: {args.batches_p_epoch}")
    if args.print_samples > 0:
        plot_data(train_loader, args.print_samples)

    mixup_fn = Mixup(
        mixup_alpha=0.8,  # more mid-range mixes for a bit of hardness (λ∼Beta(0.5,0.5))
        cutmix_alpha=1.0,  # full-sized CutMix patches
        cutmix_minmax=None,  # keep Beta(1.0,1.0) sampling
        prob=1,  # apply mixup/CutMix on 50% of batches
        switch_prob=0.5,  # 50/50 chance Mixup vs. CutMix when applied
        mode="batch",  # per-sample mixing (so 'easy' and 'hard' examples interleave)
        label_smoothing=args.kw["label_smoothing"],
        num_classes=1000,  # ImageNet-1k
    )

    return train_loader, val_loader, mixup_fn


# ### Load Model

# In[20]:


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


# ### Prep Training

# In[21]:


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


# ## Eval

# In[22]:


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


# ## Train Main

# In[23]:


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


# # Inductor Config

# In[24]:


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


# In[25]:


set_inductor_config()


# # Start Training

# ## Args

# ## Go

# In[ ]:


#assert False


# In[ ]:


start_training(args)


# # End
