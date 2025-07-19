import torch
import time
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from .config import NUM_CLASSES
from .metrics import accuracy
from modules.utils import t
from modules.context_vit_v3 import LinearContextViTv3
from modules.context_vit_v4_1 import LinearContextViTv4
from modules.dinov2 import DinoVisionTransformer as ViT


def get_context_vit(arc):
    return {"citv3" : LinearContextViTv3,
            "citv4" : LinearContextViTv4}[arc.lower()]
            #"citv5" : TiledContextViTv5}[arc.lower()]
            

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
            n_ctx=kw["n_ctx"],
            k_ctx=kw["k_ctx"],
            #tile_size=kw["tile_size"],
            #comp_size=kw["comp_size"],
        )
    )