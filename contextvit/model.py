from collections import defaultdict
import math
import torch
import time
from torch import nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from .utils import t
from .config import NUM_CLASSES
from modules.context_vit_v3 import LinearContextViTv3
from modules.context_vit_v4 import LinearContextViTv4
from modules.dinov2 import DinoVisionTransformer as ViT


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



@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


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

def get_context_vit(arc):
    return {"citv3": LinearContextViTv3, "citv4": LinearContextViTv4}[arc.lower()]


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