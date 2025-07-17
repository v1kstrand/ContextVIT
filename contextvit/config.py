import os
from pathlib import Path
from math import inf
import torch
import argparse

import torch._dynamo
torch._dynamo.config.cache_size_limit = 12
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

# Constants
SEED = 4200
EPS = 1e-6
NUM_CLASSES = 1000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WORKERS = os.cpu_count() - 1


assert "A100" in torch.cuda.get_device_name()
AMP_DTYPE = torch.bfloat16
cuda_device = "A100"

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



def assertions_and_checks(args, dict_args):
    assert not args.new_run or args.exp_key is None

    for key, value in dict_args.items():
        if not hasattr(args, key):
            raise ValueError(f"{key} : {value} not found in args")
        setattr(args, key, value)

    assert not args.kw["img_size"] % args.vkw["tmp"]["patch_size"]
    print("Num Patches:", (args.kw["img_size"] // args.vkw["tmp"]["patch_size"]) ** 2)
    print("INFO: Peak lr:",  (args.opt["lr"][0] * args.batch_size) / 512.0)



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
