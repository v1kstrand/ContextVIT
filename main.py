from math import inf
import os
from pathlib import Path

from contextvit.config import set_config
set_config()
from contextvit.training import start_training


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
        },
    },
    "models": {
        "ViT_n0_k0": {"arc": "vit", "n_ctx": -1, "k_ctx": -1, "vkw": "tmp"},
        "CiTv3_n64_k1": {"arc": "citv3", "n_ctx": 64, "k_ctx": 1, "vkw": "tmp"},
        "CiTv4_n64_k1": {"arc": "citv4", "n_ctx": 64, "k_ctx": 1, "vkw": "tmp"},
    },
    #   --  Optim  --   #
    "opt": {
        "lr": (1e-3, 1e-5, 1024),
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
    "num_workers": os.cpu_count() - 1,
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
    # "detect_anomaly": True,
}

start_training(args)
