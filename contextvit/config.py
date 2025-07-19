import os
os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

from pathlib import Path
from math import inf
import torch
import argparse

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

if "A100" in torch.cuda.get_device_name():
    AMP_DTYPE = torch.bfloat16
    cuda_device = "A100"
else:
    AMP_DTYPE = torch.float16
    cuda_device = torch.cuda.get_device_name()

def set_config():
    dynamo_config = torch._dynamo.config
    dynamo_config.compiled_autograd = True
    dynamo_config.capture_scalar_outputs = False
    torch._dynamo.config.cache_size_limit = 12
    
    inductor_config =  torch._inductor.config
    # spend longer tuning for best Triton kernels
    inductor_config.max_autotune = True
    # fuse pointwise ops into matrix-kernel epilogues
    inductor_config.epilogue_fusion = True
    # pad sizes for better tensor-core alignment
    inductor_config.shape_padding = True
    # Allow fusing mul+add into a single FMA
    inductor_config.cpp.enable_floating_point_contract_flag = "fast"

    inductor_config.b2b_gemm_pass = True

    # Turn on unsafe-math for speed (be aware: may break strict IEEE)
    inductor_config.cpp.enable_unsafe_math_opt_flag = True

    # Increase horizontal fusion width if you have many small pointwise ops
    inductor_config.cpp.max_horizontal_fusion_size = 32
    inductor_config.cpp.fallback_scatter_reduce_sum = False
    inductor_config.cpp.gemm_max_k_slices = 4  # 2
    inductor_config.cpp.gemm_cache_blocking = "4,1,8"
    inductor_config.cpp.gemm_thread_factors = "4,4,2"

    # ──── 3) Tiling & Fusion ────────────────────────────────────────────────────────
    # allow up to 3D tiling (more parallelism)
    inductor_config.triton.max_tiles = 3
    # favor higher-dim tiles for cleaner index math
    inductor_config.triton.prefer_nd_tiling = True
    # let pointwise fuse through tiles
    inductor_config.triton.tiling_prevents_pointwise_fusion = False
    # allow reduction fusion after tiling
    inductor_config.triton.tiling_prevents_reduction_fusion = False

    # ──── 4) Reduction Strategies ───────────────────────────────────────────────────
    inductor_config.triton.persistent_reductions = True  # keep reduction state in shared memory
    inductor_config.triton.cooperative_reductions = True  # cross-block sync for small outputs
    inductor_config.triton.multi_kernel = 1  # enable multi-kernel reduction search

    # ──── 5) Numeric & Codegen Tweaks ──────────────────────────────────────────────
    inductor_config.triton.divisible_by_16 = True  # hint for vectorized loads/stores
    inductor_config.triton.spill_threshold = 16  # allow up to 16 register spills
    inductor_config.triton.codegen_upcast_to_fp32 = (
        True  # upcast FP16/BF16 math to FP32 in-kernel
    )

    # 2) Host‐compile optimizations
    inductor_config.cuda.compile_opt_level = "-O3"
    inductor_config.cuda.enable_cuda_lto = True
    inductor_config.cuda.use_fast_math = True

    # 3) CUTLASS autotune settings
    inductor_config.cuda.cutlass_max_profiling_configs = None  # tune _all_ kernels
    inductor_config.cuda.cutlass_backend_min_gemm_size = 32 * 32 * 32  # small GEMMs → Triton
    inductor_config.cuda.cutlass_op_denylist_regex = "pingpong"  # filter unstable kernels
    print("Config Set")



def assertions_and_checks(args, dict_args):
    assert not args.new_run or args.exp_key is None

    for key, value in dict_args.items():
        if not hasattr(args, key):
            raise ValueError(f"{key} : {value} not found in args")
        setattr(args, key, value)

    assert not args.kw["img_size"] % args.vkw["tmp"]["patch_size"]
    print("Num Patches:", (args.kw["img_size"] // args.vkw["tmp"]["patch_size"]) ** 2)
    print("INFO: Peak lr:",  (args.opt["lr"][0] * args.batch_size) / args.opt["lr"][2])



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


