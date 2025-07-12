import time
import torch
import gc


def compile_full(m):
    return torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)


def compile_dynamic(m):
    return torch.compile(m, backend="inductor", fullgraph=False, dynamic=True)


def t(start):
    return (time.perf_counter() - start) / 60


def reload(delay=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(delay)
