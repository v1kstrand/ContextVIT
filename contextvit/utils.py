import time
import torch
import gc

def t(t):
    return (time.perf_counter() - t) / 60

def reload(n=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(n)
