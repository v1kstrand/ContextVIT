import os
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from timm.data import Mixup
import time
import torch
import subprocess
import sys
from contextvit.config import MEAN, STD
import random
import gc

def reset(n=1):
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(n)
    
def get_data_debug(get_args, load_data):
    args = get_args(check_args=False)
    args.print_samples = 0
    args.batch_size = 800
    args.prefetch_factor = 2
    args.num_workers = os.cpu_count() - 1
    args.kw["label_smoothing"] = 0.01
    args.kw["img_size"] = 128
    train_loader, val_loader, _ = load_data(args)
    return train_loader, val_loader


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
    _, axes = plt.subplots(math.ceil(n / 2), 4, figsize=(8 * 2, 10))
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

def t(t):
    return (time.perf_counter() - t) / 60

def install_if_missing(package: str):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

class PyDrive():
    def __init__(self):
        GoogleAuth, GoogleDrive = self.init()
        self.gauth = GoogleAuth()
        self.gauth.CommandLineAuth()   
        self.drive = GoogleDrive(self.gauth)
        
    def init(self):
        install_if_missing("PyDrive2")
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive   
        return GoogleAuth, GoogleDrive   

    def move_to_gdrive(self, save_name, file_path):
        gfile = self.drive.CreateFile({'title': save_name})
        gfile.SetContentFile(file_path, file_path)
        gfile.Upload()
        print(f"Successfully uploaded {file_path} and saved as {save_name}")
        
    def download_from_gdrive(self, file_id: str, dest_path: str):
        """
        Download a file from Google Drive to the local filesystem.

        Args:
            file_id (str): The ID of the file on Drive (e.g., '1AbCdEfGhIjKlMnOp').
            dest_path (str): Local path where to save the downloaded file.
        """
        gfile = self.drive.CreateFile({'id': file_id})
        gfile.GetContentFile(dest_path)
        print(f"Successfully downloaded file to {dest_path}")

def parallel_collect_paths(root_path: str, num_threads: int = 8):
    """
    Walk the directory tree under `root_path` in parallel, collecting:
      - all_files: a list of full paths to every file
      - all_dirs:  a list of full paths to every directory (excluding root_path)
    Finally, the caller can append root_path to all_dirs if needed.

    Args:
        root_path (str): The top directory to traverse.
        num_threads (int): How many worker threads to spawn.

    Returns:
        all_files (List[str]) : full paths of all files under root_path
        all_dirs  (List[str]) : full paths of all subdirectories under root_path
    """
    # Thread‐safe queue of directories to process
    q = Queue()
    q.put(root_path)

    # Shared lists (protected by `lock`)
    all_files = []
    all_dirs = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                dirpath = q.get_nowait()
            except Empty:
                # No more directories to process
                return

            try:
                # List all entries in this directory
                entries = os.listdir(dirpath)
            except Exception:
                # If we can’t read this directory for any reason, skip it
                q.task_done()
                continue

            for name in entries:
                full_path = os.path.join(dirpath, name)
                if os.path.isdir(full_path):
                    # Record this directory, then schedule it for further walking
                    with lock:
                        all_dirs.append(full_path)
                    q.put(full_path)
                else:
                    # It’s a file; record it
                    with lock:
                        all_files.append(full_path)

            q.task_done()

    # Spawn worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    # Wait until every directory has been processed
    q.join()

    # (Optional) Join threads (they should all exit once queue is empty)
    for t in threads:
        t.join()

    return all_files, all_dirs

def delete_in_parallel(root_path: str, num_threads: int = 8):
    # 1) Gather everything in parallel
    files_list, dirs_list = parallel_collect_paths(root_path, num_threads=num_threads)

    # 2) Delete all files in parallel
    def _remove_file(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to remove file {path!r}: {e}")

    if files_list:
        print(f"Deleting {len(files_list)} files ...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(_remove_file, files_list))

    # 3) Delete all directories in parallel (deepest first)
    # Sort by depth: deeper dirs (more separators) first
    dirs_sorted = sorted(dirs_list, key=lambda p: -p.count(os.sep))

    def _remove_dir(path):
        try:
            os.rmdir(path)
        except Exception as e:
            print(f"Failed to remove directory {path!r}: {e}")

    if dirs_sorted:
        print(f"Deleting {len(dirs_sorted)} directories ...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(_remove_dir, dirs_sorted))