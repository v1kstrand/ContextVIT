import random
import math
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform, Mixup
from datasets import load_from_disk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .config import MEAN, STD, WORKERS
from .utils import t


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
