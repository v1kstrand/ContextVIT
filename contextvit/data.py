
from torch.utils.data import Dataset
from datasets import load_from_disk


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
