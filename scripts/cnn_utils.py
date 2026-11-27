"""PyTorch streaming dataset and CNN image transforms (torch imported first for Windows DLL safety)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cnn_transforms(train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FrothImageDataset(Dataset):
    """Streams froth images from disk one by one without preloading the dataset into RAM."""

    def __init__(
        self,
        df: pd.DataFrame,
        project_root: Path,
        label_col: str = "category_idx",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.paths = [project_root / p for p in df["relative_path"].tolist()]
        self.labels = df[label_col].astype(int).tolist() if label_col in df.columns else [0] * len(df)
        self.image_ids = df["image_id"].astype(int).tolist()
        self.transform = transform or get_cnn_transforms(train=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        path = self.paths[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, self.labels[idx], self.image_ids[idx]
