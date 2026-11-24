"""Image-folder dataset and transforms for training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageFolderDataset(Dataset):
    """All .jpg/.jpeg/.png files directly inside ``root`` (case-insensitive)."""

    def __init__(self, root: str | Path, transform=None):
        self.files = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in _EXTENSIONS)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image) if self.transform else image


def get_transform(size: int, crop: bool, final_size: int):
    """Resize the short side to ``size`` (if > 0), then random-crop or resize to ``final_size``."""
    steps = []
    if size > 0:
        steps.append(transforms.Resize(size))
    steps.append(transforms.RandomCrop(final_size) if crop else transforms.Resize((final_size, final_size)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
