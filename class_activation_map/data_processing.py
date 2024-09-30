import os
from pathlib import Path
from typing import Tuple, Callable, Dict, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Dataset for images iteration.

    Attributes:
        img_dir: Directory with images.
        transform: The function for image transformation.
    """

    img_types: Tuple[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    def __init__(self, img_dir: str | os.PathLike, transform: Callable) -> None:
        self.img_dir = Path(img_dir)
        self.img_paths = [img_path for img_path in self.img_dir.glob("*") if img_path.suffix.lower() in self.img_types]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)
        return {"img_path": str(img_path), "img_tensor": img_tensor}
