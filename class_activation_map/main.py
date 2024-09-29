import os
from itertools import product
from pathlib import Path
from typing import List

import click
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from class_activation_map.cam_processing import get_heatmaps, get_images_with_heatmaps
from class_activation_map.data_processing import ImageDataset

def save_images(images: torch.Tensor, img_names: List[str | os.PathLike], save_dir: str | os.PathLike) -> None:
    """Saves images.

    Args:
        images: Batch of images (batch_size, height, width, channels).
        img_names: List of image names or image paths.
        save_dir: Directory for new images.
        
    Raises:
        ValueError: If the number of images does not match the number of image_names.
    """
    if images.shape[0] != len(img_names):
        raise ValueError(
            f"The number of images does not match the number of image names:"
            f"{images.shape[0]} and {len(img_names)} respectively."
        )
    save_dir = Path(save_dir)
    for img_name, img in zip(img_names, images):
        img_path = save_dir / Path(img_name).name
        image = to_pil_image(img)
        image.save(img_path)


@click.command()
@click.argument("img-dir", type=click.Path(dir_okay=True, exists=True))
@click.argument("save-dir", type=click.Path(dir_okay=True, exists=False))
@click.option("--batch-size", type=int, default=32)
@click.option("-v", "--verbose", is_flag=True, default=False, show_default=True, help="Progress bar on/off.")
def save_heatmaps(img_dir: str | os.PathLike, save_dir: str | os.PathLike, batch_size: int, verbose: bool):
    """Saves heatmaps for images in IMG_DIR to SAVE_DIR."""
    img_dir = Path(img_dir)

    # Define the model and turn on eval options.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model.to(device)

    # Define the dataloader.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = ImageDataset(img_dir, transform)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Batch iteration.
    disable = not verbose
    class_indices = [207, 251, 271, 923]
    progress_bar = tqdm(total=len(ds), disable=disable)
    step = data_loader.batch_size
    for batch in data_loader:
        images, img_paths = batch["img_tensor"], batch["img_path"]
        img_tensors = normalize(images)
        img_tensors = img_tensors.to(device)
        # Get heatmaps.
        heatmaps = get_heatmaps(img_tensors, model, class_indices)
        # Add heatmaps to images.
        images = get_images_with_heatmaps(images.cpu(), heatmaps.cpu())
        # Modify image paths corresponding to class indices.
        cam_img_paths = []
        for class_id, img_path in product(class_indices, img_paths):
            img_path = Path(img_path)
            img_path = img_path.with_stem(img_path.stem + "_" + str(class_id))
            cam_img_paths.append(img_path)
        # Write images to save_dir.
        save_images(images, cam_img_paths, save_dir)
        progress_bar.update(step)
