from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.transforms.functional import resize


def get_heatmaps(
    batch: Tensor,
    model: torch.nn.Module,
    class_indices: Optional[List[int]] = None,
) -> Tensor:
    """Gets heatmaps for the batch, which correspond to class indices for each image

    Args:
        batch: Batch of image tensors.
        model: PyTorch classification model.
        class_indices: List of class indices for heatmaps. The default value is None, so it means all classes.

    Returns:
        Batch of normalized heatmaps for specified class indices.

    Raises:
        ValueError: If class_indices length is zero.
    """
    if isinstance(class_indices, list) and len(class_indices) == 0:
        raise ValueError("class_indices should be the list with at least one index or None object.")
    class_indices = torch.LongTensor(class_indices)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])  # Exclude the last two layers (GAP and FC)

    features = feature_extractor(batch)
    heatmaps = model.fc(features.permute(0, 2, 3, 1))
    heatmaps = heatmaps.permute(0, 3, 1, 2)
    if class_indices is not None:
        heatmaps = heatmaps[:, class_indices, :, :]

    # Normalize heatmaps.
    heatmaps = torch.where(heatmaps > 0, heatmaps, 0) / torch.max(heatmaps)
    return heatmaps


def get_images_with_heatmaps(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    alpha: int = 0.5,
):
    """Adds heatmaps to the batch.

    Args:
        images: Batch of original RGB images in [0, 1] segment (not transformed for model).
        heatmaps: Batch of heatmaps with predefined class_indices.
        alpha: Parameter for blending images and heatmaps.

    Returns:
        Batch of images with heatmaps. It has new batch size which equals to batch_size * class_indices.
    """
    # Define necessary parameters for images and heatmaps transformation.
    cmap = plt.get_cmap("jet")
    batch_size = images.size(0)
    class_indices = heatmaps.size(1)
    channels = 3
    heatmap_size = [heatmaps.shape[2], heatmaps.shape[3]]
    img_size = [images.shape[2], images.shape[3]]

    # Modify images array for broadcasting.
    images = images.repeat((class_indices, 1, 1, 1))  # repeat along batch_size dimension.

    # Modify heatmaps using the colormap for broadcasting.
    heatmaps = heatmaps.permute(1, 0, 2, 3)  # Permute heatmaps corresponding to repeated images.
    heatmaps = heatmaps.reshape(batch_size * class_indices, *heatmap_size)
    heatmaps = cmap(heatmaps)
    heatmaps = heatmaps[:, :, :, :channels]  # Get 3 channels only instead of 4.
    heatmaps = torch.from_numpy(heatmaps)
    heatmaps = heatmaps.permute(0, 3, 1, 2)
    heatmaps = resize(heatmaps, img_size)

    # Add heatmaps to images.
    images_with_heatmaps = alpha * heatmaps + (1 - alpha) * images

    return images_with_heatmaps
