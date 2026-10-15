"""Preprocessing pipelines for VNDCash.

Responsibilities (Member A):
- Provide standardized image resize, color normalization, and tensor conversion.
- Support inference, validation, and test workflows.
- Provide denormalization helper for visualization in notebooks and reports.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Default ImageNet normalization stats
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (224, 224)


def get_eval_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> T.Compose:
    """Returns evaluation/inference transform pipeline (Resize -> ToTensor -> Normalize)."""
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> torch.Tensor:
    """Preprocesses a single image for model forward pass.

    Args:
        image: File path string, PIL Image, or NumPy uint8 array.
        image_size: Target (height, width) or tuple.
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        Tensor of shape [1, 3, H, W] ready for model input.
    """
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    transform = get_eval_transforms(image_size=image_size, mean=mean, std=std)
    tensor = transform(pil_img)  # Shape [3, H, W]
    return tensor.unsqueeze(0)   # Shape [1, 3, H, W]


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> np.ndarray:
    """Inverts ImageNet normalization on a [3, H, W] or [1, 3, H, W] tensor for plotting.

    Returns:
        uint8 numpy array of shape [H, W, 3] in range [0, 255].
    """
    t = tensor.detach().cpu().clone()
    if t.ndim == 4:
        t = t.squeeze(0)

    mean_tensor = torch.tensor(mean, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=t.dtype, device=t.device).view(3, 1, 1)

    t = t * std_tensor + mean_tensor
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr
