"""Data augmentation pipelines for VNDCash.

Responsibilities (Member A):
- Provide robust data augmentation pipelines strictly for the training split.
- Enforce the design rule: Data augmentation must NEVER be applied to val or test.
- Include realistic banknote transformations: rotation, scale, color jitter, blur.
"""

from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = (224, 224)


def get_train_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    aug_cfg: Optional[Dict] = None,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD
) -> T.Compose:
    """Builds augmentation pipeline for training data only.

    Includes:
    - Resize
    - Random Horizontal Flip
    - Random Rotation (-15 to +15 deg)
    - Color Jitter (simulates camera exposure and lighting shifts)
    - Random Affine (scale & translation)
    - Gaussian Blur (simulates camera blur / out of focus)
    - Normalization
    """
    cfg = aug_cfg or {}
    hflip_prob = cfg.get("horizontal_flip_prob", 0.5)
    rotation_deg = cfg.get("rotation_degrees", 15)
    brightness = cfg.get("brightness_jitter", 0.2)
    contrast = cfg.get("contrast_jitter", 0.2)
    saturation = cfg.get("saturation_jitter", 0.2)
    affine_scale = tuple(cfg.get("affine_scale", [0.9, 1.1]))
    blur_prob = cfg.get("gaussian_blur_prob", 0.2)

    transforms_list = [
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=hflip_prob),
        T.RandomRotation(degrees=(-rotation_deg, rotation_deg)),
        T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        ),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=affine_scale
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=blur_prob),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    return T.Compose(transforms_list)
