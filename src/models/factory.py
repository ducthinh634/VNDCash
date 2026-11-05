from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torchvision import models

from .baseline_cnn import BaselineCNN


def build_model(
    model_name: str = "baseline_cnn",
    num_classes: int = 9,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float = 0.3,
    **kwargs: Any
) -> nn.Module:
    """Factory function to instantiate models for Vietnamese Banknote Recognition.

    Supported models:
        - 'baseline_cnn' / 'custom_cnn': Lightweight custom CNN built for VNDCash.
        - 'mobilenet_v2': MobileNetV2 with modified 9-class classification head.
        - 'resnet18': ResNet-18 with modified 9-class classification head.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of banknote classes (default 9).
        in_channels: Number of input channels (default 3 for RGB).
        pretrained: Whether to load ImageNet pretrained weights (for torchvision models).
        dropout: Dropout rate in classifier head.
        **kwargs: Additional model-specific hyperparameters.

    Returns:
        Instantiated PyTorch nn.Module.
    """
    normalized_name = model_name.strip().lower()

    if normalized_name in ["baseline_cnn", "custom_cnn", "baseline"]:
        model = BaselineCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_rate=dropout,
            init_weights=kwargs.get("init_weights", True)
        )
        return model

    elif normalized_name in ["mobilenet_v2", "mobilenetv2"]:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # Replace first conv if in_channels != 3
        if in_channels != 3:
            orig_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels, orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )

        # Replace classification head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif normalized_name in ["resnet18", "resnet_18"]:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Replace first conv if in_channels != 3
        if in_channels != 3:
            orig_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels, orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )

        # Replace classification head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes)
        )
        return model

    else:
        raise ValueError(
            f"Unsupported model name: '{model_name}'. "
            f"Supported models are: ['baseline_cnn', 'mobilenet_v2', 'resnet18']"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculates total and trainable parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    """Inspects model output shape and parameter count with a dummy forward pass."""
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    dummy_input = torch.zeros(input_size, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    if was_training:
        model.train()

    param_info = count_parameters(model)
    return {
        "input_shape": list(dummy_input.shape),
        "output_shape": list(output.shape),
        "num_classes": output.shape[-1],
        "total_params": param_info["total"],
        "trainable_params": param_info["trainable"]
    }
