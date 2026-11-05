import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class BaselineCNN(nn.Module):
    """Custom Convolutional Neural Network (Baseline) for Vietnamese Banknote Recognition.

    Input:  Tensor of shape [Batch, 3, 224, 224]
    Output: Logits of shape [Batch, 9] (corresponding to 9 banknote denominations)

    Architecture:
        - 4 Convolutional Blocks:
            Block 1: Conv(3->32, k=3, p=1) -> BN -> ReLU -> MaxPool(2)  [112x112]
            Block 2: Conv(32->64, k=3, p=1) -> BN -> ReLU -> MaxPool(2)  [56x56]
            Block 3: Conv(64->128, k=3, p=1) -> BN -> ReLU -> MaxPool(2) [28x28]
            Block 4: Conv(128->256, k=3, p=1) -> BN -> ReLU -> MaxPool(2)[14x14]
        - Adaptive Average Pooling: [256, 4, 4]
        - Fully Connected Classifier Head with Dropout regularization:
            Linear(4096 -> 256) -> BN -> ReLU -> Dropout(p) -> Linear(256 -> num_classes)
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
        init_weights: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        # Feature Extractor (Backbone)
        self.features = nn.Sequential(
            # Block 1: [B, in_channels, 224, 224] -> [B, 32, 112, 112]
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: [B, 32, 112, 112] -> [B, 64, 56, 56]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: [B, 64, 56, 56] -> [B, 128, 28, 28]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: [B, 128, 28, 28] -> [B, 256, 14, 14]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to handle potential minor resolution differences
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, C, H, W]

        Returns:
            Class logits of shape [B, num_classes]
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits

    def _initialize_weights(self) -> None:
        """Kaiming (He) normal initialization for conv layers, xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_num_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
