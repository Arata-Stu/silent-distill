from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, name: str, input_channels: int, small_stem: bool = False) -> None:
        super().__init__()
        if not name.startswith("resnet") or not hasattr(models, name):
            raise ValueError(f"Unsupported ResNet backbone: {name}")
        self.backbone = getattr(models, name)(weights=None)
        old_conv = self.backbone.conv1
        if small_stem:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()
        else:
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        self.output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feature_map = self.backbone.layer4(x)
        pooled = self.backbone.avgpool(feature_map)
        return feature_map, torch.flatten(pooled, 1)


def flatten_voxel(voxel: torch.Tensor) -> torch.Tensor:
    if voxel.ndim != 5:
        raise ValueError(f"Expected [N,C,B,H,W], received {tuple(voxel.shape)}")
    return voxel.flatten(1, 2)

