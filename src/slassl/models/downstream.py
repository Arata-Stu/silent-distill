from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from slassl.models.encoder import ResNetEncoder, flatten_voxel


class ClassificationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_channels: int,
        num_classes: int,
        small_stem: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(backbone, input_channels, small_stem)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.output_dim, num_classes))

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        _, feature = self.encoder(flatten_voxel(voxel))
        return self.head(feature)


def build_detector(
    backbone_name: str,
    input_channels: int,
    num_classes: int,
    min_size: int,
    max_size: int,
) -> FCOS:
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name, weights=None, trainable_layers=5
    )
    old_conv = backbone.body.conv1
    backbone.body.conv1 = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    return FCOS(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        image_mean=[0.0] * input_channels,
        image_std=[1.0] * input_channels,
    )


def load_pretrained_encoder(
    module: nn.Module, checkpoint_path: str, destination_prefix: str
) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefixes = ("module.student_encoder.backbone.", "student_encoder.backbone.")
    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                encoder_state[destination_prefix + key[len(prefix) :]] = value
                break
    if not encoder_state:
        raise ValueError(f"No student encoder weights found in {checkpoint_path}")
    incompatible = module.load_state_dict(encoder_state, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
