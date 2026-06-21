from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from slassl.models.encoder import TemporalEncoder


class ClassificationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_channels: int,
        num_classes: int,
        small_stem: bool,
        dropout: float = 0.0,
        model_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(
            backbone, input_channels, small_stem, model_config=model_config
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.output_dim, num_classes)
        )

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        _, feature_sequence = self.encoder(voxel)
        return self.head(feature_sequence[:, -1])


def build_detector(
    backbone_name: str,
    input_channels: int,
    num_classes: int,
    min_size: int,
    max_size: int,
) -> FCOS:
    if not backbone_name.startswith("resnet"):
        raise ValueError("FCOS fine-tuning currently requires a ResNet backbone")
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
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    encoder_state: dict[str, torch.Tensor] = {}
    if destination_prefix == "encoder.":
        prefixes = ("module.student_encoder.", "student_encoder.")
        legacy_prefixes = (
            "module.student_encoder.backbone.",
            "student_encoder.backbone.",
        )
        for key, value in state.items():
            legacy_match = False
            for prefix in legacy_prefixes:
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    encoder_state[destination_prefix + "frame_encoder.backbone." + suffix] = value
                    legacy_match = True
                    break
            if legacy_match:
                continue
            for prefix in prefixes:
                if key.startswith(prefix):
                    encoder_state[destination_prefix + key[len(prefix) :]] = value
                    break
    else:
        prefixes = (
            "module.student_encoder.frame_encoder.backbone.",
            "student_encoder.frame_encoder.backbone.",
            "module.student_encoder.backbone.",
            "student_encoder.backbone.",
        )
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    encoder_state[destination_prefix + key[len(prefix) :]] = value
                    break
    if not encoder_state:
        raise ValueError(f"No student encoder weights found in {checkpoint_path}")
    incompatible = module.load_state_dict(encoder_state, strict=False)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    return (
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
        checkpoint_config if isinstance(checkpoint_config, dict) else None,
    )
