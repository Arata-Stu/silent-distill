from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slassl.models.encoder import TemporalEncoder


class MultiScaleDenseDecoder(nn.Module):
    def __init__(self, feature_dims: list[int], output_channels: int, channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(feature_dim, channels, kernel_size=1) for feature_dim in feature_dims]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * len(feature_dims), channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, output_channels, kernel_size=1),
        )

    def forward(
        self, features: list[torch.Tensor], output_size: tuple[int, int]
    ) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        lateral = []
        for projection, feature in zip(self.lateral, features, strict=True):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            lateral.append(value)
        output = self.refine(torch.cat(lateral, dim=1))
        return F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)


class DensePredictionModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_channels: int,
        output_channels: int,
        small_stem: bool,
        model_config: Any | None = None,
    ) -> None:
        super().__init__()
        config = model_config or {}
        self.encoder = TemporalEncoder(
            backbone, input_channels, small_stem, model_config=config
        )
        self.decoder = MultiScaleDenseDecoder(
            self.encoder.feature_dims,
            output_channels,
            channels=int(config.get("decoder_channels", 128)),
        )

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(voxel)
        features = [feature[:, -1] for feature in features]
        return self.decoder(features, tuple(voxel.shape[-2:]))
