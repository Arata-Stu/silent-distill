from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slassl.losses import (
    SilenceAwareLoss,
    event_rate_loss,
    normalized_distillation_loss,
    polarity_loss,
)
from slassl.models.encoder import ResNetEncoder, flatten_voxel


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OccupancyHead(nn.Module):
    def __init__(self, feature_channels: int, output_channels: int) -> None:
        super().__init__()
        hidden = max(feature_channels // 4, 128)
        self.layers = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, output_channels, 1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class SLASSLModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        input_channels: int,
        polarity_channels: int,
        temporal_bins: int,
        projection_hidden_dim: int,
        projection_dim: int,
        small_stem: bool,
        loss_config: Any,
    ) -> None:
        super().__init__()
        self.polarity_channels = polarity_channels
        self.temporal_bins = temporal_bins
        self.student_encoder = ResNetEncoder(backbone, input_channels, small_stem)
        self.student_projector = ProjectionHead(
            self.student_encoder.output_dim, projection_hidden_dim, projection_dim
        )
        self.teacher_encoder = deepcopy(self.student_encoder)
        self.teacher_projector = deepcopy(self.student_projector)
        for parameter in list(self.teacher_encoder.parameters()) + list(
            self.teacher_projector.parameters()
        ):
            parameter.requires_grad_(False)
        self.occupancy_head = OccupancyHead(
            self.student_encoder.output_dim, polarity_channels * temporal_bins
        )
        self.silence_loss = SilenceAwareLoss(
            modes=loss_config.negative_modes,
            negative_ratio=float(loss_config.negative_ratio),
            negative_weight=float(loss_config.negative_weight),
            near_radius=int(loss_config.near_radius),
            min_negatives=int(loss_config.min_negatives),
        )
        self.lambda_s2l = float(loss_config.get("lambda_s2l", 1.0))
        self.lambda_silence = float(loss_config.lambda_silence)
        self.lambda_polarity = float(loss_config.lambda_polarity)
        self.lambda_rate = float(loss_config.lambda_rate)

    def train(self, mode: bool = True) -> "SLASSLModel":
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        return self

    def forward(
        self, short: torch.Tensor, long: torch.Tensor, occupancy: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        student_map, student_feature = self.student_encoder(flatten_voxel(short))
        student_projection = self.student_projector(student_feature)
        with torch.no_grad():
            _, teacher_feature = self.teacher_encoder(flatten_voxel(long))
            teacher_projection = self.teacher_projector(teacher_feature)
        raw_logits = self.occupancy_head(student_map, occupancy.shape[-2:])
        logits = raw_logits.view(
            short.shape[0], self.polarity_channels, self.temporal_bins, *occupancy.shape[-2:]
        )

        s2l = normalized_distillation_loss(student_projection, teacher_projection)
        silence, sampling_stats = self.silence_loss(logits, occupancy)
        polarity = polarity_loss(logits, occupancy)
        rate = event_rate_loss(logits, occupancy)
        total = (
            self.lambda_s2l * s2l
            + self.lambda_silence * silence
            + self.lambda_polarity * polarity
            + self.lambda_rate * rate
        )
        return {
            "loss": total,
            "loss_s2l": s2l.detach(),
            "loss_silence": silence.detach(),
            "loss_polarity": polarity.detach(),
            "loss_rate": rate.detach(),
            **sampling_stats,
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        pairs = (
            (self.student_encoder, self.teacher_encoder),
            (self.student_projector, self.teacher_projector),
        )
        for student, teacher in pairs:
            for student_parameter, teacher_parameter in zip(
                student.parameters(), teacher.parameters(), strict=True
            ):
                teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)
            for student_buffer, teacher_buffer in zip(
                student.buffers(), teacher.buffers(), strict=True
            ):
                teacher_buffer.copy_(student_buffer)
