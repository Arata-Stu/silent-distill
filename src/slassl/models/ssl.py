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
from slassl.models.encoder import TemporalEncoder


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


@torch.no_grad()
def _paired_representation_stats(
    student: torch.Tensor, teacher: torch.Tensor, name: str
) -> dict[str, torch.Tensor]:
    student = student.detach().float().reshape(-1, student.shape[-1])
    teacher = teacher.detach().float().reshape(-1, teacher.shape[-1])
    normalized_student = F.normalize(student, dim=-1)
    normalized_teacher = F.normalize(teacher, dim=-1)
    stats = {
        f"diagnostics/student_std_{name}": student.std(dim=0, unbiased=False).mean(),
        f"diagnostics/teacher_std_{name}": teacher.std(dim=0, unbiased=False).mean(),
        f"diagnostics/student_normalized_std_{name}": normalized_student.std(
            dim=0, unbiased=False
        ).mean(),
        f"diagnostics/teacher_normalized_std_{name}": normalized_teacher.std(
            dim=0, unbiased=False
        ).mean(),
        f"diagnostics/student_norm_{name}": student.norm(dim=-1).mean(),
        f"diagnostics/teacher_norm_{name}": teacher.norm(dim=-1).mean(),
        f"diagnostics/cosine_{name}": F.cosine_similarity(student, teacher, dim=-1).mean(),
    }
    if len(student) > 1:
        student_similarity = normalized_student @ normalized_student.T
        teacher_similarity = normalized_teacher @ normalized_teacher.T
        pairs = len(student) * (len(student) - 1)
        stats[f"diagnostics/student_pairwise_cosine_{name}"] = (
            student_similarity.sum() - student_similarity.diagonal().sum()
        ) / pairs
        stats[f"diagnostics/teacher_pairwise_cosine_{name}"] = (
            teacher_similarity.sum() - teacher_similarity.diagonal().sum()
        ) / pairs
    return stats


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
        model_config: Any | None = None,
    ) -> None:
        super().__init__()
        model_config = model_config or {}
        self.polarity_channels = polarity_channels
        self.temporal_bins = temporal_bins
        self.student_encoder = TemporalEncoder(
            backbone, input_channels, small_stem, model_config=model_config
        )
        self.student_projector = ProjectionHead(
            self.student_encoder.output_dim, projection_hidden_dim, projection_dim
        )
        self.teacher_encoder = deepcopy(self.student_encoder)
        self.teacher_projector = deepcopy(self.student_projector)
        self.scale_indices: dict[str, int] = {}
        if bool(model_config.get("multi_scale_distillation", True)):
            requested = model_config.get("distillation_scales", "all")
            selected = (
                self.student_encoder.feature_names
                if requested == "all"
                else [str(name) for name in requested]
            )
            available = dict(
                zip(
                    self.student_encoder.feature_names,
                    range(len(self.student_encoder.feature_names)),
                    strict=True,
                )
            )
            unknown = set(selected) - set(available)
            if unknown:
                raise ValueError(
                    f"Unknown distillation scales {sorted(unknown)}; "
                    f"available scales are {list(available)}"
                )
            self.scale_indices = {name: available[name] for name in selected}
        self.student_scale_projectors = nn.ModuleDict(
            {
                name: ProjectionHead(
                    self.student_encoder.feature_dims[index],
                    projection_hidden_dim,
                    projection_dim,
                )
                for name, index in self.scale_indices.items()
            }
        )
        self.teacher_scale_projectors = deepcopy(self.student_scale_projectors)
        for parameter in list(self.teacher_encoder.parameters()) + list(
            self.teacher_projector.parameters()
        ) + list(self.teacher_scale_projectors.parameters()):
            parameter.requires_grad_(False)
        self.occupancy_head = OccupancyHead(
            self.student_encoder.feature_dims[-1], polarity_channels * temporal_bins
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
        self.teacher_scale_projectors.eval()
        return self

    def forward(
        self, short: torch.Tensor, long: torch.Tensor, occupancy: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        student_maps, student_feature = self.student_encoder(short)
        student_projection = self.student_projector(student_feature)
        with torch.no_grad():
            teacher_maps, teacher_feature = self.teacher_encoder(long)
            teacher_projection = self.teacher_projector(teacher_feature)

        diagnostics = _paired_representation_stats(
            student_feature, teacher_feature, "feature_global"
        )
        diagnostics.update(
            _paired_representation_stats(
                student_projection, teacher_projection, "projection_global"
            )
        )
        distillation_losses = {
            "global": normalized_distillation_loss(student_projection, teacher_projection)
        }
        for name, index in self.scale_indices.items():
            student_scale = student_maps[index].mean(dim=(-2, -1))
            teacher_scale = teacher_maps[index].mean(dim=(-2, -1))
            student_scale = self.student_scale_projectors[name](student_scale)
            with torch.no_grad():
                teacher_scale = self.teacher_scale_projectors[name](teacher_scale)
            distillation_losses[name] = normalized_distillation_loss(
                student_scale, teacher_scale
            )
            diagnostics.update(
                _paired_representation_stats(
                    student_scale, teacher_scale, f"projection_{name}"
                )
            )
        s2l = torch.stack(list(distillation_losses.values())).mean()

        if occupancy.ndim == 5:
            occupancy = occupancy.unsqueeze(1)
        if occupancy.ndim != 6:
            raise ValueError(f"Expected sequence occupancy target, received {occupancy.shape}")
        batch, timesteps = occupancy.shape[:2]
        deepest = student_maps[-1]
        raw_logits = self.occupancy_head(
            deepest.reshape(batch * timesteps, *deepest.shape[2:]), occupancy.shape[-2:]
        )
        logits = raw_logits.reshape(
            batch,
            timesteps,
            self.polarity_channels,
            self.temporal_bins,
            *occupancy.shape[-2:],
        )
        flat_logits = logits.reshape(
            batch * timesteps,
            self.polarity_channels,
            self.temporal_bins,
            *occupancy.shape[-2:],
        )
        flat_occupancy = occupancy.reshape_as(flat_logits)
        silence, sampling_stats = self.silence_loss(flat_logits, flat_occupancy)
        polarity = polarity_loss(flat_logits, flat_occupancy)
        rate = event_rate_loss(flat_logits, flat_occupancy)
        with torch.no_grad():
            probabilities = flat_logits.detach().float().sigmoid()
            occupancy_stats = {
                "occupancy/predicted_rate": probabilities.mean(),
                "occupancy/target_rate": flat_occupancy.detach().float().mean(),
            }
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
            **{
                f"loss_s2l_{name}": value.detach()
                for name, value in distillation_losses.items()
            },
            **sampling_stats,
            **occupancy_stats,
            **diagnostics,
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        pairs = (
            (self.student_encoder, self.teacher_encoder),
            (self.student_projector, self.teacher_projector),
            (self.student_scale_projectors, self.teacher_scale_projectors),
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
