from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
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
        stages = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        self.feature_names = ["layer1", "layer2", "layer3", "layer4"]
        self.feature_dims = [self._stage_output_dim(stage) for stage in stages]

    @staticmethod
    def _stage_output_dim(stage: nn.Sequential) -> int:
        block = stage[-1]
        convolution = block.conv3 if hasattr(block, "conv3") else block.conv2
        return int(convolution.out_channels)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        features = []
        for stage in (
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ):
            x = stage(x)
            features.append(x)
        pooled = self.backbone.avgpool(features[-1])
        return features, torch.flatten(pooled, 1)


def _sincos_position(
    height: int,
    width: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("ViT embedding dimension must be divisible by four")
    quarter = dimension // 4
    frequencies = 1.0 / (
        10_000
        ** (torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter, 1))
    )
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_phase = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_phase = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    position = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=1
    )
    return position.unsqueeze(0).to(dtype=dtype)


class ViTEncoder(nn.Module):
    """Plain ViT with dynamic 2D positions for non-square event tensors."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or depth <= 0 or embed_dim % num_heads:
            raise ValueError("Invalid ViT patch size, depth, or attention head count")
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.feature_indices = sorted(
            {max(round(depth * fraction / 4) - 1, 0) for fraction in range(1, 5)}
        )
        self.feature_names = [f"block{index + 1}" for index in self.feature_indices]
        self.feature_dims = [embed_dim] * len(self.feature_indices)
        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        pad_h = (-x.shape[-2]) % self.patch_size
        pad_w = (-x.shape[-1]) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        patches = self.patch_embed(x)
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + _sincos_position(
            height, width, tokens.shape[-1], tokens.device, tokens.dtype
        )
        features = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in self.feature_indices:
                feature = self.norm(tokens).transpose(1, 2).reshape(
                    tokens.shape[0], tokens.shape[2], height, width
                )
                features.append(feature)
        tokens = self.norm(tokens)
        return features, tokens.mean(dim=1)


def build_frame_encoder(
    name: str,
    input_channels: int,
    small_stem: bool,
    model_config: Any | None = None,
) -> ResNetEncoder | ViTEncoder:
    if name.startswith("resnet"):
        return ResNetEncoder(name, input_channels, small_stem)
    presets = {
        "vit_tiny": (192, 6, 3),
        "vit_small": (384, 8, 6),
        "vit_base": (768, 12, 12),
    }
    if name not in presets:
        raise ValueError(f"Unsupported backbone: {name}")
    default_dim, default_depth, default_heads = presets[name]
    config = model_config or {}
    return ViTEncoder(
        input_channels=input_channels,
        patch_size=int(config.get("vit_patch_size", 16)),
        embed_dim=int(config.get("vit_embed_dim", default_dim)),
        depth=int(config.get("vit_depth", default_depth)),
        num_heads=int(config.get("vit_num_heads", default_heads)),
        mlp_ratio=float(config.get("vit_mlp_ratio", 4.0)),
        dropout=float(config.get("vit_dropout", 0.0)),
    )


class TemporalEncoder(nn.Module):
    """Frame encoder followed by an optional causal recurrent feature layer."""

    def __init__(
        self,
        backbone: str,
        input_channels: int,
        small_stem: bool,
        model_config: Any | None = None,
    ) -> None:
        super().__init__()
        config = model_config or {}
        self.frame_encoder = build_frame_encoder(
            backbone, input_channels, small_stem, model_config=config
        )
        self.feature_names = self.frame_encoder.feature_names
        self.feature_dims = self.frame_encoder.feature_dims
        recurrent_type = str(config.get("recurrent", "none")).lower()
        self.recurrent_type = recurrent_type
        if recurrent_type == "none":
            self.recurrent = None
            self.output_dim = self.frame_encoder.output_dim
            return
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported recurrent layer: {recurrent_type}")
        hidden_dim = int(config.get("recurrent_hidden_dim", self.frame_encoder.output_dim))
        layers = int(config.get("recurrent_layers", 1))
        dropout = float(config.get("recurrent_dropout", 0.0)) if layers > 1 else 0.0
        recurrent_class = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.recurrent = recurrent_class(
            self.frame_encoder.output_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.output_dim = hidden_dim

    def forward(self, voxel: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if voxel.ndim == 5:
            voxel = voxel.unsqueeze(1)
        if voxel.ndim != 6:
            raise ValueError(
                f"Expected [N,C,B,H,W] or [N,T,C,B,H,W], received {tuple(voxel.shape)}"
            )
        batch, timesteps = voxel.shape[:2]
        frames = voxel.reshape(batch * timesteps, *voxel.shape[2:])
        maps, pooled = self.frame_encoder(flatten_voxel(frames))
        sequence_maps = [
            feature.reshape(batch, timesteps, *feature.shape[1:]) for feature in maps
        ]
        sequence_features = pooled.reshape(batch, timesteps, -1)
        if self.recurrent is not None:
            sequence_features, _ = self.recurrent(sequence_features)
        return sequence_maps, sequence_features


def flatten_voxel(voxel: torch.Tensor) -> torch.Tensor:
    if voxel.ndim != 5:
        raise ValueError(f"Expected [N,C,B,H,W], received {tuple(voxel.shape)}")
    return voxel.flatten(1, 2)
