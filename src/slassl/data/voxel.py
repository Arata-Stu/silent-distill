from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EventVoxelizer:
    height: int
    width: int
    bins: int = 5
    use_polarity: bool = True
    representation: str = "count"
    clip_value: float | None = 10.0

    @property
    def channels(self) -> int:
        return 2 if self.use_polarity else 1

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.bins <= 0:
            raise ValueError("height, width, and bins must be positive")

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        start_us: int,
        end_us: int,
    ) -> torch.Tensor:
        shape = (self.channels, self.bins, self.height, self.width)
        if x.size == 0:
            return torch.zeros(shape, dtype=torch.float32)

        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
            & (t >= start_us)
            & (t <= end_us)
        )
        x = x[valid].astype(np.int64, copy=False)
        y = y[valid].astype(np.int64, copy=False)
        t = t[valid]
        p = p[valid]
        if x.size == 0:
            return torch.zeros(shape, dtype=torch.float32)

        duration = max(end_us - start_us, 1)
        temporal = np.floor((t - start_us) * self.bins / duration).astype(np.int64)
        temporal = np.clip(temporal, 0, self.bins - 1)
        polarity = (p > 0).astype(np.int64) if self.use_polarity else np.zeros_like(x)
        flat = (((polarity * self.bins + temporal) * self.height + y) * self.width + x)
        counts = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)

        if self.representation == "binary":
            counts = counts > 0
        elif self.representation == "log_count":
            counts = np.log1p(counts)
        elif self.representation != "count":
            raise ValueError(f"Unknown representation: {self.representation}")
        voxel = torch.from_numpy(counts.astype(np.float32, copy=False))
        if self.clip_value is not None:
            voxel.clamp_(max=self.clip_value)
        return voxel

    def occupancy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        start_us: int,
        end_us: int,
        spatial_stride: int,
    ) -> torch.Tensor:
        if spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        out_h = (self.height + spatial_stride - 1) // spatial_stride
        out_w = (self.width + spatial_stride - 1) // spatial_stride
        shape = (self.channels, self.bins, out_h, out_w)
        if x.size == 0:
            return torch.zeros(shape, dtype=torch.float32)
        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
            & (t >= start_us)
            & (t <= end_us)
        )
        x = x[valid].astype(np.int64, copy=False) // spatial_stride
        y = y[valid].astype(np.int64, copy=False) // spatial_stride
        t = t[valid]
        p = p[valid]
        if x.size == 0:
            return torch.zeros(shape, dtype=torch.float32)
        duration = max(end_us - start_us, 1)
        temporal = np.floor((t - start_us) * self.bins / duration).astype(np.int64)
        temporal = np.clip(temporal, 0, self.bins - 1)
        polarity = (p > 0).astype(np.int64) if self.use_polarity else np.zeros_like(x)
        flat = (((polarity * self.bins + temporal) * out_h + y) * out_w + x)
        occupancy = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape) > 0
        return torch.from_numpy(occupancy.astype(np.float32, copy=False))
