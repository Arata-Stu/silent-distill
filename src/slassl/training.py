from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from slassl.data import EventWindowDataset, event_collate


def build_dataset(
    config: Any,
    task: str,
    manifest: str | None = None,
    augmentation: bool = True,
) -> EventWindowDataset:
    data = config.data
    return EventWindowDataset(
        manifest=manifest or data.manifest,
        data_root=data.root,
        height=int(data.height),
        width=int(data.width),
        bins=int(data.bins),
        short_window_us=int(data.short_window_us),
        long_window_us=int(data.long_window_us),
        task=task,
        use_polarity=bool(data.use_polarity),
        representation=data.representation,
        occupancy_stride=int(data.occupancy_stride),
        temporal_shuffle=bool(data.get("temporal_shuffle", False)),
        horizontal_flip_probability=(
            float(data.get("horizontal_flip_probability", 0.0)) if augmentation else 0.0
        ),
        event_camera=str(data.get("event_camera", "left")),
        event_group=data.get("event_group"),
        timestamp_scale_to_us=(
            float(data.timestamp_scale_to_us)
            if data.get("timestamp_scale_to_us") is not None
            else None
        ),
        timestamp_offset_us=(
            float(data.timestamp_offset_us)
            if data.get("timestamp_offset_us") is not None
            else None
        ),
        detection_class_ids=[
            int(value) for value in data.get("detection_class_ids", [0, 1, 2])
        ],
        detection_min_box_diagonal=float(data.get("detection_min_box_diagonal", 0.0)),
    )


def build_loader(
    dataset: EventWindowDataset,
    config: Any,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=shuffle
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(config.training.num_workers),
        pin_memory=True,
        persistent_workers=int(config.training.num_workers) > 0,
        drop_last=shuffle,
        collate_fn=event_collate,
    )
    return loader, sampler


def cosine_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(step / (total_steps - 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def cosine_warmup_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def append_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
