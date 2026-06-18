from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from slassl.data.h5_events import H5EventReader
from slassl.data.indexing import load_timestamp_index
from slassl.data.voxel import EventVoxelizer


def _structured_field(array: np.ndarray, *names: str) -> np.ndarray:
    fields = array.dtype.names or ()
    for name in names:
        if name in fields:
            return array[name]
    raise ValueError(f"None of {names} found in annotation fields {fields}")


class EventWindowDataset(Dataset):
    """Causal event windows backed by sequence HDF5 files and a JSONL manifest."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        height: int,
        width: int,
        bins: int,
        short_window_us: int,
        long_window_us: int,
        task: str = "ssl",
        use_polarity: bool = True,
        representation: str = "count",
        occupancy_stride: int = 8,
        temporal_shuffle: bool = False,
        horizontal_flip_probability: float = 0.0,
        event_camera: str = "left",
        event_group: str | None = None,
        timestamp_scale_to_us: float | None = None,
        timestamp_offset_us: float | None = None,
        detection_class_ids: list[int] | None = None,
        detection_min_box_diagonal: float = 0.0,
    ) -> None:
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.short_window_us = int(short_window_us)
        self.long_window_us = int(long_window_us)
        self.task = task
        self.occupancy_stride = int(occupancy_stride)
        self.temporal_shuffle = temporal_shuffle
        self.horizontal_flip_probability = horizontal_flip_probability
        self.event_camera = event_camera
        self.event_group = event_group
        self.timestamp_scale_to_us = timestamp_scale_to_us
        self.timestamp_offset_us = timestamp_offset_us
        self.detection_class_ids = detection_class_ids or [0, 1, 2]
        self.detection_min_box_diagonal = float(detection_min_box_diagonal)
        if task not in {"ssl", "classification", "detection"}:
            raise ValueError(f"Unknown task: {task}")
        if self.short_window_us <= 0 or self.long_window_us <= 0:
            raise ValueError("Event windows must be positive")
        if task == "ssl" and self.long_window_us < self.short_window_us:
            raise ValueError("SSL long_window_us must be greater than or equal to short_window_us")
        if self.occupancy_stride <= 0:
            raise ValueError("occupancy_stride must be positive")
        self.voxelizer = EventVoxelizer(
            height=height,
            width=width,
            bins=bins,
            use_polarity=use_polarity,
            representation=representation,
        )
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        self._readers: dict[str, H5EventReader] = {}
        self._annotations: dict[str, np.ndarray] = {}
        self._timestamp_indices: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _reader(self, relative_path: str) -> H5EventReader:
        if relative_path not in self._readers:
            self._readers[relative_path] = H5EventReader(
                self.data_root / relative_path,
                camera=self.event_camera,
                event_group=self.event_group,
                timestamp_scale_to_us=self.timestamp_scale_to_us,
                timestamp_offset_us=self.timestamp_offset_us,
            )
        return self._readers[relative_path]

    def _events(self, record: dict[str, Any]) -> tuple[np.ndarray, ...]:
        reader = self._reader(record["sequence"])
        end_us = int(record["timestamp_us"])
        window_us = self.long_window_us if self.task == "ssl" else self.short_window_us
        start_us = end_us - window_us
        index_map = None
        if record.get("timestamp_index"):
            index_path = record["timestamp_index"]
            if index_path not in self._timestamp_indices:
                self._timestamp_indices[index_path] = load_timestamp_index(
                    self.data_root / index_path, self.event_camera
                )
            index_map = self._timestamp_indices[index_path]
        return reader.read_us(
            start_us,
            end_us,
            index_map=index_map,
            index_period_us=int(record.get("timestamp_index_period_us", 20)),
        )

    def _detection_target(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "annotations" not in record:
            boxes = torch.tensor(record.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor(record.get("labels", []), dtype=torch.int64)
            return boxes, labels

        relative_path = record["annotations"]
        if relative_path not in self._annotations:
            self._annotations[relative_path] = np.load(
                self.data_root / relative_path, mmap_mode="r", allow_pickle=False
            )
        annotations = self._annotations[relative_path][
            int(record["annotation_start"]) : int(record["annotation_end"])
        ]
        raw_classes = _structured_field(annotations, "class_id", "class", "label").astype(int)
        x = _structured_field(annotations, "x").astype(float)
        y = _structured_field(annotations, "y").astype(float)
        w = _structured_field(annotations, "w", "width").astype(float)
        h = _structured_field(annotations, "h", "height").astype(float)
        class_map = {raw: index + 1 for index, raw in enumerate(self.detection_class_ids)}
        keep = np.array([raw_class in class_map for raw_class in raw_classes])
        if self.detection_min_box_diagonal > 0:
            keep &= np.hypot(w, h) >= self.detection_min_box_diagonal
        x1 = np.clip(x[keep], 0, self.voxelizer.width)
        y1 = np.clip(y[keep], 0, self.voxelizer.height)
        x2 = np.clip(x[keep] + w[keep], 0, self.voxelizer.width)
        y2 = np.clip(y[keep] + h[keep], 0, self.voxelizer.height)
        valid = (x2 > x1) & (y2 > y1)
        boxes_array = np.stack((x1[valid], y1[valid], x2[valid], y2[valid]), axis=1)
        labels_array = np.array(
            [class_map[raw] for raw in raw_classes[keep]], dtype=np.int64
        )[valid]
        return torch.from_numpy(boxes_array.astype(np.float32)), torch.from_numpy(labels_array)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        end_us = int(record["timestamp_us"])
        x, y, t, p = self._events(record)
        short_start = end_us - self.short_window_us
        short_mask = t >= short_start
        short_args = (x[short_mask], y[short_mask], t[short_mask], p[short_mask])
        short = self.voxelizer(*short_args, short_start, end_us)

        if self.temporal_shuffle:
            permutation = torch.randperm(short.shape[1])
            short = short[:, permutation]
        flipped = self.horizontal_flip_probability > 0 and (
            torch.rand(()) < self.horizontal_flip_probability
        )
        if flipped:
            short = short.flip(-1)

        duration_ms = self.short_window_us / 1000.0
        density = float(short_mask.sum()) / (
            self.voxelizer.height * self.voxelizer.width * max(duration_ms, 1e-6)
        )
        sample: dict[str, Any] = {
            "short": short,
            "density": torch.tensor(density, dtype=torch.float32),
            "sample_id": record.get("sample_id", str(index)),
            "timestamp_us": end_us,
        }
        if self.task == "ssl":
            long = self.voxelizer(x, y, t, p, end_us - self.long_window_us, end_us)
            occupancy = self.voxelizer.occupancy(
                *short_args, short_start, end_us, self.occupancy_stride
            )
            if self.temporal_shuffle:
                long = long[:, permutation]
                occupancy = occupancy[:, permutation]
            if flipped:
                long = long.flip(-1)
                occupancy = occupancy.flip(-1)
            sample["long"] = long
            sample["occupancy"] = occupancy
        elif self.task == "classification":
            sample["label"] = torch.tensor(int(record["label"]), dtype=torch.long)
        elif self.task == "detection":
            boxes, labels = self._detection_target(record)
            if len(boxes) != len(labels):
                raise ValueError(f"Mismatched boxes and labels for sample {sample['sample_id']}")
            if labels.numel() and int(labels.min()) < 1:
                raise ValueError("Detection labels must start at 1; label 0 is reserved")
            if flipped and boxes.numel():
                left = self.voxelizer.width - boxes[:, 2]
                right = self.voxelizer.width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = left, right
            sample["target"] = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([index], dtype=torch.int64),
            }
        return sample

    def __del__(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()


class EventSequenceDataset(Dataset):
    """Group causal windows from one recording into chronological sequences."""

    def __init__(
        self,
        dataset: EventWindowDataset,
        length: int,
        step: int = 1,
        stride: int = 1,
        max_gap_us: int = 0,
    ) -> None:
        if length <= 1:
            raise ValueError("EventSequenceDataset length must be greater than one")
        if step <= 0 or stride <= 0 or max_gap_us < 0:
            raise ValueError("Sequence step/stride must be positive and max_gap_us non-negative")
        self.dataset = dataset
        self.length = length
        self.step = step
        self.stride = stride
        self.max_gap_us = max_gap_us
        self.temporal_shuffle = dataset.temporal_shuffle
        self.horizontal_flip_probability = dataset.horizontal_flip_probability
        dataset.temporal_shuffle = False
        dataset.horizontal_flip_probability = 0.0

        by_sequence: dict[str, list[int]] = {}
        for index, record in enumerate(dataset.records):
            by_sequence.setdefault(str(record["sequence"]), []).append(index)
        self.sequences: list[tuple[int, ...]] = []
        span = (length - 1) * step
        for indices in by_sequence.values():
            indices.sort(key=lambda item: int(dataset.records[item]["timestamp_us"]))
            for end in range(span, len(indices), stride):
                selected = tuple(indices[end - offset * step] for offset in reversed(range(length)))
                timestamps = [int(dataset.records[item]["timestamp_us"]) for item in selected]
                if max_gap_us and any(
                    right - left > max_gap_us
                    for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)
                ):
                    continue
                self.sequences.append(selected)
        if not self.sequences:
            raise ValueError("No valid event sequences could be built from the manifest")

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.dataset.records

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        samples = [self.dataset[item] for item in self.sequences[index]]
        output: dict[str, Any] = {
            "short": torch.stack([sample["short"] for sample in samples]),
            "density": samples[-1]["density"],
            "density_sequence": torch.stack([sample["density"] for sample in samples]),
            "sample_id": [sample["sample_id"] for sample in samples],
            "timestamp_us": torch.tensor(
                [sample["timestamp_us"] for sample in samples], dtype=torch.int64
            ),
        }
        for key in ("long", "occupancy"):
            if key in samples[0]:
                output[key] = torch.stack([sample[key] for sample in samples])
        for key in ("label", "target"):
            if key in samples[-1]:
                output[key] = samples[-1][key]

        if self.temporal_shuffle:
            permutation = torch.randperm(output["short"].shape[2])
            for key in ("short", "long", "occupancy"):
                if key in output:
                    output[key] = output[key][:, :, permutation]
        if self.horizontal_flip_probability > 0 and (
            torch.rand(()) < self.horizontal_flip_probability
        ):
            for key in ("short", "long", "occupancy"):
                if key in output:
                    output[key] = output[key].flip(-1)
            target = output.get("target")
            if target is not None and target["boxes"].numel():
                boxes = target["boxes"]
                left = self.dataset.voxelizer.width - boxes[:, 2]
                right = self.dataset.voxelizer.width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = left, right
        return output


def event_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output
