from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from slassl.data.dense import read_dense_target, resize_flow, resize_segmentation
from slassl.data.dsec import decode_dsec_flow, read_dsec_flow_png
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
        segmentation_num_classes: int = 11,
        segmentation_ignore_index: int = 255,
        flow_target_duration_us: int | None = None,
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
        self.segmentation_num_classes = int(segmentation_num_classes)
        self.segmentation_ignore_index = int(segmentation_ignore_index)
        self.flow_target_duration_us = (
            int(flow_target_duration_us) if flow_target_duration_us is not None else None
        )
        if task not in {
            "ssl",
            "classification",
            "detection",
            "flow",
            "flow_inference",
            "segmentation",
        }:
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
        self._dense_targets: dict[str, h5py.File] = {}
        self._rectify_maps: dict[str, np.ndarray] = {}

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
        start_us = int(record.get("window_start_us", end_us - window_us))
        index_map = None
        if record.get("timestamp_index"):
            index_path = record["timestamp_index"]
            if index_path not in self._timestamp_indices:
                self._timestamp_indices[index_path] = load_timestamp_index(
                    self.data_root / index_path, self.event_camera
                )
            index_map = self._timestamp_indices[index_path]
        x, y, t, p = reader.read_us(
            start_us,
            end_us,
            index_map=index_map,
            index_period_us=int(record.get("timestamp_index_period_us", 20)),
            end_inclusive=bool(record.get("end_inclusive", True)),
        )
        if rectify_path := record.get("rectify_map"):
            if rectify_path not in self._rectify_maps:
                with h5py.File(self.data_root / rectify_path, "r") as handle:
                    if "rectify_map" not in handle:
                        raise ValueError(f"Missing rectify_map dataset in {rectify_path}")
                    self._rectify_maps[rectify_path] = np.asarray(handle["rectify_map"])
            rectify_map = self._rectify_maps[rectify_path]
            source_x = np.asarray(x, dtype=np.int64)
            source_y = np.asarray(y, dtype=np.int64)
            valid = (
                (source_x >= 0)
                & (source_x < rectify_map.shape[1])
                & (source_y >= 0)
                & (source_y < rectify_map.shape[0])
            )
            rectified = rectify_map[source_y[valid], source_x[valid]]
            x = rectified[:, 0]
            y = rectified[:, 1]
            t = t[valid]
            p = p[valid]
        return x, y, t, p

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

    def _dense_target(self, record: dict[str, Any]) -> torch.Tensor:
        relative_path = str(record["target"])
        if relative_path not in self._dense_targets:
            self._dense_targets[relative_path] = h5py.File(self.data_root / relative_path, "r")
        return read_dense_target(
            self._dense_targets[relative_path],
            str(record["target_format"]),
            int(record["target_index"]),
            camera=str(record.get("target_camera", self.event_camera)),
            segmentation_num_classes=self.segmentation_num_classes,
            segmentation_ignore_index=self.segmentation_ignore_index,
        )

    def _flow_target(
        self, record: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if str(record["target_format"]) == "dsec_flow":
            encoded = read_dsec_flow_png(self.data_root / str(record["target"]))
            flow, valid = decode_dsec_flow(encoded)
            return torch.from_numpy(flow), torch.from_numpy(valid)
        return self._dense_target(record), None

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        end_us = int(record["timestamp_us"])
        x, y, t, p = self._events(record)
        short_start = int(record.get("window_start_us", end_us - self.short_window_us))
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

        duration_ms = (end_us - short_start) / 1000.0
        density = float(short_mask.sum()) / (
            self.voxelizer.height * self.voxelizer.width * max(duration_ms, 1e-6)
        )
        sample: dict[str, Any] = {
            "short": short,
            "density": torch.tensor(density, dtype=torch.float32),
            "sample_id": record.get("sample_id", str(index)),
            "timestamp_us": end_us,
        }
        for key in ("output_sequence", "output_index", "target_dt_us", "metric_sequence"):
            if key in record:
                sample[key] = record[key]
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
        elif self.task == "flow":
            raw_target, target_valid = self._flow_target(record)
            target = resize_flow(
                raw_target, (self.voxelizer.height, self.voxelizer.width)
            )
            if target_valid is not None and tuple(target_valid.shape) != (
                self.voxelizer.height,
                self.voxelizer.width,
            ):
                target_valid = resize_segmentation(
                    target_valid.long(), (self.voxelizer.height, self.voxelizer.width)
                ).bool()
            target_dt_us = int(record.get("target_dt_us", 0))
            if self.flow_target_duration_us is not None and target_dt_us > 0:
                target *= self.flow_target_duration_us / target_dt_us
            finite = torch.isfinite(target).all(dim=0)
            valid = target_valid & finite if target_valid is not None else (
                finite & (target.norm(dim=0) > 0)
            )
            if valid_height := record.get("valid_height"):
                valid[int(valid_height) :] = False
            target = torch.nan_to_num(target)
            if flipped:
                target = target.flip(-1)
                target[0] = -target[0]
                valid = valid.flip(-1)
            valid_mask_mode = str(record.get("valid_mask_mode", "event_support"))
            if valid_mask_mode == "event_support":
                valid &= short.abs().sum(dim=(0, 1)) > 0
            elif valid_mask_mode != "target":
                raise ValueError(f"Unknown flow valid_mask_mode: {valid_mask_mode}")
            sample["target"] = target
            sample["valid_mask"] = valid
        elif self.task == "segmentation":
            target = resize_segmentation(
                self._dense_target(record), (self.voxelizer.height, self.voxelizer.width)
            )
            if flipped:
                target = target.flip(-1)
            sample["target"] = target
        return sample

    def __del__(self) -> None:
        for reader in getattr(self, "_readers", {}).values():
            reader.close()
        for handle in getattr(self, "_dense_targets", {}).values():
            handle.close()


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
        for key in ("label", "target", "valid_mask"):
            if key in samples[-1]:
                output[key] = samples[-1][key]
        if "metric_sequence" in samples[-1]:
            output["metric_sequence"] = samples[-1]["metric_sequence"]

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
