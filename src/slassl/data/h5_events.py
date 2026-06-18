from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  # Registers DSEC's Blosc compression filters.
except ImportError:
    hdf5plugin = None


@dataclass(frozen=True)
class EventLayout:
    storage: str
    path: str
    timestamp_scale_to_us: float
    timestamp_offset_us: float


def _scalar(dataset: h5py.Dataset) -> float:
    value = np.asarray(dataset[()]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar HDF5 dataset at {dataset.name}")
    return float(value[0])


def _has_split_events(value: h5py.Group | h5py.Dataset) -> bool:
    return isinstance(value, h5py.Group) and all(key in value for key in ("x", "y", "t", "p"))


def _infer_time_scale(
    timestamp_dataset: h5py.Dataset,
    index_dataset: h5py.Dataset | None,
) -> float:
    if np.issubdtype(timestamp_dataset.dtype, np.floating):
        return 1_000_000.0
    if index_dataset is None or len(index_dataset) < 2 or len(timestamp_dataset) < 2:
        return 1.0
    raw_duration = float(timestamp_dataset[-1]) - float(timestamp_dataset[0])
    if raw_duration <= 0:
        return 1.0
    expected_us = (len(index_dataset) - 1) * 1000.0
    candidates = (0.001, 1.0, 1000.0, 1_000_000.0)
    return min(
        candidates,
        key=lambda candidate: abs(np.log10(raw_duration * candidate / expected_us)),
    )


class H5EventReader:
    """Read native event HDF5 layouts used by 1 Mpx, DSEC, M3ED, and MVSEC."""

    def __init__(
        self,
        path: str | Path,
        camera: str = "left",
        event_group: str | None = None,
        timestamp_scale_to_us: float | None = None,
        timestamp_offset_us: float | None = None,
    ) -> None:
        self.path = Path(path)
        try:
            self.handle = h5py.File(self.path, "r")
        except OSError as exc:
            raise OSError(
                f"Could not open {self.path}. DSEC requires the hdf5plugin package; "
                "Metavision ECF-compressed HDF5 is intentionally unsupported."
            ) from exc
        self.camera = camera
        self.native_index_path: str | None = None
        self.native_index_period_us: int | None = None
        self.layout = self._detect_layout(
            event_group, timestamp_scale_to_us, timestamp_offset_us
        )

    def _detect_layout(
        self,
        event_group: str | None,
        scale_override: float | None,
        offset_override: float | None,
    ) -> EventLayout:
        preferred = []
        if event_group:
            preferred.append(event_group)
        preferred.extend(
            [
                "/events",
                f"/prophesee/{self.camera}",
                f"/davis/{self.camera}/events",
                "/CD/events",
                "/",
            ]
        )
        seen: set[str] = set()
        for candidate in preferred:
            if candidate in seen or (candidate != "/" and candidate not in self.handle):
                continue
            seen.add(candidate)
            value = self.handle[candidate]
            if _has_split_events(value):
                storage = "split"
                timestamp_dataset = value["t"]
                path = value.name
                index_dataset = self._index_dataset(value)
            elif isinstance(value, h5py.Dataset) and value.dtype.names:
                fields = set(value.dtype.names)
                if not {"x", "y", "t", "p"}.issubset(fields):
                    continue
                storage = "compound"
                timestamp_dataset = value
                path = value.name
                index_dataset = None
            elif isinstance(value, h5py.Dataset) and value.ndim == 2 and value.shape[1] >= 4:
                storage = "matrix"
                timestamp_dataset = value
                path = value.name
                index_dataset = None
            else:
                continue

            if scale_override is None:
                if storage == "matrix":
                    scale = 1_000_000.0
                elif storage == "compound":
                    scale = (
                        1.0 if np.issubdtype(value.dtype["t"], np.integer) else 1_000_000.0
                    )
                else:
                    scale = _infer_time_scale(timestamp_dataset, index_dataset)
            else:
                scale = float(scale_override)
            offset = (
                self._timestamp_offset() if offset_override is None else float(offset_override)
            )
            if index_dataset is not None:
                self.native_index_path = index_dataset.name
                self.native_index_period_us = 1000
            return EventLayout(storage, path, scale, offset)
        self.close()
        raise ValueError(
            f"No raw event arrays found in {self.path}. Supported layouts are split x/y/t/p "
            "datasets, compound x/y/t/p datasets, and MVSEC-style [x,y,t,p] matrices. "
            "Precomputed tensor-only /data HDF5 files cannot reconstruct arbitrary windows."
        )

    def _index_dataset(self, group: h5py.Group) -> h5py.Dataset | None:
        for path in (
            f"{group.name}/ms_map_idx",
            "/ms_to_idx",
            f"{group.name}/ms_to_idx",
        ):
            if path in self.handle:
                return self.handle[path]
        return None

    def _timestamp_offset(self) -> float:
        if "/t_offset" in self.handle:
            return _scalar(self.handle["/t_offset"])
        return 0.0

    def __len__(self) -> int:
        value = self.handle[self.layout.path]
        return len(value["t"]) if self.layout.storage == "split" else len(value)

    def _raw_timestamp(self, index: int) -> float:
        value = self.handle[self.layout.path]
        if self.layout.storage == "split":
            return float(value["t"][index])
        if self.layout.storage == "compound":
            return float(value[index]["t"])
        return float(value[index, 2])

    def timestamp_us(self, index: int) -> int:
        raw = self._raw_timestamp(index)
        return int(
            round(raw * self.layout.timestamp_scale_to_us + self.layout.timestamp_offset_us)
        )

    def searchsorted_us(
        self,
        value_us: int,
        side: str = "left",
        index_map: np.ndarray | None = None,
        index_period_us: int = 20,
    ) -> int:
        if index_map is not None and len(index_map):
            relative_us = value_us - self.layout.timestamp_offset_us
            slot = int(relative_us // index_period_us)
            if slot < 0:
                return 0
            if slot >= len(index_map):
                return len(self)
            index = min(max(int(index_map[slot]), 0), len(self))
            if side == "left":
                while index > 0 and self.timestamp_us(index - 1) >= value_us:
                    index -= 1
                while index < len(self) and self.timestamp_us(index) < value_us:
                    index += 1
            else:
                while index > 0 and self.timestamp_us(index - 1) > value_us:
                    index -= 1
                while index < len(self) and self.timestamp_us(index) <= value_us:
                    index += 1
            return index
        low, high = 0, len(self)
        while low < high:
            middle = (low + high) // 2
            item = self.timestamp_us(middle)
            if item < value_us or (side == "right" and item == value_us):
                low = middle + 1
            else:
                high = middle
        return low

    def bounds_us(self) -> tuple[int, int]:
        if len(self) == 0:
            raise ValueError(f"No events in {self.path}:{self.layout.path}")
        return self.timestamp_us(0), self.timestamp_us(len(self) - 1)

    def read_index_slice(self, left: int, right: int) -> tuple[np.ndarray, ...]:
        left = max(0, min(int(left), len(self)))
        right = max(left, min(int(right), len(self)))
        value = self.handle[self.layout.path]
        if self.layout.storage == "split":
            arrays = tuple(np.asarray(value[key][left:right]) for key in ("x", "y", "t", "p"))
        else:
            events = np.asarray(value[left:right])
            if self.layout.storage == "compound":
                arrays = tuple(np.asarray(events[key]) for key in ("x", "y", "t", "p"))
            else:
                arrays = tuple(events[:, index] for index in range(4))
        x, y, t, p = arrays
        t = np.rint(
            t.astype(np.float64) * self.layout.timestamp_scale_to_us
            + self.layout.timestamp_offset_us
        ).astype(np.int64)
        return x, y, t, p

    def native_timestamp_index(self) -> np.ndarray | None:
        if self.native_index_path is None:
            return None
        return np.asarray(self.handle[self.native_index_path], dtype=np.int64)

    def read_us(
        self,
        start_us: int,
        end_us: int,
        index_map: np.ndarray | None = None,
        index_period_us: int = 20,
    ) -> tuple[np.ndarray, ...]:
        left = self.searchsorted_us(start_us, "left", index_map, index_period_us)
        right = self.searchsorted_us(end_us, "right", index_map, index_period_us)
        return self.read_index_slice(left, right)

    def close(self) -> None:
        if hasattr(self, "handle"):
            self.handle.close()

    def describe(self) -> dict[str, Any]:
        first, last = self.bounds_us()
        return {
            "path": str(self.path),
            "storage": self.layout.storage,
            "event_path": self.layout.path,
            "timestamp_scale_to_us": self.layout.timestamp_scale_to_us,
            "timestamp_offset_us": self.layout.timestamp_offset_us,
            "events": len(self),
            "native_index_path": self.native_index_path,
            "native_index_period_us": self.native_index_period_us,
            "first_timestamp_us": first,
            "last_timestamp_us": last,
        }

    def __enter__(self) -> "H5EventReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
