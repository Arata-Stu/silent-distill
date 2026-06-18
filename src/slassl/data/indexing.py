from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from slassl.data.h5_events import H5EventReader


def load_timestamp_index(path: str | Path, camera: str = "left") -> np.ndarray:
    """Load an F3-style timestamp index, including scalar object camera dictionaries."""
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if camera not in payload:
            raise KeyError(f"Camera {camera!r} not found in timestamp index {path}")
        payload = payload[camera]
    elif isinstance(payload, np.ndarray) and payload.shape == ():
        mapping = payload.item()
        if not isinstance(mapping, dict) or camera not in mapping:
            raise ValueError(f"Expected a camera dictionary in timestamp index {path}")
        payload = mapping[camera]
    index = np.asarray(payload)
    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError(f"Timestamp index must be a 1D integer array: {path}")
    if index.size and np.any(index[1:] < index[:-1]):
        raise ValueError(f"Timestamp index is not monotonic: {path}")
    return index.astype(np.int64, copy=False)


def validate_timestamp_index(
    reader: "H5EventReader",
    index: np.ndarray,
    period_us: int,
    checks: int = 128,
) -> dict[str, Any]:
    if period_us <= 0:
        raise ValueError("period_us must be positive")
    if index.ndim != 1:
        raise ValueError("index must be one-dimensional")
    if len(index) == 0:
        return {"valid": False, "reason": "empty index", "checks": 0}

    slots = np.unique(np.linspace(0, len(index) - 1, min(checks, len(index)), dtype=int))
    errors = []
    out_of_bounds = 0
    boundary_violations = 0
    for slot in slots:
        target_us = int(round(reader.layout.timestamp_offset_us + slot * period_us))
        mapped = int(index[slot])
        if mapped < 0 or mapped > len(reader):
            out_of_bounds += 1
            continue
        expected = reader.searchsorted_us(target_us, "left")
        errors.append(abs(mapped - expected))
        if mapped > 0 and reader.timestamp_us(mapped - 1) >= target_us:
            boundary_violations += 1
        if mapped < len(reader) and reader.timestamp_us(mapped) < target_us:
            boundary_violations += 1

    error_array = np.asarray(errors, dtype=np.int64)
    max_error = int(error_array.max()) if error_array.size else None
    return {
        "valid": out_of_bounds == 0 and boundary_violations == 0 and (max_error or 0) <= 1,
        "checks": int(len(slots)),
        "period_us": int(period_us),
        "entries": int(len(index)),
        "out_of_bounds": int(out_of_bounds),
        "boundary_violations": int(boundary_violations),
        "mean_index_error": float(error_array.mean()) if error_array.size else None,
        "p95_index_error": float(np.quantile(error_array, 0.95)) if error_array.size else None,
        "max_index_error": max_error,
    }
