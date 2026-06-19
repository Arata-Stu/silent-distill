from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


DSEC_FLOW_SCALE = 128.0
DSEC_FLOW_OFFSET = 2**15
DSEC_FLOW_MIN = -DSEC_FLOW_OFFSET / DSEC_FLOW_SCALE
DSEC_FLOW_MAX = (2**16 - 1 - DSEC_FLOW_OFFSET) / DSEC_FLOW_SCALE
DSEC_FLOW_TEST_SEQUENCES = frozenset(
    {
        "interlaken_00_b",
        "interlaken_01_a",
        "thun_01_a",
        "thun_01_b",
        "zurich_city_12_a",
        "zurich_city_14_c",
        "zurich_city_15_a",
    }
)


def _png_module():
    try:
        import png
    except ImportError as exc:
        raise RuntimeError(
            "DSEC 16-bit RGB PNG support requires the 'pypng' package"
        ) from exc
    return png


def encode_dsec_flow(flow: np.ndarray, valid_value: int = 1) -> tuple[np.ndarray, int]:
    """Encode a 2xHxW flow field using the official DSEC uint16 convention."""
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected flow with shape (2,H,W), received {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("DSEC submission flow contains NaN or infinity")
    if valid_value not in {0, 1}:
        raise ValueError("DSEC PNG third-channel values must be 0 or 1")

    clipped = np.clip(array, DSEC_FLOW_MIN, DSEC_FLOW_MAX)
    clipped_values = int(np.count_nonzero(clipped != array))
    encoded = np.empty((*array.shape[1:], 3), dtype=np.uint16)
    encoded[..., :2] = np.rint(
        np.moveaxis(clipped, 0, -1) * DSEC_FLOW_SCALE + DSEC_FLOW_OFFSET
    ).astype(np.uint16)
    encoded[..., 2] = valid_value
    return encoded, clipped_values


def decode_dsec_flow(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(encoded)
    if array.dtype != np.uint16 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 uint16 DSEC flow image, received {array.shape}")
    flow = (array[..., :2].astype(np.float32) - DSEC_FLOW_OFFSET) / DSEC_FLOW_SCALE
    return np.moveaxis(flow, -1, 0), array[..., 2] == 1


def write_dsec_flow_png(path: str | Path, flow: np.ndarray, valid_value: int = 1) -> int:
    encoded, clipped_values = encode_dsec_flow(flow, valid_value=valid_value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    png = _png_module()
    writer = png.Writer(
        width=encoded.shape[1],
        height=encoded.shape[0],
        greyscale=False,
        alpha=False,
        bitdepth=16,
    )
    rows: Iterable[list[int]] = (
        row.reshape(-1).tolist() for row in encoded
    )
    with output.open("wb") as handle:
        writer.write(handle, rows)
    return clipped_values


def read_dsec_flow_png(path: str | Path) -> np.ndarray:
    png = _png_module()
    width, height, rows, info = png.Reader(filename=str(path)).read()
    if int(info["bitdepth"]) != 16 or int(info["planes"]) != 3:
        raise ValueError(
            f"DSEC flow PNG must be 16-bit RGB, received "
            f"bitdepth={info['bitdepth']} planes={info['planes']}"
        )
    array = np.vstack(
        [np.fromiter(row, dtype=np.uint16, count=width * 3) for row in rows]
    )
    return array.reshape(height, width, 3)
