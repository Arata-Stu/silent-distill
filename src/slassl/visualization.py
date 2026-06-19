from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image


def flow_magnitude_scale(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 99.0,
) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    target_magnitude = np.linalg.norm(target, axis=0)
    values = target_magnitude[valid_mask & np.isfinite(target_magnitude)]
    if not values.size:
        combined = np.concatenate(
            (
                np.linalg.norm(prediction, axis=0).reshape(-1),
                target_magnitude.reshape(-1),
            )
        )
        values = combined[np.isfinite(combined) & (combined > 0)]
    return max(float(np.percentile(values, percentile)) if values.size else 1.0, 1e-6)


def flow_to_rgb(
    flow: np.ndarray,
    magnitude_scale: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render channel-first optical flow as RGB using HSV direction and magnitude."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow shaped [2,H,W], received {flow.shape}")
    if magnitude_scale <= 0:
        raise ValueError("magnitude_scale must be positive")

    finite = np.isfinite(flow).all(axis=0)
    u = np.nan_to_num(flow[0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(flow[1], nan=0.0, posinf=0.0, neginf=0.0)
    angle = np.mod(np.arctan2(v, u), 2.0 * np.pi)
    magnitude = np.hypot(u, v)
    hsv = np.empty((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.rint(angle * (255.0 / (2.0 * np.pi))).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.rint(
        np.clip(magnitude / magnitude_scale, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    rgb = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB")).copy()
    keep = finite
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != u.shape:
            raise ValueError(f"Expected mask shaped {u.shape}, received {mask.shape}")
        keep &= mask
    rgb[~keep] = 0
    return rgb


def event_support_from_voxel(voxel: np.ndarray) -> np.ndarray:
    voxel = np.asarray(voxel)
    if voxel.ndim != 4:
        raise ValueError(f"Expected event voxel shaped [P,T,H,W], received {voxel.shape}")
    return np.abs(voxel).sum(axis=(0, 1)) > 0


def event_voxel_to_rgb(voxel: np.ndarray) -> np.ndarray:
    """Render accumulated negative events in red and positive events in blue."""
    voxel = np.asarray(voxel, dtype=np.float32)
    if voxel.ndim != 4:
        raise ValueError(f"Expected event voxel shaped [P,T,H,W], received {voxel.shape}")
    channels = np.abs(voxel).sum(axis=1)
    negative = channels[0]
    positive = channels[1] if channels.shape[0] > 1 else channels[0]
    nonzero = np.concatenate((negative[negative > 0], positive[positive > 0]))
    scale = float(np.percentile(np.log1p(nonzero), 99.0)) if nonzero.size else 1.0
    negative = np.clip(np.log1p(negative) / max(scale, 1e-6), 0.0, 1.0)
    positive = np.clip(np.log1p(positive) / max(scale, 1e-6), 0.0, 1.0)
    image = np.full((*negative.shape, 3), 20.0, dtype=np.float32)
    image[..., 0] += 235.0 * negative + 44.0 * positive
    image[..., 1] += 44.0 * negative + 140.0 * positive
    image[..., 2] += 44.0 * negative + 235.0 * positive
    return np.clip(image, 0, 255).astype(np.uint8)


def flow_sample_filename(index: int, sample_id: Any) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("_")
    return f"{index:06d}_{safe_id or 'sample'}"


def save_flow_sample(
    output_dir: str | Path,
    stem: str,
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    voxel: np.ndarray,
    magnitude_scale: float | None = None,
    magnitude_percentile: float = 99.0,
    save_arrays: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output_dir = Path(output_dir)
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    event_support = event_support_from_voxel(voxel)
    scale = magnitude_scale or flow_magnitude_scale(
        prediction, target, valid_mask, magnitude_percentile
    )
    images = {
        "prediction_full": flow_to_rgb(prediction, scale),
        "prediction_event_masked": flow_to_rgb(prediction, scale, valid_mask),
        "ground_truth_full": flow_to_rgb(target, scale),
        "ground_truth_event_masked": flow_to_rgb(target, scale, valid_mask),
        "events": event_voxel_to_rgb(voxel),
        "event_support": event_support.astype(np.uint8) * 255,
        "evaluation_valid": valid_mask.astype(np.uint8) * 255,
    }
    paths = {
        "prediction_full": output_dir / "prediction" / "full" / f"{stem}.png",
        "prediction_event_masked": (
            output_dir / "prediction" / "event_masked" / f"{stem}.png"
        ),
        "ground_truth_full": output_dir / "ground_truth" / "full" / f"{stem}.png",
        "ground_truth_event_masked": (
            output_dir / "ground_truth" / "event_masked" / f"{stem}.png"
        ),
        "events": output_dir / "events" / f"{stem}.png",
        "event_support": output_dir / "masks" / "event_support" / f"{stem}.png",
        "evaluation_valid": output_dir / "masks" / "evaluation_valid" / f"{stem}.png",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(images[key])
        image.save(path)
    if save_arrays:
        array_path = output_dir / "arrays" / f"{stem}.npz"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            array_path,
            prediction=prediction,
            ground_truth=target,
            event_support=event_support,
            evaluation_valid=valid_mask,
        )

    epe = np.linalg.norm(prediction - target, axis=0)
    valid_epe = epe[valid_mask]
    summary = {
        "magnitude_scale": float(scale),
        "event_support_pixels": int(event_support.sum()),
        "evaluation_valid_pixels": int(valid_mask.sum()),
        "aepe": float(valid_epe.mean()) if valid_epe.size else None,
    }
    return summary, images
