from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from slassl.data.h5_events import H5EventReader


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Index timestamp-aligned optical-flow or semantic targets"
    )
    result.add_argument("--dataset", choices=["m3ed", "mvsec"], required=True)
    result.add_argument("--task", choices=["flow", "segmentation"], required=True)
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--search-root", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--split", required=True)
    result.add_argument("--camera", choices=["left", "right"], default="left")
    result.add_argument("--file-glob")
    result.add_argument(
        "--include",
        action="append",
        default=[],
        help="Keep data files whose filename contains this value; may be repeated",
    )
    result.add_argument(
        "--short-window-us",
        type=int,
        default=10_000,
        help="Causal window for point targets; flow windows are inferred from GT intervals",
    )
    result.add_argument("--target-stride", type=int, default=1)
    result.add_argument(
        "--start-fraction",
        type=float,
        default=0.0,
        help="Start of the selected recording-time interval in [0,1)",
    )
    result.add_argument(
        "--end-fraction",
        type=float,
        default=1.0,
        help="End of the selected recording-time interval in (0,1]",
    )
    result.add_argument(
        "--boundary-margin-us",
        type=int,
        default=0,
        help="Exclude this duration next to each interior temporal split boundary",
    )
    return result


def _target_path(data_path: Path, dataset: str, task: str) -> Path:
    name = data_path.name
    if dataset == "m3ed":
        suffix = "_gt_flow.h5" if task == "flow" else "_semantics.h5"
        return data_path.with_name(name.removesuffix("_data.h5") + suffix)
    if task != "flow":
        raise ValueError("MVSEC does not provide semantic segmentation labels")
    return data_path.with_name(name.removesuffix("_data.hdf5") + "_gt.hdf5")


def _target_description(
    data_path: Path,
    target_path: Path,
    dataset: str,
    task: str,
    camera: str,
    event_bounds_us: tuple[int, int],
) -> tuple[np.ndarray, str, int | None, str]:
    with h5py.File(target_path, "r") as target:
        if dataset == "m3ed" and task == "flow":
            group = f"flow/prophesee/{camera}"
            if group not in target:
                raise ValueError(f"Missing {group}")
            timestamps = np.asarray(target["ts"][:], dtype=np.int64)
            return timestamps, "m3ed_flow", None, "native_us"
        if dataset == "m3ed" and task == "segmentation":
            if "predictions" not in target or "ts" not in target:
                raise ValueError("Missing predictions or ts")
            timestamps = np.asarray(target["ts"][:], dtype=np.int64)
            return timestamps, "m3ed_segmentation", None, "native_us"
        group = f"davis/{camera}"
        if f"{group}/flow_dist" not in target or f"{group}/flow_dist_ts" not in target:
            raise ValueError(f"Missing {group}/flow_dist or flow_dist_ts")
        with h5py.File(data_path, "r") as data:
            absolute_start = (
                float(data.attrs["absolute_start_time"])
                if "absolute_start_time" in data.attrs
                else None
            )
        raw_timestamps = np.asarray(target[f"{group}/flow_dist_ts"][:], dtype=np.float64)
        timestamps, alignment = _align_mvsec_timestamps(
            raw_timestamps, absolute_start, event_bounds_us
        )
        return timestamps, "mvsec_flow", 193, alignment


def _align_mvsec_timestamps(
    raw_timestamps: np.ndarray,
    absolute_start_time: float | None,
    event_bounds_us: tuple[int, int],
) -> tuple[np.ndarray, str]:
    first_us, last_us = event_bounds_us
    candidates = {
        "absolute_seconds": np.rint(raw_timestamps * 1_000_000.0).astype(np.int64),
        "native_us": np.rint(raw_timestamps).astype(np.int64),
    }
    if absolute_start_time is not None:
        candidates["relative_seconds"] = np.rint(
            (raw_timestamps - absolute_start_time) * 1_000_000.0
        ).astype(np.int64)
    event_center = (first_us + last_us) / 2.0

    def score(timestamps: np.ndarray) -> tuple[int, float]:
        overlap = int(((timestamps >= first_us) & (timestamps <= last_us)).sum())
        center = float(np.median(timestamps)) if timestamps.size else 0.0
        return overlap, -abs(center - event_center)

    alignment, timestamps = max(candidates.items(), key=lambda item: score(item[1]))
    if score(timestamps)[0] == 0:
        ranges = ", ".join(
            f"{name}=[{int(value.min())},{int(value.max())}]"
            for name, value in candidates.items()
            if value.size
        )
        raise ValueError(
            f"MVSEC flow timestamps do not overlap event range [{first_us},{last_us}]; "
            f"{ranges}. If the event timestamps were converted to relative time, the data "
            "HDF5 must contain the absolute_start_time attribute."
        )
    return timestamps, alignment


def _target_event_interval(
    dataset: str,
    task: str,
    timestamps: np.ndarray,
    target_index: int,
    point_window_us: int,
) -> tuple[int, int] | None:
    target_timestamp_us = int(timestamps[target_index])
    if task != "flow":
        return target_timestamp_us - point_window_us, target_timestamp_us
    if dataset == "m3ed":
        if target_index == 0:
            return None
        return int(timestamps[target_index - 1]), target_timestamp_us
    if dataset == "mvsec":
        if target_index + 1 >= len(timestamps):
            return None
        return target_timestamp_us, int(timestamps[target_index + 1])
    raise ValueError(f"Unsupported flow dataset: {dataset}")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _temporal_split_bounds(
    first_us: int,
    last_us: int,
    start_fraction: float,
    end_fraction: float,
    boundary_margin_us: int,
) -> tuple[int, int]:
    event_duration_us = last_us - first_us
    selected_start_us = first_us + round(event_duration_us * start_fraction)
    selected_end_us = first_us + round(event_duration_us * end_fraction)
    if start_fraction > 0:
        selected_start_us += boundary_margin_us
    if end_fraction < 1:
        selected_end_us -= boundary_margin_us
    if selected_start_us >= selected_end_us:
        raise ValueError(
            "Temporal split is empty after applying boundary-margin-us; reduce the margin"
        )
    return selected_start_us, selected_end_us


def main() -> None:
    args = parser().parse_args()
    if args.short_window_us <= 0 or args.target_stride <= 0:
        raise ValueError("short-window-us and target-stride must be positive")
    if not 0.0 <= args.start_fraction < args.end_fraction <= 1.0:
        raise ValueError("Require 0 <= start-fraction < end-fraction <= 1")
    if args.boundary_margin_us < 0:
        raise ValueError("boundary-margin-us must be non-negative")
    data_root = args.data_root.resolve()
    search_root = args.search_root.resolve()
    pattern = args.file_glob or ("*_data.h5" if args.dataset == "m3ed" else "*_data.hdf5")
    data_paths = sorted(search_root.rglob(pattern))
    if args.include:
        data_paths = [
            path for path in data_paths if any(value in path.name for value in args.include)
        ]
    records: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for data_path in tqdm(data_paths, desc=f"index {args.dataset}/{args.task}"):
        target_path = _target_path(data_path, args.dataset, args.task)
        if not target_path.is_file():
            skipped.append({"data": str(data_path), "reason": "target file not found"})
            continue
        try:
            with H5EventReader(data_path, camera=args.camera) as reader:
                first_us, last_us = reader.bounds_us()
            timestamps, target_format, valid_height, timestamp_alignment = _target_description(
                data_path,
                target_path,
                args.dataset,
                args.task,
                args.camera,
                (first_us, last_us),
            )
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({"data": str(data_path), "reason": str(exc)})
            continue
        data_relative = str(data_path.resolve().relative_to(data_root))
        target_relative = str(target_path.resolve().relative_to(data_root))
        selected_start_us, selected_end_us = _temporal_split_bounds(
            first_us,
            last_us,
            args.start_fraction,
            args.end_fraction,
            args.boundary_margin_us,
        )
        kept = 0
        durations_us: list[int] = []
        for target_index in range(0, len(timestamps), args.target_stride):
            interval = _target_event_interval(
                args.dataset,
                args.task,
                timestamps,
                target_index,
                args.short_window_us,
            )
            if interval is None:
                continue
            window_start_us, timestamp_us = interval
            if window_start_us >= timestamp_us:
                skipped.append(
                    {
                        "data": str(data_path),
                        "reason": f"non-positive target interval at index {target_index}",
                    }
                )
                continue
            if window_start_us < first_us or timestamp_us > last_us:
                continue
            if window_start_us < selected_start_us or timestamp_us > selected_end_us:
                continue
            record: dict[str, Any] = {
                "sequence": data_relative,
                "timestamp_us": timestamp_us,
                "window_start_us": window_start_us,
                "sample_id": f"{data_path.stem}:{args.task}:{target_index}",
                "metric_sequence": data_path.stem.removesuffix("_data"),
                "target": target_relative,
                "target_index": target_index,
                "target_format": target_format,
                "target_camera": args.camera,
                "target_timestamp_us": int(timestamps[target_index]),
            }
            if valid_height is not None:
                if args.dataset != "mvsec" or "outdoor" in data_path.stem.lower():
                    record["valid_height"] = valid_height
            if args.task == "flow":
                duration_us = timestamp_us - window_start_us
                record["target_dt_us"] = duration_us
                record["valid_mask_mode"] = "event_support"
                record["window_protocol"] = "target_interval"
                durations_us.append(duration_us)
            records.append(record)
            kept += 1
        duration_summary = None
        if durations_us:
            duration_summary = {
                "min": min(durations_us),
                "median": int(np.median(np.asarray(durations_us))),
                "max": max(durations_us),
            }
        sequences.append(
            {
                "data": data_relative,
                "target": target_relative,
                "samples": kept,
                "timestamp_alignment": timestamp_alignment,
                "event_range_us": [first_us, last_us],
                "target_range_us": [int(timestamps.min()), int(timestamps.max())],
                "selected_range_us": [selected_start_us, selected_end_us],
                "target_interval_duration_us": duration_summary,
            }
        )
        if kept == 0:
            skipped.append(
                {
                    "data": str(data_path),
                    "reason": (
                        f"target range [{int(timestamps.min())},{int(timestamps.max())}] "
                        f"does not yield a full window in event range [{first_us},{last_us}]"
                    ),
                }
            )

    if not records:
        reasons = "; ".join(f"{item['data']}: {item['reason']}" for item in skipped)
        raise RuntimeError(
            f"No compatible {args.dataset} {args.task} targets found. {reasons}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / f"{args.split}_{args.task}.jsonl", records)
    metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "split": args.split,
        "data_root": str(data_root),
        "camera": args.camera,
        "short_window_us": args.short_window_us,
        "point_target_window_us": args.short_window_us,
        "window_protocol": "target_interval" if args.task == "flow" else "causal_fixed",
        "target_stride": args.target_stride,
        "start_fraction": args.start_fraction,
        "end_fraction": args.end_fraction,
        "boundary_margin_us": args.boundary_margin_us,
        "samples": len(records),
        "sequences": sequences,
        "skipped": skipped,
    }
    metadata_path = args.output_dir / f"{args.split}_{args.task}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("dataset", "task", "samples")}, indent=2))


if __name__ == "__main__":
    main()
