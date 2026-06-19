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
    result.add_argument("--short-window-us", type=int, default=10_000)
    result.add_argument("--target-stride", type=int, default=1)
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
    data_path: Path, target_path: Path, dataset: str, task: str, camera: str
) -> tuple[np.ndarray, str, int | None]:
    with h5py.File(target_path, "r") as target:
        if dataset == "m3ed" and task == "flow":
            group = f"flow/prophesee/{camera}"
            if group not in target:
                raise ValueError(f"Missing {group}")
            timestamps = np.asarray(target["ts"][:], dtype=np.int64)
            return timestamps, "m3ed_flow", None
        if dataset == "m3ed" and task == "segmentation":
            if "predictions" not in target or "ts" not in target:
                raise ValueError("Missing predictions or ts")
            timestamps = np.asarray(target["ts"][:], dtype=np.int64)
            return timestamps, "m3ed_segmentation", None
        group = f"davis/{camera}"
        if f"{group}/flow_dist" not in target or f"{group}/flow_dist_ts" not in target:
            raise ValueError(f"Missing {group}/flow_dist or flow_dist_ts")
        with h5py.File(data_path, "r") as data:
            if "absolute_start_time" not in data.attrs:
                raise ValueError("MVSEC data HDF5 is missing absolute_start_time")
            absolute_start = float(data.attrs["absolute_start_time"])
        timestamps = np.rint(
            (np.asarray(target[f"{group}/flow_dist_ts"][:]) - absolute_start) * 1_000_000.0
        ).astype(np.int64)
        return timestamps, "mvsec_flow", 193


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    args = parser().parse_args()
    if args.short_window_us <= 0 or args.target_stride <= 0:
        raise ValueError("short-window-us and target-stride must be positive")
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
            timestamps, target_format, valid_height = _target_description(
                data_path, target_path, args.dataset, args.task, args.camera
            )
            with H5EventReader(data_path, camera=args.camera) as reader:
                first_us, last_us = reader.bounds_us()
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({"data": str(data_path), "reason": str(exc)})
            continue
        data_relative = str(data_path.resolve().relative_to(data_root))
        target_relative = str(target_path.resolve().relative_to(data_root))
        kept = 0
        for target_index in range(0, len(timestamps), args.target_stride):
            timestamp_us = int(timestamps[target_index])
            if timestamp_us - args.short_window_us < first_us or timestamp_us > last_us:
                continue
            record: dict[str, Any] = {
                "sequence": data_relative,
                "timestamp_us": timestamp_us,
                "sample_id": f"{data_path.stem}:{args.task}:{target_index}",
                "target": target_relative,
                "target_index": target_index,
                "target_format": target_format,
                "target_camera": args.camera,
            }
            if valid_height is not None:
                record["valid_height"] = valid_height
            if args.task == "flow" and target_index > 0:
                record["target_dt_us"] = int(
                    timestamps[target_index] - timestamps[target_index - 1]
                )
            records.append(record)
            kept += 1
        sequences.append(
            {"data": data_relative, "target": target_relative, "samples": kept}
        )

    if not records:
        raise RuntimeError(f"No compatible {args.dataset} {args.task} targets found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / f"{args.split}_{args.task}.jsonl", records)
    metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "split": args.split,
        "data_root": str(data_root),
        "camera": args.camera,
        "short_window_us": args.short_window_us,
        "target_stride": args.target_stride,
        "samples": len(records),
        "sequences": sequences,
        "skipped": skipped,
    }
    metadata_path = args.output_dir / f"{args.split}_{args.task}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("dataset", "task", "samples")}, indent=2))


if __name__ == "__main__":
    main()
