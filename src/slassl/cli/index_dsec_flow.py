from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slassl.cli.index_dense import _temporal_split_bounds
from slassl.cli.index_dsec_test import _event_paths, _relative, _write_jsonl
from slassl.data.h5_events import H5EventReader


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Index official DSEC train/validation forward optical flow"
    )
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--events-root", required=True, type=Path)
    result.add_argument("--flow-root", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--split", choices=["train", "val"], required=True)
    result.add_argument(
        "--include",
        action="append",
        default=[],
        help="Keep sequence names containing this value; may be repeated",
    )
    result.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Drop sequence names containing this value; may be repeated",
    )
    result.add_argument("--target-stride", type=int, default=1)
    result.add_argument("--start-fraction", type=float, default=0.0)
    result.add_argument("--end-fraction", type=float, default=1.0)
    result.add_argument("--boundary-margin-us", type=int, default=0)
    return result


def _event_sequences(root: Path) -> dict[str, Path]:
    sequences: dict[str, Path] = {}
    for event_file in root.rglob("events.h5"):
        event_dir = event_file.parent
        if event_dir.name == "events_left":
            sequence_dir = event_dir.parent
        elif event_dir.name == "left" and event_dir.parent.name == "events":
            sequence_dir = event_dir.parent.parent
        else:
            continue
        if sequence_dir.name in sequences:
            raise ValueError(f"Duplicate DSEC event sequence name: {sequence_dir.name}")
        sequences[sequence_dir.name] = sequence_dir
    return sequences


def _flow_sequences(root: Path, names: set[str]) -> dict[str, tuple[Path, Path]]:
    sequences: dict[str, tuple[Path, Path]] = {}
    for timestamp_file in root.rglob("forward_timestamps.txt"):
        matches = [name for name in names if name in timestamp_file.parts]
        if len(matches) != 1:
            continue
        sequence_name = matches[0]
        flow_dir = timestamp_file.parent / "forward"
        if not flow_dir.is_dir():
            continue
        if sequence_name in sequences:
            raise ValueError(f"Duplicate DSEC flow sequence name: {sequence_name}")
        sequences[sequence_name] = (timestamp_file, flow_dir)
    return sequences


def _flow_timestamps(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.int64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 2:
        raise ValueError(f"Expected from_timestamp_us,to_timestamp_us in {path}")
    if np.any(rows[:, 1] <= rows[:, 0]):
        raise ValueError(f"Invalid forward flow timestamp interval in {path}")
    return rows


def main() -> None:
    args = parser().parse_args()
    if args.target_stride <= 0:
        raise ValueError("target-stride must be positive")
    if not 0.0 <= args.start_fraction < args.end_fraction <= 1.0:
        raise ValueError("Require 0 <= start-fraction < end-fraction <= 1")
    if args.boundary_margin_us < 0:
        raise ValueError("boundary-margin-us must be non-negative")

    data_root = args.data_root.resolve()
    events_root = args.events_root.resolve()
    flow_root = args.flow_root.resolve()
    event_sequences = _event_sequences(events_root)
    flow_sequences = _flow_sequences(flow_root, set(event_sequences))
    names = sorted(set(event_sequences) & set(flow_sequences))
    if args.include:
        names = [name for name in names if any(value in name for value in args.include)]
    if args.exclude:
        names = [name for name in names if not any(value in name for value in args.exclude)]
    if not names:
        raise RuntimeError("No matching DSEC event and forward-flow sequences found")

    records: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for sequence_name in tqdm(names, desc=f"index dsec/{args.split} flow"):
        sequence_dir = event_sequences[sequence_name]
        event_path, rectify_path = _event_paths(sequence_dir)
        timestamp_file, flow_dir = flow_sequences[sequence_name]
        timestamp_rows = _flow_timestamps(timestamp_file)
        flow_files = sorted(flow_dir.glob("*.png"))
        if len(flow_files) != len(timestamp_rows):
            raise ValueError(
                f"{sequence_name}: {len(flow_files)} flow PNGs but "
                f"{len(timestamp_rows)} timestamp rows"
            )
        with H5EventReader(event_path, event_group="/events") as reader:
            first_us, last_us = reader.bounds_us()
        selected_start, selected_end = _temporal_split_bounds(
            first_us,
            last_us,
            args.start_fraction,
            args.end_fraction,
            args.boundary_margin_us,
        )
        kept = 0
        for index in range(0, len(timestamp_rows), args.target_stride):
            from_us, to_us = (int(value) for value in timestamp_rows[index])
            if from_us < first_us or to_us > last_us:
                continue
            if from_us < selected_start or to_us > selected_end:
                continue
            records.append(
                {
                    "sequence": _relative(event_path, data_root),
                    "rectify_map": _relative(rectify_path, data_root),
                    "window_start_us": from_us,
                    "timestamp_us": to_us,
                    "end_inclusive": False,
                    "sample_id": f"{sequence_name}:flow:{index:06d}",
                    "metric_sequence": sequence_name,
                    "target": _relative(flow_files[index], data_root),
                    "target_format": "dsec_flow",
                    "target_dt_us": to_us - from_us,
                    "valid_mask_mode": "target",
                    "window_protocol": "target_interval",
                }
            )
            kept += 1
        sequences.append(
            {
                "name": sequence_name,
                "events": _relative(event_path, data_root),
                "rectify_map": _relative(rectify_path, data_root),
                "timestamps": str(timestamp_file),
                "samples": kept,
                "event_range_us": [first_us, last_us],
                "selected_range_us": [selected_start, selected_end],
            }
        )

    if not records:
        raise RuntimeError("DSEC temporal selection produced no flow samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.split}_flow.jsonl"
    _write_jsonl(manifest_path, records)
    metadata = {
        "dataset": "dsec",
        "task": "flow",
        "split": args.split,
        "data_root": str(data_root),
        "samples": len(records),
        "target_stride": args.target_stride,
        "start_fraction": args.start_fraction,
        "end_fraction": args.end_fraction,
        "boundary_margin_us": args.boundary_margin_us,
        "sequences": sequences,
    }
    metadata_path = args.output_dir / f"{args.split}_flow_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": "dsec",
                "task": "flow",
                "split": args.split,
                "sequences": len(sequences),
                "samples": len(records),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
