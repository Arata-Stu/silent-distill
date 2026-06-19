from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slassl.data.dsec import DSEC_FLOW_TEST_SEQUENCES
from slassl.data.h5_events import H5EventReader


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Index official DSEC optical-flow test timestamps"
    )
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--search-root", required=True, type=Path)
    result.add_argument(
        "--timestamps-root",
        type=Path,
        help=(
            "Directory containing official <sequence>.csv files. Omit when each sequence "
            "already contains test_forward_flow_timestamps.csv as prepared by E-RAFT."
        ),
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a subset of the seven official test sequences for a local smoke check",
    )
    return result


def _event_paths(sequence_dir: Path) -> tuple[Path, Path]:
    candidates = (
        sequence_dir / "events_left",
        sequence_dir / "events" / "left",
    )
    for event_dir in candidates:
        events = event_dir / "events.h5"
        rectify = event_dir / "rectify_map.h5"
        if events.is_file() and rectify.is_file():
            return events, rectify
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"events.h5 and rectify_map.h5 not found under: {expected}")


def _timestamps(path: Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.int64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 3:
        raise ValueError(
            f"Expected from_timestamp_us,to_timestamp_us,file_index in {path}, "
            f"received shape {rows.shape}"
        )
    if np.any(rows[:, 1] <= rows[:, 0]):
        raise ValueError(
            f"DSEC flow timestamps must have to_timestamp_us > from_timestamp_us: {path}"
        )
    if len(np.unique(rows[:, 2])) != len(rows):
        raise ValueError(f"Duplicate DSEC output file indices in {path}")
    return rows


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as exc:
        raise ValueError(f"{path} is outside --data-root {root}") from exc


def _sequence_timestamp_files(
    search_root: Path, timestamps_root: Path | None
) -> list[tuple[Path, Path]]:
    if timestamps_root is None:
        return [
            (path.parent, path)
            for path in sorted(search_root.rglob("test_forward_flow_timestamps.csv"))
        ]
    timestamp_files = sorted(timestamps_root.resolve().rglob("*.csv"))
    result: list[tuple[Path, Path]] = []
    for timestamp_file in timestamp_files:
        sequence_name = timestamp_file.stem
        direct = search_root / sequence_name
        matches = [direct] if direct.is_dir() else [
            path for path in search_root.rglob(sequence_name) if path.is_dir()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one event directory named {sequence_name} under {search_root}, "
                f"found {len(matches)}"
            )
        result.append((matches[0], timestamp_file))
    return result


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    args = parser().parse_args()
    data_root = args.data_root.resolve()
    search_root = args.search_root.resolve()
    sequence_files = _sequence_timestamp_files(search_root, args.timestamps_root)
    if not sequence_files:
        raise RuntimeError(
            "No DSEC test timestamp CSV files found. Use --timestamps-root for the official "
            "unzipped <sequence>.csv directory."
        )
    sequence_names = {sequence_dir.name for sequence_dir, _ in sequence_files}
    if not args.allow_partial and sequence_names != DSEC_FLOW_TEST_SEQUENCES:
        raise ValueError(
            "DSEC official submission requires exactly the seven flow test sequences; "
            f"missing={sorted(DSEC_FLOW_TEST_SEQUENCES - sequence_names)}, "
            f"extra={sorted(sequence_names - DSEC_FLOW_TEST_SEQUENCES)}"
        )

    records: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    for sequence_dir, timestamp_file in tqdm(sequence_files, desc="index dsec/test flow"):
        sequence_name = sequence_dir.name
        event_path, rectify_path = _event_paths(sequence_dir)
        timestamp_rows = _timestamps(timestamp_file)
        with H5EventReader(event_path, event_group="/events") as reader:
            first_us, last_us = reader.bounds_us()

        requested_start = int(timestamp_rows[:, 0].min())
        requested_end = int(timestamp_rows[:, 1].max())
        if requested_start < first_us or requested_end > last_us:
            raise ValueError(
                f"{sequence_name}: requested flow windows "
                f"[{requested_start},{requested_end}] fall outside "
                f"event range [{first_us},{last_us}]"
            )

        event_relative = _relative(event_path, data_root)
        rectify_relative = _relative(rectify_path, data_root)
        for from_us, to_us, file_index in timestamp_rows:
            records.append(
                {
                    "sequence": event_relative,
                    "rectify_map": rectify_relative,
                    "window_start_us": int(from_us),
                    "timestamp_us": int(to_us),
                    "end_inclusive": False,
                    "sample_id": f"{sequence_name}:flow:{int(file_index):06d}",
                    "output_sequence": sequence_name,
                    "output_index": int(file_index),
                    "target_dt_us": int(to_us - from_us),
                }
            )
        sequences.append(
            {
                "name": sequence_name,
                "events": event_relative,
                "rectify_map": rectify_relative,
                "timestamps": str(timestamp_file.resolve()),
                "samples": len(timestamp_rows),
                "event_range_us": [first_us, last_us],
                "flow_range_us": [requested_start, requested_end],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "test_flow_submission.jsonl"
    _write_jsonl(manifest_path, records)
    metadata = {
        "dataset": "dsec",
        "task": "flow_submission",
        "split": "test",
        "data_root": str(data_root),
        "samples": len(records),
        "official_test_protocol": sequence_names == DSEC_FLOW_TEST_SEQUENCES,
        "sequences": sequences,
    }
    metadata_path = args.output_dir / "test_flow_submission_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": "dsec",
                "task": "flow_submission",
                "sequences": len(sequences),
                "samples": len(records),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
