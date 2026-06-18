from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slassl.data.h5_events import H5EventReader


def _field(array: np.ndarray, *names: str) -> np.ndarray:
    fields = array.dtype.names or ()
    for name in names:
        if name in fields:
            return array[name]
    raise ValueError(f"None of {names} found in structured array fields {fields}")


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_bbox", "_boxes", "_td", "_events"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _bbox_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*.npy"):
        if path.stem.endswith(("_bbox", "_boxes")):
            result[_normalized_stem(path)] = path
    return result


def _timestamp_index_for(path: Path, data_root: Path) -> Path | None:
    current = path.parent
    while current == data_root or data_root in current.parents:
        candidates = sorted(current.glob("50khz_*.npy"))
        if len(candidates) == 1:
            return candidates[0]
        exact = current / f"50khz_{path.stem}.npy"
        if exact.exists():
            return exact
        if current == data_root:
            break
        current = current.parent
    return None


def _detection_records(
    bbox_path: Path,
    bbox_relative: str,
    sequence: str,
    class_ids: list[int],
    first_event_us: int,
    official_filter: bool,
) -> list[dict[str, Any]]:
    boxes = np.load(bbox_path, allow_pickle=False)
    raw_classes = _field(boxes, "class_id", "class", "label").astype(int)
    timestamps = _field(boxes, "t", "ts", "timestamp").astype(np.int64)
    selected_classes = set(class_ids)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, (timestamp, raw_class) in enumerate(zip(timestamps, raw_classes, strict=True)):
        if int(raw_class) in selected_classes:
            grouped[int(timestamp)].append(index)

    w = _field(boxes, "w", "width").astype(float)
    h = _field(boxes, "h", "height").astype(float)
    records = []
    for timestamp, indices in sorted(grouped.items()):
        if official_filter and timestamp < first_event_us + 500_000:
            continue
        kept = [
            index
            for index in indices
            if not official_filter or np.hypot(w[index], h[index]) >= 60.0
        ]
        if kept:
            records.append(
                {
                    "sequence": sequence,
                    "timestamp_us": timestamp,
                    "sample_id": f"{Path(sequence).stem}:{timestamp}",
                    "annotations": bbox_relative,
                    "annotation_start": min(indices),
                    "annotation_end": max(indices) + 1,
                }
            )
    return records


def _discover_files(
    dataset: str, search_root: Path, camera: str, pattern: str | None
) -> list[Path]:
    if pattern:
        return sorted(search_root.rglob(pattern))
    if dataset == "dsec":
        return sorted(
            path
            for path in search_root.rglob("events.h5")
            if path.parent.name == camera
        )
    if dataset == "mvsec":
        return sorted(search_root.rglob("*_data.hdf5"))
    suffixes = ("*.h5", "*.hdf5")
    return sorted({path for suffix in suffixes for path in search_root.rglob(suffix)})


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Index native HDF5 event datasets without Metavision or data duplication"
    )
    result.add_argument(
        "--dataset",
        choices=["1mpx", "dsec", "m3ed", "mvsec", "custom"],
        required=True,
    )
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--search-root", type=Path, help="Defaults to DATA_ROOT/SPLIT")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--split", required=True)
    result.add_argument("--camera", choices=["left", "right"], default="left")
    result.add_argument("--file-glob", help="Override recursive HDF5 glob")
    result.add_argument("--event-group", help="Override event group/dataset path")
    result.add_argument("--timestamp-scale-to-us", type=float)
    result.add_argument("--timestamp-offset-us", type=float)
    result.add_argument("--sample-period-us", type=int, default=5000)
    result.add_argument("--long-window-us", type=int, default=5000)
    result.add_argument("--bbox-root", type=Path, help="1 Mpx annotation root; defaults to search root")
    result.add_argument("--class-ids", default="0,1,2")
    result.add_argument(
        "--official-1mpx-filter", action=argparse.BooleanOptionalAction, default=True
    )
    return result


def main() -> None:
    args = parser().parse_args()
    data_root = args.data_root.resolve()
    search_root = (args.search_root or (args.data_root / args.split)).resolve()
    files = _discover_files(args.dataset, search_root, args.camera, args.file_glob)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {search_root}")
    annotations = (
        _bbox_map((args.bbox_root or search_root).resolve())
        if args.dataset == "1mpx"
        else {}
    )
    class_ids = [int(value) for value in args.class_ids.split(",") if value]
    ssl_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    sequence_metadata = []

    for path in tqdm(files, desc=f"index {args.dataset}/{args.split}"):
        try:
            relative = str(path.resolve().relative_to(data_root))
        except ValueError as exc:
            raise ValueError(f"{path} must be located below --data-root {data_root}") from exc
        try:
            with H5EventReader(
                path,
                camera=args.camera,
                event_group=args.event_group,
                timestamp_scale_to_us=args.timestamp_scale_to_us,
                timestamp_offset_us=args.timestamp_offset_us,
            ) as reader:
                description = reader.describe()
        except (ValueError, OSError) as exc:
            if args.dataset == "m3ed":
                continue  # Ignore depth, semantic, and calibration HDF5 files.
            raise RuntimeError(f"Failed to index {path}: {exc}") from exc
        first, last = description["first_timestamp_us"], description["last_timestamp_us"]
        timestamp_index = _timestamp_index_for(path, data_root)
        timestamp_index_relative = (
            str(timestamp_index.resolve().relative_to(data_root))
            if timestamp_index is not None
            else None
        )
        for timestamp in range(first + args.long_window_us, last + 1, args.sample_period_us):
            record = {
                "sequence": relative,
                "timestamp_us": timestamp,
                "sample_id": f"{path.stem}:{timestamp}",
            }
            if timestamp_index_relative is not None:
                record["timestamp_index"] = timestamp_index_relative
                record["timestamp_index_period_us"] = 20
            ssl_records.append(record)
        description["path"] = relative
        sequence_metadata.append(description)

        bbox_path = annotations.get(_normalized_stem(path))
        if bbox_path is not None:
            try:
                bbox_relative = str(bbox_path.resolve().relative_to(data_root))
            except ValueError as exc:
                raise ValueError(
                    f"{bbox_path} must be located below --data-root {data_root}"
                ) from exc
            sequence_detection_records = _detection_records(
                bbox_path,
                bbox_relative,
                relative,
                class_ids,
                first,
                args.official_1mpx_filter,
            )
            if timestamp_index_relative is not None:
                for record in sequence_detection_records:
                    record["timestamp_index"] = timestamp_index_relative
                    record["timestamp_index_period_us"] = 20
            detection_records.extend(sequence_detection_records)

    if not sequence_metadata:
        raise RuntimeError(f"No compatible event HDF5 sequences found under {search_root}")
    if args.dataset == "1mpx" and not detection_records:
        raise RuntimeError(
            "No 1 Mpx bbox NPY files matched the HDF5 sequences. Check --bbox-root and file stems."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / f"{args.split}_ssl.jsonl", ssl_records)
    if args.dataset == "1mpx":
        _write_jsonl(args.output_dir / f"{args.split}_detection.jsonl", detection_records)
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "data_root": str(data_root),
        "camera": args.camera,
        "sequences": sequence_metadata,
        "ssl_samples": len(ssl_records),
        "detection_samples": len(detection_records),
        "class_map": {str(raw): index + 1 for index, raw in enumerate(class_ids)},
    }
    (args.output_dir / f"{args.split}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    summary_keys = ("dataset", "split", "ssl_samples", "detection_samples")
    print(json.dumps({key: metadata[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
