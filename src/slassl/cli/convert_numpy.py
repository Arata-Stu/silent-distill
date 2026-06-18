from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm


def _load_events(path: Path, timestamp_scale: float) -> tuple[np.ndarray, ...]:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        if all(key in payload for key in ("x", "y", "t", "p")):
            arrays = tuple(np.asarray(payload[key]) for key in ("x", "y", "t", "p"))
        elif "events" in payload:
            events = np.asarray(payload["events"])
            arrays = tuple(events[:, index] for index in range(4))
        else:
            raise ValueError(f"{path} must contain x/y/t/p arrays or an [N,4] events array")
    elif payload.dtype.names:
        arrays = tuple(np.asarray(payload[key]) for key in ("x", "y", "t", "p"))
    elif payload.ndim == 2 and payload.shape[1] == 4:
        arrays = tuple(payload[:, index] for index in range(4))
    else:
        raise ValueError(f"Unsupported event array in {path}")
    x, y, t, p = arrays
    t = np.rint(t.astype(np.float64) * timestamp_scale).astype(np.int64)
    order = np.argsort(t, kind="stable")
    return x[order], y[order], t[order], p[order]


def _write_sequence(path: Path, events: tuple[np.ndarray, ...], source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y, t, p = events
    with h5py.File(path, "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=x.astype(np.uint16), compression="gzip", chunks=True)
        group.create_dataset("y", data=y.astype(np.uint16), compression="gzip", chunks=True)
        group.create_dataset("t", data=t.astype(np.int64), compression="gzip", chunks=True)
        group.create_dataset("p", data=p.astype(np.int8), compression="gzip", chunks=True)
        handle.attrs["source"] = str(source)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Convert NumPy event sequences to SLA-SSL data")
    result.add_argument("--input-root", required=True, type=Path)
    result.add_argument("--output-root", required=True, type=Path)
    result.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    result.add_argument("--height", required=True, type=int)
    result.add_argument("--width", required=True, type=int)
    result.add_argument(
        "--timestamp-scale",
        type=float,
        default=1.0,
        help="Multiplier converting input timestamps to microseconds",
    )
    result.add_argument("--sample-period-us", type=int, default=5000)
    result.add_argument("--long-window-us", type=int, default=5000)
    return result


def main() -> None:
    args = parser().parse_args()
    all_files = [
        path
        for split in args.splits
        for pattern in ("*.npz", "*.npy")
        for path in (args.input_root / split).rglob(pattern)
    ]
    class_names = sorted({path.parent.name for path in all_files})
    class_map = {name: index for index, name in enumerate(class_names)}
    metadata = {
        "height": args.height,
        "width": args.width,
        "time_unit": "microseconds",
        "class_map": class_map,
        "event_layout": "events/{x,y,t,p}",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split in args.splits:
        source_root = args.input_root / split
        files = sorted(source_root.rglob("*.npz")) + sorted(source_root.rglob("*.npy"))
        ssl_records, classification_records = [], []
        for source in tqdm(files, desc=f"convert {split}"):
            events = _load_events(source, args.timestamp_scale)
            if len(events[0]) == 0:
                continue
            relative_source = source.relative_to(source_root)
            sequence_name = "__".join(relative_source.with_suffix("").parts) + ".h5"
            relative = Path("sequences") / split / sequence_name
            _write_sequence(args.output_root / relative, events, source)
            first, last = int(events[2][0]), int(events[2][-1])
            label = class_map[source.parent.name]
            for timestamp in range(
                first + args.long_window_us, last + 1, args.sample_period_us
            ):
                base = {
                    "sequence": str(relative),
                    "timestamp_us": timestamp,
                    "sample_id": f"{relative.stem}:{timestamp}",
                }
                ssl_records.append(base)
                classification_records.append({**base, "label": label})
        _write_jsonl(args.output_root / "manifests" / f"{split}_ssl.jsonl", ssl_records)
        _write_jsonl(
            args.output_root / "manifests" / f"{split}_classification.jsonl",
            classification_records,
        )


if __name__ == "__main__":
    main()
