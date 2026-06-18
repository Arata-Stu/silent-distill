from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slassl.config import config_parser, load_config
from slassl.data.h5_events import H5EventReader
from slassl.data.indexing import load_timestamp_index


def main() -> None:
    args = config_parser("Compute causal-window event-density statistics").parse_args()
    config = load_config(args.config, args.set)
    manifest = Path(config.data.manifest)
    with manifest.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    limit = int(config.get("max_samples", 0))
    if limit > 0 and len(records) > limit:
        indices = np.linspace(0, len(records) - 1, num=limit, dtype=int)
        records = [records[index] for index in indices]
    readers: dict[str, H5EventReader] = {}
    timestamp_indices: dict[str, np.ndarray] = {}
    densities = []
    window_us = int(config.data.short_window_us)
    denominator = int(config.data.height) * int(config.data.width) * (window_us / 1000.0)
    try:
        for record in records:
            sequence = record["sequence"]
            if sequence not in readers:
                readers[sequence] = H5EventReader(
                    Path(config.data.root) / sequence,
                    camera=str(config.data.get("event_camera", "left")),
                    event_group=config.data.get("event_group"),
                    timestamp_scale_to_us=(
                        float(config.data.timestamp_scale_to_us)
                        if config.data.get("timestamp_scale_to_us") is not None
                        else None
                    ),
                    timestamp_offset_us=(
                        float(config.data.timestamp_offset_us)
                        if config.data.get("timestamp_offset_us") is not None
                        else None
                    ),
                )
            reader = readers[sequence]
            index_map = None
            if record.get("timestamp_index"):
                index_path = record["timestamp_index"]
                if index_path not in timestamp_indices:
                    timestamp_indices[index_path] = load_timestamp_index(
                        Path(config.data.root) / index_path,
                        str(config.data.get("event_camera", "left")),
                    )
                index_map = timestamp_indices[index_path]
            end = int(record["timestamp_us"])
            period = int(record.get("timestamp_index_period_us", 20))
            left = reader.searchsorted_us(end - window_us, "left", index_map, period)
            right = reader.searchsorted_us(end, "right", index_map, period)
            densities.append((right - left) / denominator)
    finally:
        for reader in readers.values():
            reader.close()
    values = np.asarray(densities)
    payload = {
        "samples": len(values),
        "unit": "events/pixel/ms",
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.9)
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
