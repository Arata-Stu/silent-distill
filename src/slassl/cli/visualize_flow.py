from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from slassl.cli.finetune import build_model
from slassl.config import config_parser, load_config
from slassl.training import build_dataset
from slassl.utils import seed_everything
from slassl.visualization import (
    flow_sample_filename,
    save_flow_sample,
)


def parser() -> argparse.ArgumentParser:
    result = config_parser("Visualize optical-flow predictions and event-supported masks")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--sample-stride", type=int, default=1)
    result.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples; use 0 for all remaining samples",
    )
    result.add_argument(
        "--max-flow",
        type=float,
        help="Fixed flow magnitude mapped to full brightness; default is per-sample GT p99",
    )
    result.add_argument("--magnitude-percentile", type=float, default=99.0)
    result.add_argument("--save-arrays", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.start_index < 0 or args.sample_stride <= 0 or args.max_samples < 0:
        raise ValueError("start-index/max-samples must be non-negative and stride positive")
    if args.max_flow is not None and args.max_flow <= 0:
        raise ValueError("max-flow must be positive")

    config = load_config(args.config, args.set)
    if config.task != "flow":
        raise ValueError(f"Flow visualization requires task=flow, received {config.task}")
    seed_everything(int(config.seed))
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    dataset = build_dataset(config, "flow", config.data.manifest, augmentation=False)
    model = build_model(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    indices = list(range(args.start_index, len(dataset), args.sample_stride))
    if args.max_samples:
        indices = indices[: args.max_samples]
    if not indices:
        raise ValueError("No samples selected from the flow manifest")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    summaries: list[dict[str, Any]] = []
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for position, dataset_index in enumerate(tqdm(indices, desc="visualize optical flow")):
            sample = dataset[dataset_index]
            short = sample["short"]
            with torch.inference_mode():
                prediction = model(short.unsqueeze(0).to(device))[0].detach().cpu().numpy()
            target = sample["target"].detach().cpu().numpy()
            valid_mask = sample["valid_mask"].detach().cpu().numpy().astype(bool)
            display_voxel = short[-1] if short.ndim == 5 else short
            voxel = display_voxel.detach().cpu().numpy()
            stem = flow_sample_filename(dataset_index, sample["sample_id"])
            summary, _ = save_flow_sample(
                output_dir,
                stem,
                prediction,
                target,
                valid_mask,
                voxel,
                magnitude_scale=args.max_flow,
                magnitude_percentile=args.magnitude_percentile,
                save_arrays=args.save_arrays,
            )
            summary.update({
                "dataset_index": dataset_index,
                "sample_id": str(sample["sample_id"]),
                "timestamp_us": int(sample["timestamp_us"]),
            })
            metadata_handle.write(json.dumps(summary, separators=(",", ":")) + "\n")
            summaries.append(summary)

    summary = {
        "checkpoint": str(config.checkpoint),
        "manifest": str(config.data.manifest),
        "output_dir": str(output_dir),
        "samples": len(summaries),
        "mean_sample_aepe": float(
            np.mean([item["aepe"] for item in summaries if item["aepe"] is not None])
        )
        if any(item["aepe"] is not None for item in summaries)
        else None,
        "masked_protocol": "finite_nonzero_gt_and_event_support",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
