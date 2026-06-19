from __future__ import annotations

import json
from pathlib import Path

import torch

from slassl.cli.finetune import build_model
from slassl.config import config_parser, load_config
from slassl.evaluation import (
    evaluate_classification,
    evaluate_detection,
    evaluate_flow,
    evaluate_segmentation,
)
from slassl.training import build_dataset, build_loader
from slassl.utils import dump_json, seed_everything


def main() -> None:
    args = config_parser("Evaluate a downstream SLA-SSL checkpoint").parse_args()
    config = load_config(args.config, args.set)
    seed_everything(int(config.seed))
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    dataset = build_dataset(config, config.task, config.data.manifest, augmentation=False)
    loader, _ = build_loader(dataset, config, rank=0, world_size=1, shuffle=False)
    model = build_model(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    low_max = float(config.evaluation.low_density_max)
    high_min = float(config.evaluation.high_density_min)
    if config.task == "classification":
        metrics = evaluate_classification(model, loader, device, low_max, high_min)
    elif config.task == "detection":
        metrics = evaluate_detection(model, loader, device, low_max, high_min)
    elif config.task == "flow":
        metrics = evaluate_flow(model, loader, device, low_max, high_min)
    elif config.task == "segmentation":
        metrics = evaluate_segmentation(
            model,
            loader,
            device,
            int(config.model.num_classes),
            int(config.data.get("segmentation_ignore_index", 255)),
            low_max,
            high_min,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")
    metrics["metadata"] = {
        "accumulation_time_ms": float(config.data.short_window_us) / 1000.0,
        "checkpoint": str(config.checkpoint),
    }
    dump_json(metrics, config.output_file)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
