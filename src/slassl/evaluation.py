from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import torch
from tqdm import tqdm


def _density_group(value: float, low_max: float, high_min: float) -> str:
    if value < low_max:
        return "low"
    if value >= high_min:
        return "high"
    return "medium"


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            output[key] = value.detach().cpu().tolist() if value.ndim else float(value.item())
        else:
            output[key] = value
    return output


@torch.inference_mode()
def evaluate_classification(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    low_max: float,
    high_min: float,
) -> dict[str, Any]:
    model.eval()
    correct = {group: 0 for group in ("all", "low", "medium", "high")}
    top5_correct = {group: 0 for group in correct}
    total = {group: 0 for group in correct}
    elapsed_seconds = 0.0
    for batch in tqdm(loader, desc="evaluate classification"):
        inputs = batch["short"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        prediction = logits.argmax(dim=1).cpu()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu()
        labels = batch["label"]
        densities = batch["density"]
        for pred, candidates, label, density in zip(
            prediction, top5, labels, densities, strict=True
        ):
            group = _density_group(float(density), low_max, high_min)
            matched = int(pred == label)
            top5_matched = int((candidates == label).any())
            correct["all"] += matched
            correct[group] += matched
            top5_correct["all"] += top5_matched
            top5_correct[group] += top5_matched
            total["all"] += 1
            total[group] += 1
    output = {
        group: {
            "top1_accuracy": correct[group] / max(total[group], 1),
            "top5_accuracy": top5_correct[group] / max(total[group], 1),
            "samples": total[group],
        }
        for group in correct
    }
    output["runtime"] = {
        "model_latency_ms_per_sample": 1000.0 * elapsed_seconds / max(total["all"], 1)
    }
    return output


@torch.inference_mode()
def evaluate_detection(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    low_max: float,
    high_min: float,
) -> dict[str, Any]:
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError("Detection evaluation requires: pip install 'sla-ssl[eval]'") from exc

    model.eval()
    metrics = {
        group: MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        for group in ("all", "low", "medium", "high")
    }
    counts = {group: 0 for group in metrics}
    elapsed_seconds = 0.0
    for batch in tqdm(loader, desc="evaluate detection"):
        short = batch["short"]
        if short.ndim == 6:
            short = short[:, -1]
        images = [voxel.flatten(0, 1).to(device) for voxel in short]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        for prediction, target, density in zip(
            predictions, batch["target"], batch["density"], strict=True
        ):
            prediction = {key: value.detach().cpu() for key, value in prediction.items()}
            target = {key: value.detach().cpu() for key, value in target.items()}
            group = _density_group(float(density), low_max, high_min)
            metrics["all"].update([prediction], [target])
            metrics[group].update([prediction], [target])
            counts["all"] += 1
            counts[group] += 1
    output: dict[str, Any] = {}
    for group, metric in metrics.items():
        output[group] = {"samples": counts[group]}
        if counts[group]:
            output[group].update(_jsonable_metrics(metric.compute()))
    output["runtime"] = {
        "model_latency_ms_per_sample": 1000.0 * elapsed_seconds / max(counts["all"], 1)
    }
    return output
