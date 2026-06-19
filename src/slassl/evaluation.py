from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import torch
import torch.nn.functional as F
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


def _dense_groups() -> tuple[str, ...]:
    return ("all", "low", "medium", "high")


def _flow_metric_sums() -> dict[str, float]:
    return {key: 0.0 for key in ("epe", "1pe", "2pe", "3pe", "angle")}


def _flow_metric_summary(
    sums: dict[str, float], valid_pixels: int, samples: int
) -> dict[str, Any]:
    count = max(valid_pixels, 1)
    return {
        "aepe": sums["epe"] / count,
        "1pe": 100.0 * sums["1pe"] / count,
        "2pe": 100.0 * sums["2pe"] / count,
        "3pe": 100.0 * sums["3pe"] / count,
        "aae_degrees": sums["angle"] / count,
        "valid_pixels": valid_pixels,
        "samples": samples,
    }


@torch.inference_mode()
def evaluate_flow(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    low_max: float,
    high_min: float,
) -> dict[str, Any]:
    model.eval()
    groups = _dense_groups()
    sums = {group: _flow_metric_sums() for group in groups}
    valid_pixels = {group: 0 for group in groups}
    samples = {group: 0 for group in groups}
    sequence_sums: dict[str, dict[str, float]] = {}
    sequence_valid_pixels: dict[str, int] = {}
    sequence_samples: dict[str, int] = {}
    elapsed_seconds = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc="evaluate optical flow"):
        inputs = batch["short"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid_masks = batch["valid_mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        total_samples += len(inputs)
        metric_sequences = batch.get("metric_sequence", [None] * len(inputs))
        for prediction, target, valid, density, metric_sequence in zip(
            predictions,
            targets,
            valid_masks,
            batch["density"],
            metric_sequences,
            strict=True,
        ):
            count = int(valid.sum().item())
            if not count:
                continue
            group = _density_group(float(density), low_max, high_min)
            selected_prediction = prediction[:, valid].T
            selected_target = target[:, valid].T
            epe = (selected_prediction - selected_target).norm(dim=1)
            prediction_3d = torch.cat(
                (selected_prediction, torch.ones((count, 1), device=device)), dim=1
            )
            target_3d = torch.cat(
                (selected_target, torch.ones((count, 1), device=device)), dim=1
            )
            cosine = F.cosine_similarity(prediction_3d, target_3d, dim=1).clamp(-1, 1)
            angle = torch.acos(cosine) * (180.0 / torch.pi)
            for selected_group in ("all", group):
                sums[selected_group]["epe"] += float(epe.sum().item())
                sums[selected_group]["1pe"] += float((epe > 1).sum().item())
                sums[selected_group]["2pe"] += float((epe > 2).sum().item())
                sums[selected_group]["3pe"] += float((epe > 3).sum().item())
                sums[selected_group]["angle"] += float(angle.sum().item())
                valid_pixels[selected_group] += count
                samples[selected_group] += 1
            if metric_sequence is not None:
                sequence = str(metric_sequence)
                sequence_sums.setdefault(sequence, _flow_metric_sums())
                sequence_valid_pixels.setdefault(sequence, 0)
                sequence_samples.setdefault(sequence, 0)
                sequence_sums[sequence]["epe"] += float(epe.sum().item())
                sequence_sums[sequence]["1pe"] += float((epe > 1).sum().item())
                sequence_sums[sequence]["2pe"] += float((epe > 2).sum().item())
                sequence_sums[sequence]["3pe"] += float((epe > 3).sum().item())
                sequence_sums[sequence]["angle"] += float(angle.sum().item())
                sequence_valid_pixels[sequence] += count
                sequence_samples[sequence] += 1
    output: dict[str, Any] = {
        group: _flow_metric_summary(sums[group], valid_pixels[group], samples[group])
        for group in groups
    }
    if sequence_sums:
        per_sequence = {
            sequence: _flow_metric_summary(
                sequence_sums[sequence],
                sequence_valid_pixels[sequence],
                sequence_samples[sequence],
            )
            for sequence in sorted(sequence_sums)
        }
        metric_names = ("aepe", "1pe", "2pe", "3pe", "aae_degrees")
        output["per_sequence"] = per_sequence
        output["sequence_average"] = {
            key: sum(values[key] for values in per_sequence.values()) / len(per_sequence)
            for key in metric_names
        }
        output["sequence_average"].update(
            {
                "sequences": len(per_sequence),
                "valid_pixels": sum(sequence_valid_pixels.values()),
                "samples": sum(sequence_samples.values()),
            }
        )
    output["runtime"] = {
        "model_latency_ms_per_sample": 1000.0 * elapsed_seconds / max(total_samples, 1)
    }
    return output


def _segmentation_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    true_positive = confusion.diagonal().float()
    ground_truth = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    union = ground_truth + predicted - true_positive
    present = union > 0
    iou = torch.zeros_like(union)
    iou[present] = true_positive[present] / union[present]
    return {
        "mean_iou": float(iou[present].mean().item()) if present.any() else 0.0,
        "overall_accuracy": float(true_positive.sum().item() / max(ground_truth.sum().item(), 1)),
        "per_class_iou": iou.tolist(),
        "valid_pixels": int(ground_truth.sum().item()),
    }


@torch.inference_mode()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    low_max: float,
    high_min: float,
) -> dict[str, Any]:
    model.eval()
    groups = _dense_groups()
    confusion = {
        group: torch.zeros((num_classes, num_classes), dtype=torch.int64) for group in groups
    }
    samples = {group: 0 for group in groups}
    elapsed_seconds = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc="evaluate semantic segmentation"):
        inputs = batch["short"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = model(inputs).argmax(dim=1).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        total_samples += len(inputs)
        for prediction, target, density in zip(
            predictions, batch["target"], batch["density"], strict=True
        ):
            valid = target != ignore_index
            valid &= (target >= 0) & (target < num_classes)
            encoded = target[valid].long() * num_classes + prediction[valid].long()
            sample_confusion = torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            group = _density_group(float(density), low_max, high_min)
            for selected_group in ("all", group):
                confusion[selected_group] += sample_confusion
                samples[selected_group] += 1
    output = {
        group: {**_segmentation_metrics(confusion[group]), "samples": samples[group]}
        for group in groups
    }
    output["runtime"] = {
        "model_latency_ms_per_sample": 1000.0 * elapsed_seconds / max(total_samples, 1)
    }
    return output
