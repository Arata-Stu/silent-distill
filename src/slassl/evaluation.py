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
    sums: dict[str, float], count: int, valid_pixels: int, samples: int
) -> dict[str, Any]:
    denominator = max(count, 1)
    return {
        "aepe": sums["epe"] / denominator,
        "1pe": 100.0 * sums["1pe"] / denominator,
        "2pe": 100.0 * sums["2pe"] / denominator,
        "3pe": 100.0 * sums["3pe"] / denominator,
        "aae_degrees": sums["angle"] / denominator,
        "valid_pixels": valid_pixels,
        "samples": samples,
    }


def _flow_protocol_state(groups: tuple[str, ...]) -> dict[str, Any]:
    return {
        "pixel_sums": {group: _flow_metric_sums() for group in groups},
        "sample_sums": {group: _flow_metric_sums() for group in groups},
        "valid_pixels": {group: 0 for group in groups},
        "samples": {group: 0 for group in groups},
        "sequence_pixel_sums": {},
        "sequence_sample_sums": {},
        "sequence_valid_pixels": {},
        "sequence_samples": {},
    }


def _accumulate_flow_metrics(
    sums: dict[str, float], epe: torch.Tensor, angle: torch.Tensor, reduce: str
) -> None:
    if reduce == "pixel":
        sums["epe"] += float(epe.sum().item())
        sums["1pe"] += float((epe > 1).sum().item())
        sums["2pe"] += float((epe > 2).sum().item())
        sums["3pe"] += float((epe > 3).sum().item())
        sums["angle"] += float(angle.sum().item())
        return
    if reduce != "sample":
        raise ValueError(f"Unknown flow metric reduction: {reduce}")
    sums["epe"] += float(epe.mean().item())
    sums["1pe"] += float((epe > 1).float().mean().item())
    sums["2pe"] += float((epe > 2).float().mean().item())
    sums["3pe"] += float((epe > 3).float().mean().item())
    sums["angle"] += float(angle.mean().item())


def _flow_protocol_summary(
    state: dict[str, Any], groups: tuple[str, ...]
) -> dict[str, Any]:
    density_groups: dict[str, Any] = {}
    for group in groups:
        density_groups[group] = {
            "sample_average": _flow_metric_summary(
                state["sample_sums"][group],
                state["samples"][group],
                state["valid_pixels"][group],
                state["samples"][group],
            ),
            "pixel_average": _flow_metric_summary(
                state["pixel_sums"][group],
                state["valid_pixels"][group],
                state["valid_pixels"][group],
                state["samples"][group],
            ),
        }

    per_sequence = {
        sequence: _flow_metric_summary(
            sums,
            state["sequence_samples"][sequence],
            state["sequence_valid_pixels"][sequence],
            state["sequence_samples"][sequence],
        )
        for sequence, sums in sorted(state["sequence_sample_sums"].items())
    }
    per_sequence_pixel_average = {
        sequence: _flow_metric_summary(
            sums,
            state["sequence_valid_pixels"][sequence],
            state["sequence_valid_pixels"][sequence],
            state["sequence_samples"][sequence],
        )
        for sequence, sums in sorted(state["sequence_pixel_sums"].items())
    }
    output: dict[str, Any] = {
        "sample_average": density_groups["all"]["sample_average"],
        "pixel_average": density_groups["all"]["pixel_average"],
        "density_groups": density_groups,
        "per_sequence": per_sequence,
        "per_sequence_pixel_average": per_sequence_pixel_average,
    }
    if per_sequence:
        metric_names = ("aepe", "1pe", "2pe", "3pe", "aae_degrees")
        output["sequence_average"] = {
            key: sum(values[key] for values in per_sequence.values()) / len(per_sequence)
            for key in metric_names
        }
        output["sequence_average"].update(
            {
                "sequences": len(per_sequence),
                "valid_pixels": sum(state["sequence_valid_pixels"].values()),
                "samples": sum(state["sequence_samples"].values()),
            }
        )
        output["sequence_pixel_average"] = {
            key: sum(values[key] for values in per_sequence_pixel_average.values())
            / len(per_sequence_pixel_average)
            for key in metric_names
        }
        output["sequence_pixel_average"].update(
            {
                "sequences": len(per_sequence_pixel_average),
                "valid_pixels": sum(state["sequence_valid_pixels"].values()),
                "samples": sum(state["sequence_samples"].values()),
            }
        )
    return output


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
    protocol_states = {
        "event_supported": _flow_protocol_state(groups),
        "target_valid": _flow_protocol_state(groups),
    }
    elapsed_seconds = 0.0
    total_samples = 0
    for batch in tqdm(loader, desc="evaluate optical flow"):
        inputs = batch["short"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        protocol_masks = {
            "event_supported": batch["valid_mask"].to(device, non_blocking=True),
            "target_valid": batch.get("target_valid_mask", batch["valid_mask"]).to(
                device, non_blocking=True
            ),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_seconds += time.perf_counter() - started
        total_samples += len(inputs)
        metric_sequences = batch.get("metric_sequence", [None] * len(inputs))
        for sample_index, (prediction, target, density, metric_sequence) in enumerate(
            zip(
                predictions,
                targets,
                batch["density"],
                metric_sequences,
                strict=True,
            )
        ):
            group = _density_group(float(density), low_max, high_min)
            for protocol_name, masks in protocol_masks.items():
                valid = masks[sample_index]
                count = int(valid.sum().item())
                if not count:
                    continue
                selected_prediction = prediction[:, valid].T
                selected_target = target[:, valid].T
                if not torch.isfinite(selected_prediction).all():
                    raise ValueError(
                        f"Non-finite flow prediction on valid pixels for protocol {protocol_name}"
                    )
                epe = (selected_prediction - selected_target).norm(dim=1)
                prediction_3d = torch.cat(
                    (selected_prediction, torch.ones((count, 1), device=device)), dim=1
                )
                target_3d = torch.cat(
                    (selected_target, torch.ones((count, 1), device=device)), dim=1
                )
                cosine = F.cosine_similarity(prediction_3d, target_3d, dim=1).clamp(-1, 1)
                angle = torch.acos(cosine) * (180.0 / torch.pi)
                state = protocol_states[protocol_name]
                for selected_group in ("all", group):
                    _accumulate_flow_metrics(
                        state["pixel_sums"][selected_group], epe, angle, "pixel"
                    )
                    _accumulate_flow_metrics(
                        state["sample_sums"][selected_group], epe, angle, "sample"
                    )
                    state["valid_pixels"][selected_group] += count
                    state["samples"][selected_group] += 1
                if metric_sequence is not None:
                    sequence = str(metric_sequence)
                    state["sequence_pixel_sums"].setdefault(
                        sequence, _flow_metric_sums()
                    )
                    state["sequence_sample_sums"].setdefault(
                        sequence, _flow_metric_sums()
                    )
                    state["sequence_valid_pixels"].setdefault(sequence, 0)
                    state["sequence_samples"].setdefault(sequence, 0)
                    _accumulate_flow_metrics(
                        state["sequence_pixel_sums"][sequence], epe, angle, "pixel"
                    )
                    _accumulate_flow_metrics(
                        state["sequence_sample_sums"][sequence], epe, angle, "sample"
                    )
                    state["sequence_valid_pixels"][sequence] += count
                    state["sequence_samples"][sequence] += 1
    protocols = {
        name: _flow_protocol_summary(state, groups)
        for name, state in protocol_states.items()
    }
    primary = protocols["event_supported"]
    output: dict[str, Any] = {
        **protocols,
        "all": primary["pixel_average"],
        "pixel_average": primary["pixel_average"],
        "sample_average": primary["sample_average"],
        "per_sequence": primary["per_sequence_pixel_average"],
    }
    if "sequence_pixel_average" in primary:
        output["sequence_average"] = primary["sequence_pixel_average"]
    for group in groups[1:]:
        output[group] = primary["density_groups"][group]["pixel_average"]
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
