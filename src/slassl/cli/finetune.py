from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from slassl.config import hydra_config_path
from slassl.evaluation import (
    evaluate_classification,
    evaluate_detection,
    evaluate_flow,
    evaluate_segmentation,
)
from slassl.models import ClassificationModel, DensePredictionModel, build_detector
from slassl.models.downstream import load_pretrained_encoder
from slassl.training import (
    append_metrics,
    build_dataset,
    build_loader,
    cosine_warmup_scheduler,
    create_summary_writer,
    gradient_accumulation_groups,
    log_tensorboard_scalars,
    resolved_config,
)
from slassl.utils import (
    atomic_torch_save,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    seed_everything,
    unwrap_model,
)


def build_model(config: Any) -> torch.nn.Module:
    channels = (2 if config.data.use_polarity else 1) * int(config.data.bins)
    if config.task == "classification":
        return ClassificationModel(
            config.model.backbone,
            channels,
            int(config.model.num_classes),
            bool(config.model.small_stem),
            float(config.model.get("dropout", 0.0)),
            config.model,
        )
    if config.task == "detection":
        if str(config.model.get("recurrent", "none")) != "none":
            raise ValueError("Recurrent fine-tuning is currently supported for classification only")
        return build_detector(
            config.model.backbone,
            channels,
            int(config.model.num_classes),
            int(config.model.min_size),
            int(config.model.max_size),
        )
    if config.task in {"flow", "segmentation"}:
        output_channels = 2 if config.task == "flow" else int(config.model.num_classes)
        return DensePredictionModel(
            config.model.backbone,
            channels,
            output_channels,
            bool(config.model.small_stem),
            config.model,
        )
    raise ValueError(f"Unknown downstream task: {config.task}")


def load_ssl_weights(model: torch.nn.Module, config: Any) -> None:
    checkpoint = config.training.get("pretrained_checkpoint")
    if not checkpoint:
        return
    prefix = "backbone.body." if config.task == "detection" else "encoder."
    missing, unexpected = load_pretrained_encoder(model, checkpoint, prefix)
    allowed_missing = (
        "head.",
        "decoder.",
        "backbone.fpn.",
        "head.classification_head.",
    )
    important_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if important_missing or unexpected:
        print(
            json.dumps(
                {"pretrained_missing": important_missing, "pretrained_unexpected": unexpected},
                indent=2,
            )
        )


def _set_backbone_trainable(model: torch.nn.Module, task: str, trainable: bool) -> None:
    backbone = model.backbone if task == "detection" else model.encoder
    for parameter in backbone.parameters():
        parameter.requires_grad_(trainable)


def _validation(model: torch.nn.Module, loader: Any, config: Any, device: torch.device) -> dict:
    low_max = float(config.evaluation.low_density_max)
    high_min = float(config.evaluation.high_density_min)
    if config.task == "classification":
        return evaluate_classification(model, loader, device, low_max, high_min)
    if config.task == "detection":
        return evaluate_detection(model, loader, device, low_max, high_min)
    if config.task == "flow":
        return evaluate_flow(model, loader, device, low_max, high_min)
    return evaluate_segmentation(
        model,
        loader,
        device,
        int(config.model.num_classes),
        int(config.data.get("segmentation_ignore_index", 255)),
        low_max,
        high_min,
    )


@hydra.main(
    version_base="1.3",
    config_path=hydra_config_path("finetune"),
    config_name="prophesee_1mp_detection",
)
def main(config: DictConfig) -> None:
    rank, world_size, local_rank = init_distributed()
    if world_size > 1 and int(config.training.get("freeze_backbone_epochs", 0)) > 0:
        raise ValueError("freeze_backbone_epochs > 0 is supported only for single-GPU fine-tuning")
    seed_everything(int(config.seed), rank)
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(f"cuda:{local_rank}" if config.device == "cuda" else config.device)

    train_dataset = build_dataset(config, config.task, config.data.train_manifest)
    train_loader, train_sampler = build_loader(
        train_dataset, config, rank, world_size, shuffle=True
    )
    validation_loader = None
    if config.data.get("validation_manifest") and world_size == 1:
        validation_dataset = build_dataset(
            config, config.task, config.data.validation_manifest, augmentation=False
        )
        validation_loader, _ = build_loader(
            validation_dataset, config, rank=0, world_size=1, shuffle=False
        )

    model = build_model(config)
    load_ssl_weights(model, config)
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    accumulation_steps = int(config.training.get("gradient_accumulation_steps", 1))
    accumulation_groups = gradient_accumulation_groups(len(train_loader), accumulation_steps)
    if not accumulation_groups:
        raise ValueError("Training loader is empty; reduce training.batch_size or add samples")
    updates_per_epoch = len(accumulation_groups)
    total_steps = int(config.training.epochs) * updates_per_epoch
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps,
        int(config.training.warmup_epochs) * updates_per_epoch,
        float(config.training.minimum_lr) / float(config.training.learning_rate),
    )
    amp_enabled = bool(config.training.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_dir = Path(config.output_dir)
    writer = create_summary_writer(config, output_dir) if is_main_process() else None
    start_epoch, global_step = 0, 0
    if config.training.get("resume"):
        checkpoint = torch.load(config.training.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(resolved_config(config), handle, indent=2)
        if writer is not None:
            writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(config, resolve=True)}\n```")

    try:
        for epoch in range(start_epoch, int(config.training.epochs)):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            trainable = epoch >= int(config.training.get("freeze_backbone_epochs", 0))
            _set_backbone_trainable(unwrap_model(model), config.task, trainable)
            model.train()
            progress = tqdm(
                train_loader, disable=not is_main_process(), desc=f"finetune {epoch + 1}"
            )
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), device=device)
            for batch_index, batch in enumerate(progress):
                group_index = batch_index // accumulation_steps
                group_start = group_index * accumulation_steps
                group_size = accumulation_groups[group_index]
                should_update = batch_index - group_start + 1 == group_size
                synchronization = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not should_update
                    else nullcontext()
                )
                with synchronization:
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        if config.task == "classification":
                            inputs = batch["short"].to(device, non_blocking=True)
                            labels = batch["label"].to(device, non_blocking=True)
                            loss = F.cross_entropy(model(inputs), labels)
                        elif config.task == "detection":
                            short = batch["short"]
                            if short.ndim == 6:
                                short = short[:, -1]
                            images = [voxel.flatten(0, 1).to(device) for voxel in short]
                            targets = [
                                {key: value.to(device) for key, value in target.items()}
                                for target in batch["target"]
                            ]
                            loss_dict = model(images, targets)
                            loss = sum(loss_dict.values())
                        else:
                            inputs = batch["short"].to(device, non_blocking=True)
                            targets = batch["target"].to(device, non_blocking=True)
                            predictions = model(inputs)
                            if config.task == "flow":
                                valid = batch["valid_mask"].to(device, non_blocking=True)
                                error = (
                                    (predictions - targets).square().sum(dim=1).add(1e-6).sqrt()
                                )
                                loss = error[valid].mean() if valid.any() else error.sum() * 0
                            else:
                                loss = F.cross_entropy(
                                    predictions,
                                    targets.long(),
                                    ignore_index=int(
                                        config.data.get("segmentation_ignore_index", 255)
                                    ),
                                )
                        scaled_loss = loss / group_size
                    scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.detach().float()
                if not should_update:
                    continue
                if float(config.training.gradient_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                if is_main_process() and global_step % int(config.training.log_every) == 0:
                    metrics = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float((accumulated_loss / group_size).item()),
                        "lr": scheduler.get_last_lr()[0],
                        "optimization/micro_batches_per_update": group_size,
                        "optimization/effective_batch_size": (
                            int(config.training.batch_size) * group_size * world_size
                        ),
                    }
                    append_metrics(output_dir / "metrics.jsonl", metrics)
                    log_tensorboard_scalars(writer, "train", metrics, global_step)
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}")
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss.zero_()

            if validation_loader is not None and (
                (epoch + 1) % int(config.evaluation.every_epochs) == 0
            ):
                metrics = _validation(unwrap_model(model), validation_loader, config, device)
                append_metrics(
                    output_dir / "validation_metrics.jsonl", {"epoch": epoch, **metrics}
                )
                log_tensorboard_scalars(writer, "validation", metrics, epoch + 1)
            if is_main_process() and (
                (epoch + 1) % int(config.training.save_every_epochs) == 0
                or epoch + 1 == int(config.training.epochs)
            ):
                state = {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "task": config.task,
                    "config": resolved_config(config),
                }
                atomic_torch_save(state, output_dir / f"checkpoint_{epoch + 1:04d}.pt")
                atomic_torch_save(state, output_dir / "checkpoint_last.pt")
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
