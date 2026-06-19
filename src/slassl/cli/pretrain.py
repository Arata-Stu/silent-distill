from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from slassl.config import hydra_config_path
from slassl.models import EventAutoEncoderModel, SLASSLModel
from slassl.training import (
    append_metrics,
    build_dataset,
    build_loader,
    cosine_momentum,
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


@hydra.main(
    version_base="1.3",
    config_path=hydra_config_path("pretrain"),
    config_name="prophesee_1mp",
)
def main(config: DictConfig) -> None:
    rank, world_size, local_rank = init_distributed()
    seed_everything(int(config.seed), rank)
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(f"cuda:{local_rank}" if config.device == "cuda" else config.device)

    dataset = build_dataset(config, "ssl")
    loader, sampler = build_loader(dataset, config, rank, world_size, shuffle=True)
    polarity_channels = 2 if config.data.use_polarity else 1
    objective = str(config.get("objective", "sla_ssl")).lower()
    model_arguments = {
        "backbone": config.model.backbone,
        "input_channels": polarity_channels * int(config.data.bins),
        "small_stem": bool(config.model.small_stem),
        "model_config": config.model,
    }
    if objective == "sla_ssl":
        model = SLASSLModel(
            **model_arguments,
            polarity_channels=polarity_channels,
            temporal_bins=int(config.data.bins),
            projection_hidden_dim=int(config.model.projection_hidden_dim),
            projection_dim=int(config.model.projection_dim),
            loss_config=config.loss,
        ).to(device)
    elif objective == "autoencoder":
        model = EventAutoEncoderModel(**model_arguments).to(device)
    else:
        raise ValueError(f"Unknown pretraining objective: {objective}")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    accumulation_steps = int(config.training.get("gradient_accumulation_steps", 1))
    accumulation_groups = gradient_accumulation_groups(len(loader), accumulation_steps)
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
    resume = config.training.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
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
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(loader, disable=not is_main_process(), desc=f"pretrain {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            accumulated_metrics: dict[str, torch.Tensor] = {}
            accumulated_metric_counts: dict[str, int] = {}
            accumulated_samples = 0
            for batch_index, batch in enumerate(progress):
                group_index = batch_index // accumulation_steps
                group_start = group_index * accumulation_steps
                group_size = accumulation_groups[group_index]
                should_update = batch_index - group_start + 1 == group_size
                should_log_update = (
                    is_main_process()
                    and (global_step + 1) % int(config.training.log_every) == 0
                )
                short = batch["short"].to(device, non_blocking=True)
                long = (
                    batch["long"].to(device, non_blocking=True)
                    if objective == "sla_ssl"
                    else None
                )
                occupancy = (
                    batch["occupancy"].to(device, non_blocking=True)
                    if objective == "sla_ssl"
                    else None
                )
                synchronization = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not should_update
                    else nullcontext()
                )
                with synchronization:
                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        losses = model(short, long, occupancy)
                        scaled_loss = losses["loss"] / group_size
                    scaler.scale(scaled_loss).backward()

                if should_log_update:
                    for key, value in losses.items():
                        accumulated_metrics[key] = accumulated_metrics.get(
                            key, torch.zeros_like(value.detach(), dtype=torch.float32)
                        ) + value.detach().float()
                        accumulated_metric_counts[key] = (
                            accumulated_metric_counts.get(key, 0) + 1
                        )
                    density_key = "data/event_density"
                    density = batch["density"].detach().float().mean()
                    accumulated_metrics[density_key] = accumulated_metrics.get(
                        density_key, torch.zeros_like(density)
                    ) + density
                    accumulated_metric_counts[density_key] = (
                        accumulated_metric_counts.get(density_key, 0) + 1
                    )
                accumulated_samples += int(short.shape[0])
                if not should_update:
                    continue

                scaler.unscale_(optimizer)
                gradient_clip_norm = float(config.training.gradient_clip_norm)
                gradients = [
                    parameter.grad.detach()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if gradient_clip_norm > 0:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                elif gradients:
                    gradient_norm = torch.linalg.vector_norm(
                        torch.stack([gradient.norm(2) for gradient in gradients])
                    )
                else:
                    gradient_norm = torch.zeros((), device=device)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                momentum = cosine_momentum(
                    global_step,
                    total_steps,
                    float(config.model.ema_momentum),
                    float(config.model.ema_momentum_end),
                )
                unwrap_model(model).update_teacher(momentum)
                global_step += 1

                if is_main_process() and global_step % int(config.training.log_every) == 0:
                    averaged_metrics = {
                        key: float((value / accumulated_metric_counts[key]).item())
                        for key, value in accumulated_metrics.items()
                    }
                    metrics = {
                        "epoch": epoch,
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "ema_momentum": momentum,
                        "optimization/gradient_norm": float(gradient_norm.item()),
                        "optimization/amp_scale": float(scaler.get_scale()),
                        "optimization/micro_batches_per_update": group_size,
                        "optimization/effective_batch_size": accumulated_samples * world_size,
                        **averaged_metrics,
                    }
                    append_metrics(output_dir / "metrics.jsonl", metrics)
                    log_tensorboard_scalars(writer, "train", metrics, global_step)
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}")
                optimizer.zero_grad(set_to_none=True)
                accumulated_metrics.clear()
                accumulated_metric_counts.clear()
                accumulated_samples = 0

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
