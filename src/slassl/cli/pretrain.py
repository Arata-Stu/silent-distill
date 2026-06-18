from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from slassl.config import config_parser, load_config
from slassl.models import SLASSLModel
from slassl.training import (
    append_metrics,
    build_dataset,
    build_loader,
    cosine_momentum,
    cosine_warmup_scheduler,
)
from slassl.utils import (
    atomic_torch_save,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    seed_everything,
    unwrap_model,
)


def main() -> None:
    args = config_parser("Pretrain SLA-SSL").parse_args()
    config = load_config(args.config, args.set)
    rank, world_size, local_rank = init_distributed()
    seed_everything(int(config.seed), rank)
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(f"cuda:{local_rank}" if config.device == "cuda" else config.device)

    dataset = build_dataset(config, "ssl")
    loader, sampler = build_loader(dataset, config, rank, world_size, shuffle=True)
    polarity_channels = 2 if config.data.use_polarity else 1
    model = SLASSLModel(
        backbone=config.model.backbone,
        input_channels=polarity_channels * int(config.data.bins),
        polarity_channels=polarity_channels,
        temporal_bins=int(config.data.bins),
        projection_hidden_dim=int(config.model.projection_hidden_dim),
        projection_dim=int(config.model.projection_dim),
        small_stem=bool(config.model.small_stem),
        loss_config=config.loss,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    total_steps = int(config.training.epochs) * len(loader)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps,
        int(config.training.warmup_epochs) * len(loader),
        float(config.training.minimum_lr) / float(config.training.learning_rate),
    )
    amp_enabled = bool(config.training.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_dir = Path(config.output_dir)
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
            json.dump(config, handle, indent=2)

    try:
        for epoch in range(start_epoch, int(config.training.epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(loader, disable=not is_main_process(), desc=f"pretrain {epoch + 1}")
            for batch in progress:
                short = batch["short"].to(device, non_blocking=True)
                long = batch["long"].to(device, non_blocking=True)
                occupancy = batch["occupancy"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    losses = model(short, long, occupancy)
                scaler.scale(losses["loss"]).backward()
                if float(config.training.gradient_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
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
                    metrics = {
                        "epoch": epoch,
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "ema_momentum": momentum,
                        **{key: float(value.item()) for key, value in losses.items()},
                    }
                    append_metrics(output_dir / "metrics.jsonl", metrics)
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}")

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
                    "config": dict(config),
                }
                atomic_torch_save(state, output_dir / f"checkpoint_{epoch + 1:04d}.pt")
                atomic_torch_save(state, output_dir / "checkpoint_last.pt")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
