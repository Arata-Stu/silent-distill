from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from slassl.cli.finetune import build_model
from slassl.config import config_parser, load_config
from slassl.data.dsec import (
    DSEC_FLOW_TEST_SEQUENCES,
    read_dsec_flow_png,
    write_dsec_flow_png,
)
from slassl.training import build_dataset, build_loader
from slassl.utils import dump_json, seed_everything


def _expected_files(records: list[dict[str, Any]]) -> set[Path]:
    expected = {
        Path(str(record["output_sequence"])) / f"{int(record['output_index']):06d}.png"
        for record in records
    }
    if len(expected) != len(records):
        raise ValueError("DSEC submission manifest contains duplicate sequence/file indices")
    return expected


def _validate_submission(
    submission_dir: Path,
    expected: set[Path],
    height: int,
    width: int,
) -> None:
    actual = {
        path.relative_to(submission_dir)
        for path in submission_dir.rglob("*.png")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"DSEC submission file mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    for relative in tqdm(sorted(expected), desc="validate DSEC PNGs"):
        encoded = read_dsec_flow_png(submission_dir / relative)
        if encoded.shape != (height, width, 3):
            raise ValueError(
                f"{relative}: expected {(height, width, 3)}, received {encoded.shape}"
            )
        if not np.isin(encoded[..., 2], (0, 1)).all():
            raise ValueError(f"{relative}: third channel must contain only 0 or 1")


def _write_archive(submission_dir: Path, archive_file: Path, files: set[Path]) -> None:
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(files):
            archive.write(submission_dir / relative, arcname=relative.as_posix())


def main() -> None:
    args = config_parser("Export a DSEC optical-flow benchmark submission").parse_args()
    config = load_config(args.config, args.set)
    if str(config.task) != "flow":
        raise ValueError(f"DSEC flow export requires task=flow, received {config.task}")
    seed_everything(int(config.seed))
    if not torch.cuda.is_available() and config.device == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)

    dataset = build_dataset(
        config, "flow_inference", config.data.manifest, augmentation=False
    )
    loader, _ = build_loader(dataset, config, rank=0, world_size=1, shuffle=False)
    model = build_model(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    submission_dir = Path(config.submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    expected = _expected_files(dataset.records)
    submission_sequences = {path.parent.name for path in expected}
    if bool(config.get("require_official_sequences", True)) and (
        submission_sequences != DSEC_FLOW_TEST_SEQUENCES
    ):
        raise ValueError(
            "DSEC official submission requires exactly the seven flow test sequences; "
            f"missing={sorted(DSEC_FLOW_TEST_SEQUENCES - submission_sequences)}, "
            f"extra={sorted(submission_sequences - DSEC_FLOW_TEST_SEQUENCES)}"
        )
    clipped_values = 0
    elapsed_seconds = 0.0
    samples = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="export DSEC optical flow"):
            inputs = batch["short"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predictions = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - started
            samples += len(inputs)
            for prediction, sequence, file_index in zip(
                predictions,
                batch["output_sequence"],
                batch["output_index"],
                strict=True,
            ):
                output_path = submission_dir / str(sequence) / f"{int(file_index):06d}.png"
                clipped_values += write_dsec_flow_png(
                    output_path,
                    prediction.detach().cpu().numpy(),
                    valid_value=int(config.get("valid_channel_value", 1)),
                )

    _validate_submission(
        submission_dir,
        expected,
        height=int(config.data.height),
        width=int(config.data.width),
    )
    archive_file = Path(config.archive_file)
    _write_archive(submission_dir, archive_file, expected)
    report = {
        "dataset": "dsec",
        "task": "forward_optical_flow_submission",
        "checkpoint": str(config.checkpoint),
        "manifest": str(config.data.manifest),
        "submission_dir": str(submission_dir),
        "archive_file": str(archive_file),
        "sequences": len(submission_sequences),
        "samples": samples,
        "clipped_flow_values": clipped_values,
        "model_latency_ms_per_sample": 1000.0 * elapsed_seconds / max(samples, 1),
        "metrics": None,
        "metrics_note": "DSEC test ground truth is hidden; metrics are computed by the server.",
    }
    dump_json(report, config.report_file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
