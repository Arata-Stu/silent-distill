import json

import h5py
import numpy as np
import pytest
import torch

from slassl.data.dataset import EventWindowDataset
from slassl.data.dense import map_cityscapes_19_to_11, read_dense_target
from slassl.evaluation import evaluate_flow
from slassl.models.dense import DensePredictionModel
from slassl.cli.index_dense import (
    _align_mvsec_timestamps,
    _target_event_interval,
    _temporal_split_bounds,
)


def test_reads_m3ed_flow_and_semantic_targets(tmp_path) -> None:
    path = tmp_path / "targets.h5"
    with h5py.File(path, "w") as handle:
        flow = handle.create_group("flow/prophesee/left")
        flow.create_dataset("x", data=np.ones((1, 3, 4), dtype=np.float32))
        flow.create_dataset("y", data=np.full((1, 3, 4), 2, dtype=np.float32))
        handle.create_dataset("predictions", data=np.array([[[0, 1], [10, 18]]]))

    with h5py.File(path, "r") as handle:
        target = read_dense_target(handle, "m3ed_flow", 0)
        labels = read_dense_target(handle, "m3ed_segmentation", 0)
    assert target.shape == (2, 3, 4)
    assert target[:, 0, 0].tolist() == [1.0, 2.0]
    assert labels.tolist() == map_cityscapes_19_to_11(
        np.array([[0, 1], [10, 18]])
    ).tolist()


def test_event_dataset_returns_aligned_flow_target(tmp_path) -> None:
    events_path = tmp_path / "events.h5"
    with h5py.File(events_path, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.array([1, 2, 3]))
        events.create_dataset("y", data=np.array([1, 2, 3]))
        events.create_dataset("t", data=np.array([5_000, 7_000, 10_000]))
        events.create_dataset("p", data=np.array([0, 1, 1]))

    flow_path = tmp_path / "flow.h5"
    with h5py.File(flow_path, "w") as handle:
        flow = handle.create_group("flow/prophesee/left")
        flow.create_dataset("x", data=np.ones((1, 8, 8), dtype=np.float32))
        flow.create_dataset("y", data=np.ones((1, 8, 8), dtype=np.float32))

    manifest = tmp_path / "flow.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": events_path.name,
                "timestamp_us": 10_000,
                "target": flow_path.name,
                "target_index": 0,
                "target_format": "m3ed_flow",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = EventWindowDataset(
        manifest,
        tmp_path,
        height=8,
        width=8,
        bins=2,
        short_window_us=5_000,
        long_window_us=10_000,
        task="flow",
    )
    sample = dataset[0]
    assert sample["target"].shape == (2, 8, 8)
    assert sample["target_valid_mask"].any()
    assert sample["valid_mask"].any()


def test_dense_prediction_model_restores_input_resolution() -> None:
    config = {
        "vit_patch_size": 4,
        "vit_embed_dim": 32,
        "vit_depth": 4,
        "vit_num_heads": 4,
        "decoder_channels": 16,
        "recurrent": "none",
    }
    model = DensePredictionModel("vit_tiny", 4, 2, False, config)
    output = model(torch.randn(2, 2, 2, 16, 20))
    assert output.shape == (2, 2, 16, 20)


def test_aligns_mvsec_absolute_and_relative_timestamps() -> None:
    raw = np.array([1_600_000_001.0, 1_600_000_002.0])
    relative, relative_mode = _align_mvsec_timestamps(
        raw, 1_600_000_000.0, (0, 3_000_000)
    )
    absolute, absolute_mode = _align_mvsec_timestamps(
        raw, 1_600_000_000.0, (1_600_000_000_000_000, 1_600_000_003_000_000)
    )
    assert relative.tolist() == [1_000_000, 2_000_000]
    assert relative_mode == "relative_seconds"
    assert absolute_mode == "absolute_seconds"


def test_aligns_raw_mvsec_without_absolute_start_time_attribute() -> None:
    raw = np.array([1_600_000_001.0, 1_600_000_002.0])
    timestamps, mode = _align_mvsec_timestamps(
        raw, None, (1_600_000_000_000_000, 1_600_000_003_000_000)
    )
    assert timestamps.tolist() == [1_600_000_001_000_000, 1_600_000_002_000_000]
    assert mode == "absolute_seconds"


def test_temporal_split_leaves_a_gap_between_train_and_validation() -> None:
    train = _temporal_split_bounds(0, 1_000_000, 0.0, 0.8, 50_000)
    validation = _temporal_split_bounds(0, 1_000_000, 0.8, 1.0, 50_000)
    assert train == (0, 750_000)
    assert validation == (850_000, 1_000_000)


def test_m3ed_flow_uses_the_interval_ending_at_its_timestamp() -> None:
    timestamps = np.array([10_000, 60_000, 110_000])
    assert _target_event_interval("m3ed", "flow", timestamps, 0, 10_000) is None
    assert _target_event_interval("m3ed", "flow", timestamps, 1, 10_000) == (
        10_000,
        60_000,
    )


def test_mvsec_flow_uses_the_interval_starting_at_its_timestamp() -> None:
    timestamps = np.array([10_000, 60_000, 110_000])
    assert _target_event_interval("mvsec", "flow", timestamps, 1, 10_000) == (
        60_000,
        110_000,
    )
    assert _target_event_interval("mvsec", "flow", timestamps, 2, 10_000) is None


def test_flow_evaluation_reports_sample_and_pixel_averages() -> None:
    class ZeroFlow(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.shape[0], 2, *inputs.shape[-2:])

    targets = torch.zeros(2, 2, 1, 2)
    targets[0, 0, 0] = torch.tensor([1.0, 3.0])
    targets[1, 0, 0] = torch.tensor([10.0, 100.0])
    valid = torch.tensor([[[True, True]], [[True, False]]])
    loader = [
        {
            "short": torch.zeros(2, 2, 1, 1, 2),
            "target": targets,
            "valid_mask": valid,
            "target_valid_mask": valid,
            "density": torch.zeros(2),
            "metric_sequence": ["sequence", "sequence"],
        }
    ]
    metrics = evaluate_flow(ZeroFlow(), loader, torch.device("cpu"), 0.001, 0.01)
    assert metrics["event_supported"]["sample_average"]["aepe"] == pytest.approx(6.0)
    assert metrics["event_supported"]["pixel_average"]["aepe"] == pytest.approx(14 / 3)
    assert metrics["event_supported"]["sequence_average"]["aepe"] == pytest.approx(6.0)
