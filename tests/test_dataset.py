import json

import h5py
import numpy as np

from slassl.data.dataset import EventWindowDataset


def test_dataset_never_reads_future_events(tmp_path) -> None:
    sequence = tmp_path / "sequence.h5"
    with h5py.File(sequence, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.array([0, 1, 2, 3]))
        events.create_dataset("y", data=np.array([0, 1, 2, 3]))
        events.create_dataset("t", data=np.array([1000, 1500, 2000, 2500]))
        events.create_dataset("p", data=np.array([0, 1, 0, 1]))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sequence": "sequence.h5", "timestamp_us": 2000}) + "\n",
        encoding="utf-8",
    )
    dataset = EventWindowDataset(
        manifest=manifest,
        data_root=tmp_path,
        height=4,
        width=4,
        bins=2,
        short_window_us=500,
        long_window_us=1000,
        task="ssl",
        occupancy_stride=2,
    )
    sample = dataset[0]
    assert sample["short"].sum() == 2
    assert sample["long"].sum() == 3
    assert sample["occupancy"].sum() == 2


def test_downstream_sample_does_not_materialize_ssl_targets(tmp_path) -> None:
    sequence = tmp_path / "sequence.h5"
    with h5py.File(sequence, "w") as handle:
        events = handle.create_group("events")
        for key, values in {
            "x": [0],
            "y": [0],
            "t": [1000],
            "p": [1],
        }.items():
            events.create_dataset(key, data=np.array(values))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sequence": "sequence.h5", "timestamp_us": 1000, "label": 0}) + "\n",
        encoding="utf-8",
    )
    dataset = EventWindowDataset(
        manifest, tmp_path, 2, 2, 1, 1000, 5000, task="classification"
    )
    sample = dataset[0]
    assert "long" not in sample
    assert "occupancy" not in sample


def test_detection_targets_are_read_directly_from_npy_slice(tmp_path) -> None:
    sequence = tmp_path / "sequence.h5"
    with h5py.File(sequence, "w") as handle:
        events = handle.create_group("events")
        for key, values in {"x": [0], "y": [0], "t": [1000], "p": [1]}.items():
            events.create_dataset(key, data=np.array(values))
    dtype = np.dtype(
        [("t", "i8"), ("x", "f4"), ("y", "f4"), ("w", "f4"), ("h", "f4"), ("class_id", "i4")]
    )
    np.save(
        tmp_path / "sequence_bbox.npy",
        np.array([(1000, 1, 2, 10, 20, 0), (1000, 2, 3, 5, 5, 6)], dtype=dtype),
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": "sequence.h5",
                "timestamp_us": 1000,
                "annotations": "sequence_bbox.npy",
                "annotation_start": 0,
                "annotation_end": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = EventWindowDataset(
        manifest,
        tmp_path,
        32,
        32,
        1,
        1000,
        5000,
        task="detection",
        detection_class_ids=[0, 1, 2],
    )
    target = dataset[0]["target"]
    assert target["boxes"].tolist() == [[1.0, 2.0, 11.0, 22.0]]
    assert target["labels"].tolist() == [1]
