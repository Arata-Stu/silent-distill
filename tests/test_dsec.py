import json

import h5py
import numpy as np

from slassl.data.dataset import EventWindowDataset
from slassl.data.dsec import (
    DSEC_FLOW_MAX,
    DSEC_FLOW_MIN,
    decode_dsec_flow,
    encode_dsec_flow,
    read_dsec_flow_png,
    write_dsec_flow_png,
)


def test_dsec_flow_encoding_rounds_to_official_precision() -> None:
    flow = np.array(
        [[[0.0, 1.25], [DSEC_FLOW_MIN, DSEC_FLOW_MAX]], [[-2.5, 3.0], [4.0, 5.0]]],
        dtype=np.float32,
    )
    encoded, clipped = encode_dsec_flow(flow)
    decoded, valid = decode_dsec_flow(encoded)
    assert clipped == 0
    assert encoded.dtype == np.uint16
    assert valid.all()
    np.testing.assert_allclose(decoded, flow, atol=1 / 128)


def test_dsec_flow_png_is_16_bit_rgb(tmp_path) -> None:
    flow = np.zeros((2, 3, 4), dtype=np.float32)
    path = tmp_path / "000001.png"
    assert write_dsec_flow_png(path, flow) == 0
    encoded = read_dsec_flow_png(path)
    assert encoded.shape == (3, 4, 3)
    assert encoded.dtype == np.uint16
    assert np.all(encoded[..., :2] == 2**15)
    assert np.all(encoded[..., 2] == 1)


def test_dsec_dataset_uses_gt_valid_mask_without_event_support(tmp_path) -> None:
    event_path = tmp_path / "events.h5"
    with h5py.File(event_path, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.array([1], dtype=np.int16))
        events.create_dataset("y", data=np.array([1], dtype=np.int16))
        events.create_dataset("t", data=np.array([500], dtype=np.int64))
        events.create_dataset("p", data=np.array([1], dtype=np.int8))

    rectify_path = tmp_path / "rectify_map.h5"
    x, y = np.meshgrid(np.arange(4), np.arange(3))
    with h5py.File(rectify_path, "w") as handle:
        handle.create_dataset("rectify_map", data=np.stack((x, y), axis=-1))

    flow_path = tmp_path / "000000.png"
    write_dsec_flow_png(flow_path, np.zeros((2, 3, 4), dtype=np.float32))
    manifest = tmp_path / "flow.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence": event_path.name,
                "rectify_map": rectify_path.name,
                "window_start_us": 0,
                "timestamp_us": 1000,
                "end_inclusive": False,
                "target": flow_path.name,
                "target_format": "dsec_flow",
                "valid_mask_mode": "target",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = EventWindowDataset(
        manifest,
        tmp_path,
        height=3,
        width=4,
        bins=1,
        short_window_us=1000,
        long_window_us=1000,
        task="flow",
    )
    sample = dataset[0]
    assert sample["short"].sum() == 1
    assert sample["target"].abs().sum() == 0
    assert sample["valid_mask"].all()
