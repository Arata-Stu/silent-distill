import numpy as np

from slassl.visualization import (
    event_support_from_voxel,
    flow_magnitude_scale,
    flow_to_rgb,
)


def test_flow_visualization_applies_mask() -> None:
    flow = np.zeros((2, 2, 3), dtype=np.float32)
    flow[0] = 1.0
    mask = np.array([[True, False, True], [False, True, False]])
    image = flow_to_rgb(flow, magnitude_scale=1.0, mask=mask)
    assert image.shape == (2, 3, 3)
    assert image.dtype == np.uint8
    assert np.all(image[~mask] == 0)
    assert np.all(image[mask].max(axis=1) == 255)


def test_event_support_and_shared_flow_scale() -> None:
    voxel = np.zeros((2, 5, 2, 3), dtype=np.float32)
    voxel[1, 2, 0, 1] = 1
    support = event_support_from_voxel(voxel)
    prediction = np.zeros((2, 2, 3), dtype=np.float32)
    target = np.zeros_like(prediction)
    target[0, 0, 1] = 4.0
    assert support.sum() == 1
    assert support[0, 1]
    assert flow_magnitude_scale(prediction, target, support, percentile=100.0) == 4.0
