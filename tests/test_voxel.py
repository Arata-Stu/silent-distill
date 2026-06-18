import numpy as np

from slassl.data.voxel import EventVoxelizer


def test_voxel_preserves_polarity_and_causal_time_bins() -> None:
    voxelizer = EventVoxelizer(height=4, width=5, bins=2, use_polarity=True)
    voxel = voxelizer(
        x=np.array([1, 1, 2]),
        y=np.array([2, 2, 3]),
        t=np.array([1000, 1499, 2001]),
        p=np.array([0, 1, 1]),
        start_us=1000,
        end_us=2000,
    )
    assert voxel.shape == (2, 2, 4, 5)
    assert voxel[0, 0, 2, 1] == 1
    assert voxel[1, 0, 2, 1] == 1
    assert voxel.sum() == 2


def test_occupancy_uses_spatial_stride() -> None:
    voxelizer = EventVoxelizer(height=5, width=7, bins=1, use_polarity=False)
    target = voxelizer.occupancy(
        np.array([6]),
        np.array([4]),
        np.array([10]),
        np.array([1]),
        0,
        10,
        spatial_stride=4,
    )
    assert target.shape == (1, 1, 2, 2)
    assert target[0, 0, 1, 1] == 1

