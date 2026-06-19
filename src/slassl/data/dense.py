from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn.functional as F


# DSEC/M3ED 11-class protocol, cross-checked against Fast Feature Fields (Apache-2.0).
CITYSCAPES_19_TO_11 = np.array(
    [5, 6, 1, 9, 2, 4, 10, 10, 7, 7, 0, 3, 3, 8, 8, 8, 8, 8, 8],
    dtype=np.uint8,
)


def map_cityscapes_19_to_11(labels: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mapping = np.full(256, ignore_index, dtype=np.uint8)
    mapping[: len(CITYSCAPES_19_TO_11)] = CITYSCAPES_19_TO_11
    return mapping[np.asarray(labels, dtype=np.uint8)]


def read_dense_target(
    handle: h5py.File,
    target_format: str,
    index: int,
    camera: str = "left",
    segmentation_num_classes: int = 11,
    segmentation_ignore_index: int = 255,
) -> torch.Tensor:
    if target_format == "m3ed_flow":
        group = handle[f"flow/prophesee/{camera}"]
        target = np.stack((group["x"][index], group["y"][index]), axis=0)
        return torch.from_numpy(target.astype(np.float32, copy=False))
    if target_format == "mvsec_flow":
        target = np.asarray(handle[f"davis/{camera}/flow_dist"][index], dtype=np.float32)
        if target.ndim != 3:
            raise ValueError(f"Expected a 3D MVSEC flow target, received {target.shape}")
        if target.shape[0] != 2 and target.shape[-1] == 2:
            target = np.moveaxis(target, -1, 0)
        if target.shape[0] != 2:
            raise ValueError(f"Expected MVSEC flow channels first or last, received {target.shape}")
        return torch.from_numpy(np.ascontiguousarray(target))
    if target_format == "m3ed_segmentation":
        target = np.asarray(handle["predictions"][index]).squeeze()
        if target.ndim != 2:
            raise ValueError(f"Expected a 2D M3ED semantic target, received {target.shape}")
        if segmentation_num_classes == 11:
            target = map_cityscapes_19_to_11(target, segmentation_ignore_index)
        return torch.from_numpy(target.astype(np.int64, copy=False))
    raise ValueError(f"Unsupported dense target format: {target_format}")


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    source_height, source_width = flow.shape[-2:]
    if (source_height, source_width) == size:
        return flow
    resized = F.interpolate(
        flow.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)
    resized[0] *= size[1] / source_width
    resized[1] *= size[0] / source_height
    return resized


def resize_segmentation(labels: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(labels.shape[-2:]) == size:
        return labels
    return F.interpolate(labels[None, None].float(), size=size, mode="nearest")[0, 0].long()
