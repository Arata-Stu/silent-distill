"""Generate M3ED ego-motion optical-flow targets from depth and camera poses.

Adapted and modified from Fast Feature Fields at revision 320eec6 under the
Apache License 2.0. See docs/THIRD_PARTY.md and
third_party/fast-feature-fields/LICENSE.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import logm
from tqdm import tqdm


F3_REVISION = "320eec6debf4491058d19a905b450f8de7fe4722"
F3_SOURCE = (
    "https://github.com/grasp-lyrl/fast-feature-fields/"
    "blob/320eec6/src/f3/tasks/optical_flow/generate_gt.py"
)
SMOOTHING_RADIUS = 10


class FlowProjector:
    def __init__(self, resolution: np.ndarray, intrinsics: np.ndarray) -> None:
        width, height = (int(value) for value in resolution)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid event-camera resolution: {resolution.tolist()}")
        if len(intrinsics) < 4:
            raise ValueError("M3ED event-camera intrinsics must contain fx, fy, cx, cy")
        fx, fy, cx, cy = (float(value) for value in intrinsics[:4])
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Invalid event-camera focal lengths: fx={fx}, fy={fy}")

        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        self.x_map = ((x_coordinates - cx) / fx).reshape(-1)
        self.y_map = ((y_coordinates - cy) / fy).reshape(-1)
        self.omega_matrix = np.zeros((height * width, 2, 3), dtype=np.float64)
        self.omega_matrix[:, 0, 0] = self.x_map * self.y_map
        self.omega_matrix[:, 1, 0] = 1.0 + np.square(self.y_map)
        self.omega_matrix[:, 0, 1] = -(1.0 + np.square(self.x_map))
        self.omega_matrix[:, 1, 1] = -(self.x_map * self.y_map)
        self.omega_matrix[:, 0, 2] = self.y_map
        self.omega_matrix[:, 1, 2] = -self.x_map

    def compute(
        self,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        depth: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if depth.shape != (self.height, self.width):
            raise ValueError(
                f"Depth shape {depth.shape} does not match event resolution "
                f"{(self.height, self.width)}"
            )
        flat_depth = np.asarray(depth).reshape(-1)
        # F3 checks finiteness only. Excluding zero/negative depth here prevents
        # infinities while preserving identical values at physically valid pixels.
        valid = np.isfinite(flat_depth) & (flat_depth > 0)
        x_flow = np.zeros(flat_depth.shape, dtype=np.float64)
        y_flow = np.zeros(flat_depth.shape, dtype=np.float64)
        if not valid.any() or dt == 0:
            return x_flow.reshape(depth.shape), y_flow.reshape(depth.shape)

        inverse_depth = 1.0 / flat_depth[valid]
        x_map = self.x_map[valid]
        y_map = self.y_map[valid]
        omega_matrix = self.omega_matrix[valid]
        x_flow[valid] = inverse_depth * (x_map * velocity[2] - velocity[0])
        x_flow[valid] += np.dot(omega_matrix[:, 0, :], angular_velocity)
        y_flow[valid] = inverse_depth * (y_map * velocity[2] - velocity[1])
        y_flow[valid] += np.dot(omega_matrix[:, 1, :], angular_velocity)
        x_flow[valid] *= dt * self.fx
        y_flow[valid] *= dt * self.fy
        return x_flow.reshape(depth.shape), y_flow.reshape(depth.shape)


def _camera_velocity(
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
    previous_time_s: float,
    current_time_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dt = current_time_s - previous_time_s
    if dt <= 0:
        raise ValueError(f"M3ED ground-truth timestamps must increase, received dt={dt}")
    relative_pose = previous_pose @ np.linalg.inv(current_pose)
    velocity = relative_pose[:3, 3] / dt
    angular_matrix = np.real_if_close(logm(relative_pose[:3, :3]) / dt)
    if np.iscomplexobj(angular_matrix):
        raise ValueError("Camera rotation logarithm has a non-negligible complex component")
    angular_velocity = np.array(
        [angular_matrix[2, 1], angular_matrix[0, 2], angular_matrix[1, 0]],
        dtype=np.float64,
    )
    return np.asarray(velocity, dtype=np.float64), angular_velocity, dt


def _default_output_path(events_path: Path) -> Path:
    suffix = "_data.h5"
    if not events_path.name.endswith(suffix):
        raise ValueError("--output is required when the event file does not end in _data.h5")
    return events_path.with_name(events_path.name.removesuffix(suffix) + "_gt_flow.h5")


def _require_dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    if path not in handle:
        raise ValueError(f"Missing HDF5 dataset '{path}' in {handle.filename}")
    value = handle[path]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"Expected HDF5 dataset '{path}' in {handle.filename}")
    return value


def generate_m3ed_flow(
    events_path: str | Path,
    depth_path: str | Path,
    output_path: str | Path | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    events_path = Path(events_path).resolve()
    depth_path = Path(depth_path).resolve()
    output_path = Path(output_path).resolve() if output_path is not None else _default_output_path(
        events_path
    )
    if not events_path.is_file():
        raise FileNotFoundError(f"M3ED event HDF5 file not found: {events_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"M3ED depth HDF5 file not found: {depth_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output_path}"
        )

    with h5py.File(events_path, "r") as events, h5py.File(depth_path, "r") as ground_truth:
        resolution = np.asarray(
            _require_dataset(events, "prophesee/left/calib/resolution")[:]
        )
        intrinsics = np.asarray(
            _require_dataset(events, "prophesee/left/calib/intrinsics")[:]
        )
        projector = FlowProjector(resolution, intrinsics)
        poses = _require_dataset(ground_truth, "Cn_T_C0")
        depths = _require_dataset(ground_truth, "depth/prophesee/left")
        timestamps = _require_dataset(ground_truth, "ts")
        timestamp_map = _require_dataset(ground_truth, "ts_map_prophesee_left")
        lengths = {len(value) for value in (poses, depths, timestamps, timestamp_map)}
        if len(lengths) != 1:
            raise ValueError(
                "M3ED depth, pose, timestamp, and timestamp-map datasets must have equal length"
            )
        total_frames = lengths.pop()
        stop_index = total_frames if stop_index is None else min(stop_index, total_frames)
        if start_index < 0 or stop_index <= start_index:
            raise ValueError(
                f"Require 0 <= start-index < stop-index <= {total_frames}, received "
                f"{start_index} and {stop_index}"
            )
        frame_count = stop_index - start_index
        depth_shape = tuple(int(value) for value in depths.shape[1:])
        if depth_shape != (projector.height, projector.width):
            raise ValueError(
                f"Depth shape {depth_shape} does not match event resolution "
                f"{(projector.height, projector.width)}"
            )

        velocities = np.zeros((frame_count, 3), dtype=np.float64)
        angular_velocities = np.zeros((frame_count, 3), dtype=np.float64)
        durations = np.zeros(frame_count, dtype=np.float64)
        selected_timestamps = np.asarray(
            timestamps[start_index:stop_index], dtype=np.uint64
        )
        selected_timestamp_map = np.asarray(
            timestamp_map[start_index:stop_index], dtype=np.uint64
        )
        previous_pose: np.ndarray | None = None
        previous_time_s: float | None = None
        frame_indices = range(start_index, stop_index)
        for offset, frame_index in enumerate(
            tqdm(frame_indices, desc="extract M3ED camera velocity", disable=not show_progress)
        ):
            current_pose = np.asarray(poses[frame_index], dtype=np.float64)
            current_time_s = float(timestamps[frame_index]) / 1_000_000.0
            if current_pose.shape != (4, 4):
                raise ValueError(f"Expected a 4x4 pose at index {frame_index}")
            if previous_pose is not None and previous_time_s is not None:
                velocity, angular_velocity, dt = _camera_velocity(
                    previous_pose, current_pose, previous_time_s, current_time_s
                )
                velocities[offset] = velocity
                angular_velocities[offset] = angular_velocity
                durations[offset] = dt
            previous_pose = current_pose
            previous_time_s = current_time_s

        # F3 aliases its smoothed arrays to the raw arrays, so each frame's
        # smoothed velocity contributes to subsequent moving averages.
        smoothed_velocities = velocities.copy()
        smoothed_angular_velocities = angular_velocities.copy()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent, delete=False
        )
        temporary_path = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            with h5py.File(temporary_path, "w") as output:
                output.attrs["generator"] = "sla-generate-m3ed-flow"
                output.attrs["source"] = F3_SOURCE
                output.attrs["source_revision"] = F3_REVISION
                output.attrs["start_index"] = start_index
                output.attrs["stop_index"] = stop_index
                output.create_dataset("ts", data=selected_timestamps)
                output.create_dataset(
                    "ts_map_prophesee_left", data=selected_timestamp_map
                )
                chunk_shape = (1, projector.height, projector.width)
                flow_shape = (frame_count, projector.height, projector.width)
                x_dataset = output.create_dataset(
                    "flow/prophesee/left/x",
                    shape=flow_shape,
                    dtype=np.float32,
                    chunks=chunk_shape,
                    compression="lzf",
                )
                y_dataset = output.create_dataset(
                    "flow/prophesee/left/y",
                    shape=flow_shape,
                    dtype=np.float32,
                    chunks=chunk_shape,
                    compression="lzf",
                )
                for offset, frame_index in enumerate(
                    tqdm(
                        range(start_index, stop_index),
                        desc="generate M3ED optical flow",
                        disable=not show_progress,
                    )
                ):
                    lower = max(0, offset - SMOOTHING_RADIUS)
                    upper = min(frame_count, offset + SMOOTHING_RADIUS + 1)
                    velocity = smoothed_velocities[lower:upper].mean(axis=0)
                    angular_velocity = smoothed_angular_velocities[lower:upper].mean(
                        axis=0
                    )
                    smoothed_velocities[offset] = velocity
                    smoothed_angular_velocities[offset] = angular_velocity
                    x_flow, y_flow = projector.compute(
                        velocity,
                        angular_velocity,
                        np.asarray(depths[frame_index]),
                        durations[offset],
                    )
                    x_dataset[offset] = x_flow.astype(np.float32, copy=False)
                    y_dataset[offset] = y_flow.astype(np.float32, copy=False)
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return output_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate F3-compatible M3ED ego-motion optical-flow ground truth"
    )
    result.add_argument("--events-h5", required=True, type=Path)
    result.add_argument("--depth-h5", required=True, type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--start-index", "--start-ind", dest="start_index", type=int, default=0)
    result.add_argument("--stop-index", "--stop-ind", dest="stop_index", type=int)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    output_path = generate_m3ed_flow(
        args.events_h5,
        args.depth_h5,
        args.output,
        args.start_index,
        args.stop_index,
        args.overwrite,
    )
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
