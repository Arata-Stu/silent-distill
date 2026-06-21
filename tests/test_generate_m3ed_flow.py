import h5py
import numpy as np

from slassl.cli.generate_m3ed_flow import F3_REVISION, generate_m3ed_flow


def test_generates_f3_compatible_m3ed_flow_h5(tmp_path) -> None:
    events_path = tmp_path / "sequence_data.h5"
    with h5py.File(events_path, "w") as events:
        calibration = events.create_group("prophesee/left/calib")
        calibration.create_dataset("resolution", data=np.array([4, 3]))
        calibration.create_dataset("intrinsics", data=np.array([2.0, 2.0, 1.5, 1.0]))

    depth_path = tmp_path / "sequence_depth.h5"
    with h5py.File(depth_path, "w") as ground_truth:
        ground_truth.create_dataset("Cn_T_C0", data=np.repeat(np.eye(4)[None], 3, axis=0))
        ground_truth.create_dataset(
            "depth/prophesee/left", data=np.ones((3, 3, 4), dtype=np.float32)
        )
        ground_truth.create_dataset("ts", data=np.array([100_000, 200_000, 300_000]))
        ground_truth.create_dataset("ts_map_prophesee_left", data=np.array([10, 20, 30]))

    output_path = generate_m3ed_flow(
        events_path, depth_path, show_progress=False
    )

    assert output_path == tmp_path / "sequence_gt_flow.h5"
    with h5py.File(output_path, "r") as output:
        assert output["ts"][:].tolist() == [100_000, 200_000, 300_000]
        assert output["ts_map_prophesee_left"][:].tolist() == [10, 20, 30]
        assert output["flow/prophesee/left/x"].shape == (3, 3, 4)
        assert output["flow/prophesee/left/y"].shape == (3, 3, 4)
        assert np.count_nonzero(output["flow/prophesee/left/x"][:]) == 0
        assert np.count_nonzero(output["flow/prophesee/left/y"][:]) == 0
        assert output.attrs["source_revision"] == F3_REVISION


def test_generates_selected_frame_range(tmp_path) -> None:
    events_path = tmp_path / "selected_data.h5"
    with h5py.File(events_path, "w") as events:
        calibration = events.create_group("prophesee/left/calib")
        calibration.create_dataset("resolution", data=np.array([2, 2]))
        calibration.create_dataset("intrinsics", data=np.array([1.0, 1.0, 0.5, 0.5]))

    depth_path = tmp_path / "selected_depth.h5"
    with h5py.File(depth_path, "w") as ground_truth:
        ground_truth.create_dataset("Cn_T_C0", data=np.repeat(np.eye(4)[None], 4, axis=0))
        ground_truth.create_dataset(
            "depth/prophesee/left", data=np.ones((4, 2, 2), dtype=np.float32)
        )
        ground_truth.create_dataset("ts", data=np.array([1, 2, 3, 4]) * 100_000)
        ground_truth.create_dataset("ts_map_prophesee_left", data=np.arange(4))

    output_path = generate_m3ed_flow(
        events_path,
        depth_path,
        output_path=tmp_path / "range.h5",
        start_index=1,
        stop_index=3,
        show_progress=False,
    )

    with h5py.File(output_path, "r") as output:
        assert output["ts"][:].tolist() == [200_000, 300_000]
        assert output["flow/prophesee/left/x"].shape == (2, 2, 2)
