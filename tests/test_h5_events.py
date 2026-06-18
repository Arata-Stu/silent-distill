import h5py
import numpy as np
import pytest

from slassl.data.h5_events import H5EventReader


def test_missing_file_reports_the_path(tmp_path) -> None:
    path = tmp_path / "missing.h5"

    with pytest.raises(FileNotFoundError, match="Event HDF5 file not found:.*missing.h5"):
        H5EventReader(path)


def test_reads_dsec_split_layout_with_time_offset(tmp_path) -> None:
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        events.create_dataset("x", data=np.array([1, 2], dtype=np.uint16))
        events.create_dataset("y", data=np.array([3, 4], dtype=np.uint16))
        events.create_dataset("t", data=np.array([0, 1000], dtype=np.int64))
        events.create_dataset("p", data=np.array([0, 1], dtype=np.int8))
        handle.create_dataset("ms_to_idx", data=np.array([0, 1], dtype=np.int64))
        handle.create_dataset("t_offset", data=np.array(50_000, dtype=np.int64))

    with H5EventReader(path, event_group="/events") as reader:
        assert reader.bounds_us() == (50_000, 51_000)
        x, _, timestamps, _ = reader.read_us(50_500, 51_000)
        assert x.tolist() == [2]
        assert timestamps.tolist() == [51_000]


def test_reads_m3ed_split_layout(tmp_path) -> None:
    path = tmp_path / "m3ed.h5"
    with h5py.File(path, "w") as handle:
        events = handle.create_group("prophesee/left")
        for key, values in {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "t": [0, 1000, 2000],
            "p": [0, 1, 0],
        }.items():
            events.create_dataset(key, data=np.asarray(values))
        events.create_dataset("ms_map_idx", data=np.array([0, 1, 2]))

    with H5EventReader(path, camera="left") as reader:
        assert reader.layout.path == "/prophesee/left"
        assert reader.layout.timestamp_scale_to_us == 1.0
        assert reader.bounds_us() == (0, 2000)


def test_reads_mvsec_matrix_and_converts_seconds(tmp_path) -> None:
    path = tmp_path / "outdoor_day_data.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "davis/left/events",
            data=np.array([[1, 2, 0.001, 0], [3, 4, 0.002, 1]], dtype=np.float64),
        )

    with H5EventReader(path, camera="left") as reader:
        assert reader.layout.storage == "matrix"
        assert reader.bounds_us() == (1000, 2000)
        x, y, timestamps, polarity = reader.read_us(0, 1500)
        assert x.tolist() == [1.0]
        assert y.tolist() == [2.0]
        assert timestamps.tolist() == [1000]
        assert polarity.tolist() == [0.0]


def test_reads_compound_event_dataset(tmp_path) -> None:
    path = tmp_path / "compound.h5"
    dtype = np.dtype([("x", "u2"), ("y", "u2"), ("t", "i8"), ("p", "i1")])
    values = np.array([(1, 2, 100, 1), (2, 3, 200, 0)], dtype=dtype)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("events", data=values)

    with H5EventReader(path) as reader:
        assert reader.layout.storage == "compound"
        assert reader.bounds_us() == (100, 200)
