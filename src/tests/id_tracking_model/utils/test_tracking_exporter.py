"""
Tests for TrackingDataExporter, TrackingDataCollector, and NumpyEncoder
"""

import json
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pytest
import torch

from sailsprep.id_tracking_model.utils.tracking_exporter import (
    NumpyEncoder,
    TrackingDataCollector,
    TrackingDataExporter,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_detection(
    bbox: Any = None,
    keypoints: Any = None,
    confidence: float = 0.9,
) -> Dict[str, Any]:
    """Build a minimal detection dict."""
    if bbox is None:
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    if keypoints is None:
        # 17 keypoints, each [x, y, conf]
        keypoints = np.random.rand(17, 3).astype(np.float32)
    return {"bbox": bbox, "keypoints": keypoints, "confidence": confidence}


def make_profile(
    face: bool = True, upper: bool = True, lower: bool = True, creation_frame: int = 0
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"creation_frame": creation_frame}
    if face:
        profile["face_feature"] = np.random.rand(512).astype(np.float32)
    if upper:
        profile["upper_feature"] = np.random.rand(512).astype(np.float32)
    if lower:
        profile["lower_feature"] = np.random.rand(512).astype(np.float32)
    return profile


@pytest.fixture()
def exporter() -> TrackingDataExporter:
    return TrackingDataExporter()


@pytest.fixture()
def populated_exporter() -> TrackingDataExporter:
    """Exporter with two tracks across three frames."""
    exp = TrackingDataExporter()
    exp.set_video_metadata("video.mp4", 100, 30.0, 1920, 1080)
    detections = [make_detection(), make_detection()]
    exp.add_frame_data(0, detections, {0: 1, 1: 2})
    exp.add_frame_data(1, detections, {0: 1, 1: 2})
    exp.add_frame_data(2, [detections[0]], {0: 1})
    return exp


# ---------------------------------------------------------------------------
# NumpyEncoder
# ---------------------------------------------------------------------------

class TestNumpyEncoder:
    def test_encodes_ndarray(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [1.0, 2.0, 3.0]

    def test_encodes_torch_tensor(self) -> None:
        t = torch.tensor([4.0, 5.0])
        result = json.loads(json.dumps(t, cls=NumpyEncoder))
        assert result == [4.0, 5.0]

    def test_encodes_np_integer(self) -> None:
        val = np.int32(7)
        result = json.loads(json.dumps(val, cls=NumpyEncoder))
        assert result == 7
        assert isinstance(result, int)

    def test_encodes_np_floating(self) -> None:
        val = np.float32(3.14)
        result = json.loads(json.dumps(val, cls=NumpyEncoder))
        assert abs(result - 3.14) < 1e-3

    def test_raises_for_unknown_type(self) -> None:
        with pytest.raises(TypeError):
            json.dumps(object(), cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# TrackingDataExporter – set_video_metadata
# ---------------------------------------------------------------------------

class TestSetVideoMetadata:
    def test_stores_all_fields(self, exporter: TrackingDataExporter) -> None:
        exporter.set_video_metadata("vid.mp4", 200, 25.0, 640, 480)
        md = exporter.video_metadata
        assert md["input_path"] == "vid.mp4"
        assert md["total_frames"] == 200
        assert md["fps"] == 25.0
        assert md["width"] == 640
        assert md["height"] == 480
        assert "export_timestamp" in md

    def test_overwrites_previous(self, exporter: TrackingDataExporter) -> None:
        exporter.set_video_metadata("a.mp4", 10, 10.0, 100, 100)
        exporter.set_video_metadata("b.mp4", 20, 20.0, 200, 200)
        assert exporter.video_metadata["input_path"] == "b.mp4"


# ---------------------------------------------------------------------------
# TrackingDataExporter – add_frame_data
# ---------------------------------------------------------------------------

class TestAddFrameData:
    def test_single_detection_stored(self, exporter: TrackingDataExporter) -> None:
        det = make_detection()
        exporter.add_frame_data(5, [det], {0: 42})
        assert 42 in exporter.tracking_data
        assert 5 in exporter.tracking_data[42]["frames"]

    def test_start_and_end_frame(self, exporter: TrackingDataExporter) -> None:
        det = make_detection()
        exporter.add_frame_data(3, [det], {0: 1})
        exporter.add_frame_data(7, [det], {0: 1})
        assert exporter.tracking_data[1]["start_frame"] == 3
        assert exporter.tracking_data[1]["end_frame"] == 7

    def test_bbox_padded_with_confidence(self, exporter: TrackingDataExporter) -> None:
        det = make_detection(bbox=np.array([0.0, 0.0, 10.0, 10.0]))
        exporter.add_frame_data(0, [det], {0: 1})
        bbox = exporter.tracking_data[1]["frames"][0]["bbox"]
        assert len(bbox) == 5

    def test_bbox_with_existing_confidence_not_double_padded(
        self, exporter: TrackingDataExporter
    ) -> None:
        det = make_detection(bbox=np.array([0.0, 0.0, 10.0, 10.0, 0.95]))
        exporter.add_frame_data(0, [det], {0: 1})
        bbox = exporter.tracking_data[1]["frames"][0]["bbox"]
        assert len(bbox) == 5

    def test_det_idx_out_of_range_skipped(self, exporter: TrackingDataExporter) -> None:
        det = make_detection()
        exporter.add_frame_data(0, [det], {5: 1})  # det_idx=5 but only 1 detection
        assert 1 not in exporter.tracking_data

    def test_numpy_bbox_converted_to_list(self, exporter: TrackingDataExporter) -> None:
        det = make_detection(bbox=np.array([1.0, 2.0, 3.0, 4.0]))
        exporter.add_frame_data(0, [det], {0: 1})
        bbox = exporter.tracking_data[1]["frames"][0]["bbox"]
        assert isinstance(bbox, list)

    def test_list_bbox_accepted(self, exporter: TrackingDataExporter) -> None:
        det = make_detection(bbox=[1.0, 2.0, 3.0, 4.0])
        exporter.add_frame_data(0, [det], {0: 1})
        assert 0 in exporter.tracking_data[1]["frames"]


# ---------------------------------------------------------------------------
# TrackingDataExporter – remove_track
# ---------------------------------------------------------------------------

class TestRemoveTrack:
    def test_returns_true_when_removed(self, exporter: TrackingDataExporter) -> None:
        exporter.add_frame_data(0, [make_detection()], {0: 99})
        assert exporter.remove_track(99) is True
        assert 99 not in exporter.tracking_data

    def test_returns_false_when_missing(self, exporter: TrackingDataExporter) -> None:
        assert exporter.remove_track(999) is False


# ---------------------------------------------------------------------------
# TrackingDataExporter – _process_keypoints
# ---------------------------------------------------------------------------

class TestProcessKeypoints:
    def test_numpy_2d_three_cols(self, exporter: TrackingDataExporter) -> None:
        kpts = np.array([[1.0, 2.0, 0.8], [3.0, 4.0, 0.9]])
        result = exporter._process_keypoints(kpts)
        assert len(result) == 2
        assert result[0] == [1.0, 2.0, 0.8]

    def test_numpy_2d_two_cols(self, exporter: TrackingDataExporter) -> None:
        kpts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = exporter._process_keypoints(kpts)
        assert result[0][2] == 0.0  # confidence filled in

    def test_torch_tensor_3d_batch(self, exporter: TrackingDataExporter) -> None:
        # Shape (1, 5, 3) – batch dimension should be removed
        kpts = torch.rand(1, 5, 3)
        result = exporter._process_keypoints(kpts)
        assert len(result) == 5

    def test_torch_tensor_2d(self, exporter: TrackingDataExporter) -> None:
        kpts = torch.rand(4, 3)
        result = exporter._process_keypoints(kpts)
        assert len(result) == 4
        for pt in result:
            assert len(pt) == 3

    def test_returns_floats(self, exporter: TrackingDataExporter) -> None:
        kpts = np.array([[1, 2, 1]], dtype=np.int32)
        result = exporter._process_keypoints(kpts)
        for val in result[0]:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# TrackingDataExporter – finalize_data
# ---------------------------------------------------------------------------

class TestFinalizeData:
    def test_processing_time_stored(self, exporter: TrackingDataExporter) -> None:
        exporter.set_video_metadata("v.mp4", 10, 30.0, 100, 100)
        exporter.finalize_data(12.5)
        assert exporter.video_metadata["processing_time"] == 12.5


# ---------------------------------------------------------------------------
# TrackingDataExporter – export_to_json
# ---------------------------------------------------------------------------

class TestExportToJson:
    def test_creates_json_file(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "output")
            populated_exporter.export_to_json(out)
            assert os.path.exists(out + ".json")

    def test_json_extension_not_doubled(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "output.json")
            populated_exporter.export_to_json(out)
            assert os.path.exists(out)
            assert not os.path.exists(out + ".json")

    def test_output_structure(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            populated_exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            assert "video_metadata" in data
            assert "frame_data" in data
            assert "track_summary" in data
            assert "export_info" in data

    def test_frame_data_keys(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            populated_exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            for frame in data["frame_data"].values():
                assert "frame_number" in frame
                assert "detections" in frame

    def test_detections_sorted_by_track_id(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            populated_exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            for frame in data["frame_data"].values():
                ids = [d["track_id"] for d in frame["detections"]]
                assert ids == sorted(ids)

    def test_detection_fields(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            populated_exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            det = data["frame_data"]["0"]["detections"][0]
            assert "track_id" in det
            assert "bbox" in det
            assert "confidence" in det
            assert "keypoints" in det

    def test_export_info_counts(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            populated_exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            assert data["export_info"]["total_tracks"] == 2
            assert data["export_info"]["total_frames"] == 3

    def test_empty_exporter_exports_empty_frames(
        self, exporter: TrackingDataExporter
    ) -> None:
        exporter.set_video_metadata("v.mp4", 10, 30.0, 100, 100)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "empty.json")
            exporter.export_to_json(out)
            with open(out) as f:
                data = json.load(f)
            assert data["frame_data"] == {}


# ---------------------------------------------------------------------------
# TrackingDataExporter – export_tracks_to_hdf5
# ---------------------------------------------------------------------------

class TestExportTracksToHdf5:
    def test_creates_h5_files(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile(), 2: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            files = list(Path(tmp).glob("*.h5"))
            assert len(files) == 2

    def test_skips_filtered_tracks(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Only provide profile for track 1 – track 2 should be skipped
            profiles = {1: make_profile()}
            count = populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            assert count == 1
            files = list(Path(tmp).glob("*.h5"))
            assert len(files) == 1

    def test_returns_exported_count(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile(), 2: make_profile()}
            count = populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            assert count == 2

    def test_h5_metadata_attrs(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert f["metadata"].attrs["track_id"] == 1
                assert f["metadata"].attrs["start_frame"] == 0
                assert f["metadata"].attrs["end_frame"] == 2
                assert f["metadata"].attrs["num_frames"] == 3

    def test_h5_video_metadata_stored(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert f["metadata"].attrs["video_fps"] == 30.0
                assert f["metadata"].attrs["video_width"] == 1920
                assert f["metadata"].attrs["video_height"] == 1080

    def test_h5_frames_group_exists(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert "frames" in f
                assert "frame_000000" in f["frames"]

    def test_h5_bbox_shape(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                bbox = f["frames"]["frame_000000"]["bbox"][:]
                assert bbox.shape == (4,)

    def test_h5_confidence_attr(self, populated_exporter: TrackingDataExporter) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                conf = f["frames"]["frame_000000"].attrs["confidence"]
                assert 0.0 <= float(conf) <= 1.0

    def test_h5_keypoints_stored(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                kpts = f["frames"]["frame_000000"]["keypoints"][:]
                assert kpts.ndim == 2
                assert kpts.shape[1] == 3

    def test_h5_embeddings_stored(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile(face=True, upper=True, lower=True)}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert "face_feature" in f["embeddings"]
                assert "upper_feature" in f["embeddings"]
                assert "lower_feature" in f["embeddings"]

    def test_h5_missing_embeddings_skipped(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile(face=False, upper=False, lower=False)}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert "face_feature" not in f["embeddings"]

    def test_h5_creation_frame_attr(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = {1: make_profile(creation_frame=5)}
            populated_exporter.export_tracks_to_hdf5(tmp, profiles)
            h5_file = next(Path(tmp).glob("track_0001.h5"))
            with h5py.File(h5_file, "r") as f:
                assert f["embeddings"].attrs["creation_frame"] == 5

    def test_output_dir_created_if_missing(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "subdir", "hdf5_out")
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(new_dir, profiles)
            assert Path(new_dir).is_dir()

    def test_file_at_output_dir_replaced_by_dir(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            collision = os.path.join(tmp, "collision")
            # Create a file where the directory should be
            Path(collision).write_text("oops")
            profiles = {1: make_profile()}
            populated_exporter.export_tracks_to_hdf5(collision, profiles)
            assert Path(collision).is_dir()

    def test_empty_profiles_exports_nothing(
        self, populated_exporter: TrackingDataExporter
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            count = populated_exporter.export_tracks_to_hdf5(tmp, {})
            assert count == 0


# ---------------------------------------------------------------------------
# TrackingDataCollector
# ---------------------------------------------------------------------------

class TestTrackingDataCollector:
    def test_default_init(self) -> None:
        collector = TrackingDataCollector()
        assert collector.output_path is None
        assert collector.enable_hdf5 is False

    def test_collect_frame_data(self) -> None:
        collector = TrackingDataCollector()
        det = make_detection()
        collector.collect_frame_data(0, [det], {0: 1})
        assert 1 in collector.exporter.tracking_data

    def test_set_video_info(self) -> None:
        collector = TrackingDataCollector()
        collector.set_video_info("v.mp4", 50, 24.0, 320, 240)
        assert collector.exporter.video_metadata["fps"] == 24.0

    def test_remove_track(self, capsys: pytest.CaptureFixture[str]) -> None:
        collector = TrackingDataCollector()
        collector.collect_frame_data(0, [make_detection()], {0: 7})
        collector.remove_track(7)
        assert 7 not in collector.exporter.tracking_data
        captured = capsys.readouterr()
        assert "7" in captured.out

    def test_remove_nonexistent_track_silent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        collector = TrackingDataCollector()
        collector.remove_track(999)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_export_data_json_only(self) -> None:
        collector = TrackingDataCollector()
        collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
        collector.collect_frame_data(0, [make_detection()], {0: 1})
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result")
            collector.export_data(out, 1.0)
            assert os.path.exists(out + ".json")

    def test_export_data_uses_output_path_over_arg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixed = os.path.join(tmp, "fixed")
            collector = TrackingDataCollector(output_path=fixed)
            collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
            collector.collect_frame_data(0, [make_detection()], {0: 1})
            collector.export_data(os.path.join(tmp, "ignored"), 1.0)
            assert os.path.exists(fixed + ".json")

    def test_export_data_json_extension_not_doubled(self) -> None:
        collector = TrackingDataCollector()
        collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
        collector.collect_frame_data(0, [make_detection()], {0: 1})
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            collector.export_data(out, 1.0)
            assert os.path.exists(out)
            assert not os.path.exists(out + ".json")

    def test_export_data_hdf5_enabled(self) -> None:
        collector = TrackingDataCollector(enable_hdf5=True)
        collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
        collector.collect_frame_data(0, [make_detection()], {0: 1})
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result")
            profiles = {1: make_profile()}
            collector.export_data(out, 1.0, person_profiles=profiles)
            hdf5_dir = out + "_hdf5"
            assert Path(hdf5_dir).is_dir()
            assert len(list(Path(hdf5_dir).glob("*.h5"))) == 1

    def test_export_data_hdf5_disabled_no_dir(self) -> None:
        collector = TrackingDataCollector(enable_hdf5=False)
        collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
        collector.collect_frame_data(0, [make_detection()], {0: 1})
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result")
            profiles = {1: make_profile()}
            collector.export_data(out, 1.0, person_profiles=profiles)
            assert not Path(out + "_hdf5").exists()

    def test_export_data_hdf5_path_already_has_suffix(self) -> None:
        collector = TrackingDataCollector(enable_hdf5=True)
        collector.set_video_info("v.mp4", 10, 30.0, 100, 100)
        collector.collect_frame_data(0, [make_detection()], {0: 1})
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result_hdf5")
            profiles = {1: make_profile()}
            collector.export_data(out, 1.0, person_profiles=profiles)
            assert Path(out).is_dir()
            # JSON should strip the _hdf5 suffix
            assert os.path.exists(os.path.join(tmp, "result.json"))