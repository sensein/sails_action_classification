"""
Tests for single_child_id_api.py

Run with:
    poetry run pytest src/tests/test_single_child_id_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal but realistic fake objects
# ---------------------------------------------------------------------------

def _make_tracking_data(
    num_tracks: int = 2,
    frames_per_track: int = 5,
    fps: float = 30.0,
    input_path: str = "/data/video.mp4",
    total_frames: int = 150,
    width: int = 1920,
    height: int = 1080,
) -> Dict[str, Any]:
    """Return a dict shaped like a real tracking JSON file."""
    tracking_results: Dict[str, Any] = {}
    for t in range(num_tracks):
        start = t * frames_per_track
        end = start + frames_per_track - 1
        frames: Dict[str, Any] = {}
        for f in range(start, end + 1):
            frames[str(f)] = {
                "keypoints": [[float(f), float(f), 0.9]] * 17,
                "bbox": [10.0, 20.0, 100.0, 200.0],
            }
        tracking_results[str(t)] = {
            "start_frame": start,
            "end_frame": end,
            "frames": frames,
        }
    return {
        "video_metadata": {
            "fps": fps,
            "input_path": input_path,
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
        "tracking_results": tracking_results,
    }


def _make_segment(track_id: int, start: int, end: int, fps: float = 30.0) -> Mock:
    """Return a mock Track/segment with duration helpers."""
    seg = Mock()
    seg.id = track_id
    seg.start_frame = start
    seg.end_frame = end
    seg.duration_seconds = Mock(return_value=(end - start + 1) / fps)
    seg.duration_frames = Mock(return_value=end - start + 1)
    return seg


def _make_node(track_id: int, index: int, selected: bool = False) -> Mock:
    node = Mock()
    node.tracklet = Mock()
    node.tracklet.id = track_id
    node.tracklet.duration_seconds = Mock(return_value=1.0)
    node.score = 0.85
    node.weight = 1.0
    node.evidence = Mock()
    node.evidence.flags = ["child"]
    node.evidence.p_age = 0.9
    node.evidence.p_skeleton = 0.8
    node.evidence.p_rigidity = 0.7
    return node


def _make_edge(src: int, dst: int) -> Mock:
    edge = Mock()
    edge.src_index = src
    edge.dst_index = dst
    edge.score = 0.75
    edge.reasons = {"temporal": 0.5, "overlap": 0.25}
    return edge


def _make_child_result(track_ids: list[int] | None = None) -> Mock:
    """Return a mock ChildResult."""
    if track_ids is None:
        track_ids = [0]

    result = Mock()
    result.child_track_id_sequence = track_ids
    result.confidence = 0.92
    result.uncertainty = 0.08

    segments = [_make_segment(tid, tid * 5, tid * 5 + 4) for tid in track_ids]
    result.segments = segments

    nodes = [_make_node(tid, i) for i, tid in enumerate(track_ids)]
    edges = [_make_edge(0, 1)] if len(track_ids) > 1 else []

    result.diagnostics = {
        "nodes": nodes,
        "edges": edges,
        "path_indices": list(range(len(track_ids))),
    }
    return result


def _make_config() -> Mock:
    cfg = Mock()
    cfg.age_estimation_method = "siglip"
    cfg.enable_body_visibility_filter = True
    cfg.min_visible_keypoints = 4
    cfg.min_track_frames = 10
    cfg.sampling_percentage = 0.25
    cfg.sampling_max_frames_per_track = 30
    cfg.age_child_years_threshold = 10.0
    cfg.enable_skeleton_ratios = False
    cfg.skeleton_min_confidence = 0.3
    cfg.min_rigidity_score = 0.5
    return cfg


# ---------------------------------------------------------------------------
# Module path used in all patches
# ---------------------------------------------------------------------------
_MOD = "sailsprep.id_tracking_model.target_id.child_id.single_child_id_api"


# ===========================================================================
# convert_tracking_json_to_tracks
# ===========================================================================

class TestConvertTrackingJsonToTracks:
    """Tests for convert_tracking_json_to_tracks."""

    def _import(self):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            convert_tracking_json_to_tracks,
        )
        return convert_tracking_json_to_tracks

    def test_returns_correct_number_of_tracks(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=3)
        tracks = fn(data)
        assert len(tracks) == 3

    def test_track_ids_match_json_keys(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=2)
        tracks = fn(data)
        track_ids = {t.id for t in tracks}
        assert track_ids == {0, 1}

    def test_fps_propagated_to_tracks(self):
        fn = self._import()
        data = _make_tracking_data(fps=25.0)
        tracks = fn(data)
        assert all(t.fps == 25.0 for t in tracks)

    def test_video_path_propagated_to_tracks(self):
        fn = self._import()
        data = _make_tracking_data(input_path="/videos/test.mp4")
        tracks = fn(data)
        assert all(t.video_path == "/videos/test.mp4" for t in tracks)

    def test_frame_numbers_are_sorted(self):
        fn = self._import()
        # Build data with deliberately unordered frame keys
        data = _make_tracking_data(num_tracks=1, frames_per_track=4)
        # Shuffle the frame keys in the underlying dict
        original = data["tracking_results"]["0"]["frames"]
        shuffled = {k: original[k] for k in sorted(original, reverse=True)}
        data["tracking_results"]["0"]["frames"] = shuffled
        tracks = fn(data)
        fn_nums = tracks[0].frame_numbers
        assert fn_nums == sorted(fn_nums)

    def test_bboxes_stored_as_tuples(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=1, frames_per_track=3)
        tracks = fn(data)
        for bbox in tracks[0].bboxes:
            assert isinstance(bbox, tuple)

    def test_start_end_frame_correct(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=1, frames_per_track=5)
        tracks = fn(data)
        assert tracks[0].start_frame == 0
        assert tracks[0].end_frame == 4

    def test_meta_total_detections_matches_frame_count(self):
        fn = self._import()
        frames_per_track = 6
        data = _make_tracking_data(num_tracks=1, frames_per_track=frames_per_track)
        tracks = fn(data)
        assert tracks[0].meta["total_detections"] == frames_per_track

    def test_empty_tracking_results_returns_empty_list(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=0)
        tracks = fn(data)
        assert tracks == []

    def test_face_crops_is_none(self):
        fn = self._import()
        data = _make_tracking_data(num_tracks=1)
        tracks = fn(data)
        assert tracks[0].face_crops is None


# ===========================================================================
# child_result_to_dict
# ===========================================================================

class TestChildResultToDict:
    """Tests for child_result_to_dict."""

    def _call(self, result=None, tracking_data=None, config=None,
              video_name="test_video", processing_time=1.23):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            child_result_to_dict,
        )
        result = result or _make_child_result()
        tracking_data = tracking_data or _make_tracking_data()
        config = config or _make_config()
        return child_result_to_dict(result, tracking_data, config, video_name, processing_time)

    # --- top-level keys ---

    def test_has_required_top_level_keys(self):
        d = self._call()
        assert set(d.keys()) == {"video_info", "child_identification", "detailed_analysis", "configuration"}

    # --- video_info ---

    def test_video_info_filename(self):
        d = self._call(video_name="my_clip")
        assert d["video_info"]["filename"] == "my_clip"

    def test_video_info_fps(self):
        data = _make_tracking_data(fps=24.0)
        d = self._call(tracking_data=data)
        assert d["video_info"]["fps"] == 24.0

    def test_video_info_processing_time_rounded(self):
        d = self._call(processing_time=3.14159)
        assert d["video_info"]["processing_time_seconds"] == 3.14

    def test_video_info_width_height_present(self):
        data = _make_tracking_data(width=1280, height=720)
        d = self._call(tracking_data=data)
        assert d["video_info"]["width"] == 1280
        assert d["video_info"]["height"] == 720

    def test_video_info_width_defaults_to_unknown_when_missing(self):
        data = _make_tracking_data()
        del data["video_metadata"]["width"]
        d = self._call(tracking_data=data)
        assert d["video_info"]["width"] == "unknown"

    # --- child_identification ---

    def test_child_identification_selected_track_ids(self):
        result = _make_child_result(track_ids=[0, 1])
        d = self._call(result=result)
        assert d["child_identification"]["selected_track_ids"] == [0, 1]

    def test_child_identification_confidence_rounded(self):
        result = _make_child_result()
        result.confidence = 0.9234567
        d = self._call(result=result)
        assert d["child_identification"]["confidence"] == 0.9235

    def test_child_identification_num_segments(self):
        result = _make_child_result(track_ids=[0, 1, 2])
        d = self._call(result=result)
        assert d["child_identification"]["num_segments"] == 3

    def test_child_identification_segments_structure(self):
        result = _make_child_result(track_ids=[0])
        d = self._call(result=result)
        seg = d["child_identification"]["segments"][0]
        assert {"track_id", "start_frame", "end_frame", "duration_seconds", "duration_frames"} <= seg.keys()

    # --- detailed_analysis ---

    def test_detailed_analysis_node_count(self):
        result = _make_child_result(track_ids=[0, 1])
        d = self._call(result=result)
        assert d["detailed_analysis"]["total_nodes"] == 2

    def test_detailed_analysis_edge_count(self):
        result = _make_child_result(track_ids=[0, 1])
        d = self._call(result=result)
        assert d["detailed_analysis"]["total_edges"] == 1

    def test_detailed_analysis_node_score_rounded(self):
        result = _make_child_result(track_ids=[0])
        result.diagnostics["nodes"][0].score = 0.123456
        d = self._call(result=result)
        assert d["detailed_analysis"]["nodes"][0]["score"] == 0.1235

    def test_detailed_analysis_node_selected_flag(self):
        result = _make_child_result(track_ids=[0])
        result.diagnostics["path_indices"] = [0]
        d = self._call(result=result)
        assert d["detailed_analysis"]["nodes"][0]["selected"] is True

    def test_detailed_analysis_none_probs_preserved(self):
        result = _make_child_result(track_ids=[0])
        result.diagnostics["nodes"][0].evidence.p_age = None
        result.diagnostics["nodes"][0].evidence.p_skeleton = None
        result.diagnostics["nodes"][0].evidence.p_rigidity = None
        d = self._call(result=result)
        node = d["detailed_analysis"]["nodes"][0]
        assert node["age_prob"] is None
        assert node["skeleton_prob"] is None
        assert node["rigidity_prob"] is None

    def test_detailed_analysis_edge_reasons_rounded(self):
        result = _make_child_result(track_ids=[0, 1])
        result.diagnostics["edges"][0].reasons = {"temporal": 0.333333}
        d = self._call(result=result)
        assert d["detailed_analysis"]["edges"][0]["reasons"]["temporal"] == 0.3333

    # --- configuration ---

    def test_configuration_reflects_config_object(self):
        cfg = _make_config()
        d = self._call(config=cfg)
        assert d["configuration"]["age_estimation_method"] == "siglip"
        assert d["configuration"]["enable_body_visibility_filter"] is True
        assert d["configuration"]["min_visible_keypoints"] == 4

    # --- JSON serializability ---

    def test_result_is_json_serializable(self):
        d = self._call()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ===========================================================================
# create_child_video
# ===========================================================================

class TestCreateChildVideo:
    """Tests for create_child_video — cv2 and subprocess are fully mocked."""

    def _import(self):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            create_child_video,
        )
        return create_child_video

    def _make_cap(self, width=640, height=480, fps=30.0, total_frames=10):
        """Return a mock cv2.VideoCapture."""
        import numpy as np
        cap = Mock()
        cap.isOpened.return_value = True
        frame = np.zeros((height, width, 3), dtype="uint8")

        # Read returns (True, frame) for total_frames times, then (False, None)
        side_effects = [(True, frame)] * total_frames + [(False, None)]
        cap.read.side_effect = side_effects

        def get_prop(prop):
            return {
                cv2_prop("CAP_PROP_FPS"): fps,
                cv2_prop("CAP_PROP_FRAME_WIDTH"): float(width),
                cv2_prop("CAP_PROP_FRAME_HEIGHT"): float(height),
                cv2_prop("CAP_PROP_FRAME_COUNT"): float(total_frames),
            }.get(prop, 0.0)

        cap.get.side_effect = get_prop
        return cap

    @pytest.fixture(autouse=True)
    def _patch_cv2_cap(self, monkeypatch):
        """Patch cv2.VideoCapture globally for this class."""
        import numpy as np

        self._frame = np.zeros((480, 640, 3), dtype="uint8")
        side_effects = [(True, self._frame)] * 5 + [(False, None)]

        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = side_effects

        prop_map = {}

        def _get(prop):
            defaults = {1: 30.0, 3: 640.0, 4: 480.0, 7: 5.0}  # CAP_PROP_* integers
            return defaults.get(prop, 0.0)

        cap.get.side_effect = _get

        monkeypatch.setattr(f"{_MOD}.cv2.VideoCapture", Mock(return_value=cap))
        self._cap_mock = cap

    def test_returns_false_when_video_cannot_open(self, tmp_path):
        fn = self._import()
        self._cap_mock.isOpened.return_value = False

        result_mock = _make_child_result()
        data = _make_tracking_data()

        ok = fn("/nonexistent.mp4", result_mock, data, tmp_path / "out.mp4")
        assert ok is False

    @patch(f"{_MOD}.subprocess.Popen")
    def test_returns_true_on_success_with_ffmpeg(self, mock_popen, tmp_path):
        fn = self._import()

        proc = Mock()
        proc.stdin = Mock()
        proc.stderr = Mock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        result_mock = _make_child_result()
        data = _make_tracking_data()

        ok = fn("/video.mp4", result_mock, data, tmp_path / "out.mp4")
        assert ok is True

    @patch(f"{_MOD}.subprocess.Popen", side_effect=FileNotFoundError)
    def test_falls_back_to_cv2_when_ffmpeg_missing(self, _mock_popen, tmp_path, monkeypatch):
        fn = self._import()

        writer = Mock()
        monkeypatch.setattr(f"{_MOD}.cv2.VideoWriter", Mock(return_value=writer))
        monkeypatch.setattr(f"{_MOD}.cv2.VideoWriter_fourcc", Mock(return_value=0x7634706d),
                            raising=False)

        result_mock = _make_child_result()
        data = _make_tracking_data()

        ok = fn("/video.mp4", result_mock, data, tmp_path / "out.mp4")
        assert ok is True
        writer.release.assert_called_once()

    @patch(f"{_MOD}.subprocess.Popen")
    def test_returns_false_when_ffmpeg_exits_nonzero(self, mock_popen, tmp_path):
        fn = self._import()

        proc = Mock()
        proc.stdin = Mock()
        proc.stderr = Mock()
        proc.stderr.read.return_value = b"ffmpeg error"
        proc.wait.return_value = 1
        mock_popen.return_value = proc

        result_mock = _make_child_result()
        data = _make_tracking_data()

        ok = fn("/video.mp4", result_mock, data, tmp_path / "out.mp4")
        assert ok is False

    @patch(f"{_MOD}.subprocess.Popen")
    def test_max_frames_limits_processing(self, mock_popen, tmp_path):
        fn = self._import()

        proc = Mock()
        proc.stdin = Mock()
        proc.stderr = Mock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        result_mock = _make_child_result()
        data = _make_tracking_data()

        fn("/video.mp4", result_mock, data, tmp_path / "out.mp4", max_frames=2)
        # Only 2 frames written means stdin.write called at most 2 times
        assert proc.stdin.write.call_count <= 2

    @patch(f"{_MOD}.subprocess.Popen")
    def test_broken_pipe_on_write_is_handled(self, mock_popen, tmp_path):
        fn = self._import()

        proc = Mock()
        proc.stdin = Mock()
        proc.stdin.write.side_effect = BrokenPipeError
        proc.stderr = Mock()
        proc.stderr.read.return_value = b"broken pipe"
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        result_mock = _make_child_result()
        data = _make_tracking_data()

        # Should not raise, should complete gracefully
        ok = fn("/video.mp4", result_mock, data, tmp_path / "out.mp4")
        assert isinstance(ok, bool)


def cv2_prop(name: str) -> int:
    """Return the integer value of a cv2 property constant."""
    import cv2
    return getattr(cv2, name, 0)


# ===========================================================================
# identify_child_in_video
# ===========================================================================

class TestIdentifyChildInVideo:
    """Integration-style tests for identify_child_in_video (all I/O mocked)."""

    @pytest.fixture()
    def tracking_json(self, tmp_path) -> Path:
        data = _make_tracking_data(num_tracks=2, frames_per_track=15)
        p = tmp_path / "clip_tracking.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture()
    def output_video(self, tmp_path) -> Path:
        return tmp_path / "outputs" / "clip_child.mp4"

    @pytest.fixture(autouse=True)
    def _patch_internals(self, monkeypatch):
        """Patch identify_single_child, ChildIdentificationConfig, and create_child_video."""
        self._mock_result = _make_child_result(track_ids=[0])
        self._mock_config = _make_config()

        monkeypatch.setattr(
            f"{_MOD}.identify_single_child",
            Mock(return_value=self._mock_result),
        )
        monkeypatch.setattr(
            f"{_MOD}.ChildIdentificationConfig",
            Mock(return_value=self._mock_config),
        )
        monkeypatch.setattr(
            f"{_MOD}.create_child_video",
            Mock(return_value=True),
        )

    def test_returns_dict_with_required_keys(self, tracking_json, output_video):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        result = identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
        )
        assert "video_info" in result
        assert "child_identification" in result
        assert "detailed_analysis" in result
        assert "configuration" in result

    def test_video_name_strips_tracking_suffix(self, tracking_json, output_video):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        result = identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
        )
        # stem of "clip_tracking.json" is "clip_tracking"; replace removes "_tracking"
        assert result["video_info"]["filename"] == "clip"

    def test_estimated_age_defaults_to_18_months(self, tracking_json, output_video, monkeypatch):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        captured: dict = {}

        original_annotation = __import__(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_id_api",
            fromlist=["AnnotationInfo"],
        ).AnnotationInfo

        def capture_annotation(**kwargs):
            captured.update(kwargs)
            return original_annotation(**kwargs)

        monkeypatch.setattr(f"{_MOD}.AnnotationInfo", capture_annotation)

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
            estimated_age_months=None,
        )
        assert captured.get("age_in_months") == 18.0

    def test_estimated_age_explicit_value_used(self, tracking_json, output_video, monkeypatch):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        captured: dict = {}

        original_annotation = __import__(
            "sailsprep.id_tracking_model.target_id.child_id.single_child_id_api",
            fromlist=["AnnotationInfo"],
        ).AnnotationInfo

        def capture_annotation(**kwargs):
            captured.update(kwargs)
            return original_annotation(**kwargs)

        monkeypatch.setattr(f"{_MOD}.AnnotationInfo", capture_annotation)

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
            estimated_age_months=36.0,
        )
        assert captured.get("age_in_months") == 36.0

    def test_config_parameters_passed_through(self, tracking_json, output_video, monkeypatch):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        config_cls_mock = Mock(return_value=self._mock_config)
        monkeypatch.setattr(f"{_MOD}.ChildIdentificationConfig", config_cls_mock)

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
            age_estimation_method="deepface",
            min_track_frames=20,
            sampling_percentage=0.5,
        )
        call_kwargs = config_cls_mock.call_args.kwargs
        assert call_kwargs["age_estimation_method"] == "deepface"
        assert call_kwargs["min_track_frames"] == 20
        assert call_kwargs["sampling_percentage"] == 0.5

    def test_create_child_video_called_with_correct_output_path(
        self, tracking_json, output_video, monkeypatch
    ):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        video_mock = Mock(return_value=True)
        monkeypatch.setattr(f"{_MOD}.create_child_video", video_mock)

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
        )
        call_kwargs = video_mock.call_args.kwargs
        assert call_kwargs["output_path"] == Path(output_video)

    def test_kwargs_forwarded_to_config(self, tracking_json, output_video, monkeypatch):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        config_cls_mock = Mock(return_value=self._mock_config)
        monkeypatch.setattr(f"{_MOD}.ChildIdentificationConfig", config_cls_mock)

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=output_video,
            custom_param=42,
        )
        call_kwargs = config_cls_mock.call_args.kwargs
        assert call_kwargs.get("custom_param") == 42

    def test_output_directory_created_if_missing(self, tracking_json, tmp_path, monkeypatch):
        from sailsprep.id_tracking_model.target_id.child_id.single_child_id_api import (
            identify_child_in_video,
        )
        deep_output = tmp_path / "a" / "b" / "c" / "out.mp4"
        assert not deep_output.parent.exists()

        identify_child_in_video(
            tracking_json_path=str(tracking_json),
            video_path="/data/clip.mp4",
            video_output_path=deep_output,
        )
        assert deep_output.parent.exists()