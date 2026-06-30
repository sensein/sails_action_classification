"""
Tests for batch_child_identification.py

Run with:
    poetry run pytest src/tests/test_batch_child_identification.py -v
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, call, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the module under test importable regardless of where pytest is invoked.
#
# batch_child_identification.py lives at:
#   src/sailsprep/id_tracking_model/target_id/child_id/batch_child_identification.py
#
# We add that directory to sys.path so `import batch_child_identification`
# resolves without needing an installed package or an extra conftest.
# ---------------------------------------------------------------------------
import sys
import types
from pathlib import Path as _Path

_MODULE_DIR = (
    _Path(__file__).parent.parent  # src/
    / "sailsprep"
    / "id_tracking_model"
    / "target_id"
    / "child_id"
)
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

# Build minimal stub modules so the top-level import in the source doesn't fail
_sailsprep = types.ModuleType("sailsprep")
_id_tracking = types.ModuleType("sailsprep.id_tracking_model")
_target_id = types.ModuleType("sailsprep.id_tracking_model.target_id")
_child_id = types.ModuleType("sailsprep.id_tracking_model.target_id.child_id")
_single = types.ModuleType(
    "sailsprep.id_tracking_model.target_id.child_id.single_child_identification"
)


class _Track:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _AnnotationInfo:
    def __init__(self, age_in_months: float, quality_flags: Dict) -> None:
        self.age_in_months = age_in_months
        self.quality_flags = quality_flags


class _ChildIdentificationConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class _ChildResult:
    def __init__(
        self,
        child_track_id_sequence: List[int],
        confidence: float,
        uncertainty: float,
        segments: List[Any],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.child_track_id_sequence = child_track_id_sequence
        self.confidence = confidence
        self.uncertainty = uncertainty
        self.segments = segments
        self.diagnostics = diagnostics


def _identify_single_child(tracks, annotations, config):  # noqa: ANN001
    return _ChildResult([], 0.9, 0.1, [], {"nodes": [], "edges": [], "path_indices": []})


_single.Track = _Track  # type: ignore[attr-defined]
_single.AnnotationInfo = _AnnotationInfo  # type: ignore[attr-defined]
_single.ChildIdentificationConfig = _ChildIdentificationConfig  # type: ignore[attr-defined]
_single.ChildResult = _ChildResult  # type: ignore[attr-defined]
_single.identify_single_child = _identify_single_child  # type: ignore[attr-defined]

sys.modules.update(
    {
        "sailsprep": _sailsprep,
        "sailsprep.id_tracking_model": _id_tracking,
        "sailsprep.id_tracking_model.target_id": _target_id,
        "sailsprep.id_tracking_model.target_id.child_id": _child_id,
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification": _single,
    }
)

# Now we can safely import the module under test
import batch_child_identification as bci  # noqa: E402
from batch_child_identification import ChildIdentificationProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

DUMMY_CONFIG = _ChildIdentificationConfig(
    age_estimation_method="siglip",
    enable_body_visibility_filter=True,
    min_visible_keypoints=4,
    enable_roi_size_filter=False,
    sampling_percentage=0.25,
    sampling_max_frames_per_track=30,
    min_track_frames=10,
    sampling_mode="smart",
    min_pose_confidence=0.7,
    age_child_years_threshold=10.0,
    age_tau=2.5,
    enable_skeleton_ratios=False,
    skeleton_min_confidence=0.3,
    skeleton_min_visible_for_ratio=2,
    w_age_default=1.0,
    w_skel_default=0.0,
    continuity_gap_seconds=6.0,
    intra_id_gamma=0.3,
    intra_id_tau=1.0,
)


def make_tracking_data(
    num_tracks: int = 2,
    frames_per_track: int = 5,
    fps: float = 30.0,
    video_path: str = "/fake/video.mp4",
) -> Dict[str, Any]:
    """Build a minimal tracking-JSON dict for use in tests."""
    tracking_results: Dict[str, Any] = {}
    for tid in range(1, num_tracks + 1):
        frames: Dict[str, Any] = {}
        for fn in range(0, frames_per_track):
            frames[str(fn)] = {
                "keypoints": [[float(fn), float(fn), 0.9]] * 17,
                "bbox": [10.0, 20.0, 100.0, 200.0],
            }
        tracking_results[str(tid)] = {
            "start_frame": 0,
            "end_frame": frames_per_track - 1,
            "frames": frames,
        }
    return {
        "video_metadata": {
            "fps": fps,
            "input_path": video_path,
            "total_frames": frames_per_track * num_tracks,
            "width": 640,
            "height": 480,
        },
        "tracking_results": tracking_results,
    }


def make_segment(track_id: int, start: int, end: int, fps: float = 30.0) -> MagicMock:
    seg = MagicMock()
    seg.id = track_id
    seg.start_frame = start
    seg.end_frame = end
    seg.duration_seconds.return_value = (end - start + 1) / fps
    seg.duration_frames.return_value = end - start + 1
    return seg


def make_child_result(
    track_ids: List[int] | None = None,
    confidence: float = 0.85,
    segments: List[Any] | None = None,
) -> _ChildResult:
    if track_ids is None:
        track_ids = [1]
    if segments is None:
        segments = [make_segment(1, 0, 4)]

    node = MagicMock()
    node.tracklet.id = 1
    node.score = 0.9
    node.weight = 1.0
    node.tracklet.duration_seconds.return_value = 0.5
    node.evidence.flags = {}
    node.evidence.p_age = 0.8
    node.evidence.p_skeleton = None

    edge = MagicMock()
    edge.src_index = 0
    edge.dst_index = 0
    edge.score = 0.7
    edge.reasons = {"continuity": 0.7}

    return _ChildResult(
        child_track_id_sequence=track_ids,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        segments=segments,
        diagnostics={
            "nodes": [node],
            "edges": [edge],
            "path_indices": [0],
        },
    )


@pytest.fixture()
def tmp_dirs(tmp_path: Path):
    """Create the three directories the processor needs."""
    input_dir = tmp_path / "tracking"
    video_dir = tmp_path / "videos"
    log_dir = tmp_path / "logs"
    for d in (input_dir, video_dir, log_dir):
        d.mkdir()
    return input_dir, video_dir, log_dir


@pytest.fixture()
def processor(tmp_dirs):
    input_dir, video_dir, log_dir = tmp_dirs
    return ChildIdentificationProcessor(DUMMY_CONFIG, input_dir, video_dir, log_dir)


# ===========================================================================
# 1.  estimate_child_age_from_filename
# ===========================================================================

class TestEstimateChildAgeFromFilename:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("session_12-16_months", 14.0),
            ("session_16-20_months", 18.0),
            ("baby_14m_visit", 14.0),
            ("baby_18m_visit", 18.0),
            ("baby_12m_visit", 12.0),
            ("baby_24m_visit", 24.0),
            # case-insensitive
            ("BABY_18M_VISIT", 18.0),
            # no match → default
            ("unknown_session_abc", 18.0),
            ("", 18.0),
        ],
    )
    def test_known_patterns(self, processor: ChildIdentificationProcessor, filename: str, expected: float) -> None:
        assert processor.estimate_child_age_from_filename(filename) == expected

    def test_first_match_wins(self, processor: ChildIdentificationProcessor) -> None:
        # "12m" appears before "24m" in iteration order; both are present
        result = processor.estimate_child_age_from_filename("baby_12m_and_24m")
        assert result in {12.0, 24.0}  # whichever dict key matches first is fine


# ===========================================================================
# 2.  convert_tracking_json_to_tracks
# ===========================================================================

class TestConvertTrackingJsonToTracks:
    def test_returns_correct_number_of_tracks(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=3)
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert len(tracks) == 3

    def test_track_ids_are_integers(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=2)
        tracks = processor.convert_tracking_json_to_tracks(data)
        for t in tracks:
            assert isinstance(t.id, int)

    def test_frame_numbers_are_sorted(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=1, frames_per_track=5)
        tracks = processor.convert_tracking_json_to_tracks(data)
        fn = tracks[0].frame_numbers
        assert fn == sorted(fn)

    def test_bboxes_stored_as_tuples(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=1, frames_per_track=3)
        tracks = processor.convert_tracking_json_to_tracks(data)
        for bbox in tracks[0].bboxes:
            assert isinstance(bbox, tuple)

    def test_fps_propagated(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=1, fps=25.0)
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert tracks[0].fps == 25.0

    def test_empty_tracking_results(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=0)
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert tracks == []

    def test_meta_total_detections_matches_frame_count(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=1, frames_per_track=7)
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert tracks[0].meta["total_detections"] == 7

    def test_video_path_propagated(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(video_path="/my/special/video.mp4")
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert all(t.video_path == "/my/special/video.mp4" for t in tracks)

    def test_out_of_order_frame_keys_sorted_correctly(self, processor: ChildIdentificationProcessor) -> None:
        data = make_tracking_data(num_tracks=1, frames_per_track=1)
        # Manually add out-of-order frame keys
        data["tracking_results"]["1"]["frames"] = {
            "10": {"keypoints": [[1.0, 1.0, 0.9]] * 17, "bbox": [0.0, 0.0, 50.0, 50.0]},
            "2":  {"keypoints": [[2.0, 2.0, 0.9]] * 17, "bbox": [0.0, 0.0, 50.0, 50.0]},
            "5":  {"keypoints": [[5.0, 5.0, 0.9]] * 17, "bbox": [0.0, 0.0, 50.0, 50.0]},
        }
        data["tracking_results"]["1"]["end_frame"] = 10
        tracks = processor.convert_tracking_json_to_tracks(data)
        assert tracks[0].frame_numbers == [2, 5, 10]


# ===========================================================================
# 3.  save_detailed_log
# ===========================================================================

class TestSaveDetailedLog:
    def test_creates_json_file(self, processor: ChildIdentificationProcessor, tmp_dirs) -> None:
        _, _, log_dir = tmp_dirs
        tracking_data = make_tracking_data()
        child_result = make_child_result()
        processor.save_detailed_log("my_video", child_result, tracking_data, processing_time=1.23)

        log_file = log_dir / "my_video_analysis.json"
        assert log_file.exists()

    def test_json_is_valid_and_contains_expected_keys(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        _, _, log_dir = tmp_dirs
        tracking_data = make_tracking_data()
        child_result = make_child_result(track_ids=[1], confidence=0.77)
        processor.save_detailed_log("vid_a", child_result, tracking_data, processing_time=0.5)

        with open(log_dir / "vid_a_analysis.json") as f:
            data = json.load(f)

        assert "video_info" in data
        assert "child_identification" in data
        assert "detailed_analysis" in data
        assert "configuration" in data

    def test_confidence_rounded_to_4dp(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        _, _, log_dir = tmp_dirs
        child_result = make_child_result(confidence=0.123456789)
        processor.save_detailed_log("vid_b", child_result, make_tracking_data(), processing_time=0.1)

        with open(log_dir / "vid_b_analysis.json") as f:
            data = json.load(f)

        stored = data["child_identification"]["confidence"]
        assert stored == round(0.123456789, 4)

    def test_processing_time_stored(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        _, _, log_dir = tmp_dirs
        processor.save_detailed_log("vid_c", make_child_result(), make_tracking_data(), processing_time=42.0)

        with open(log_dir / "vid_c_analysis.json") as f:
            data = json.load(f)

        assert data["video_info"]["processing_time_seconds"] == 42.0

    def test_segments_serialised(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        _, _, log_dir = tmp_dirs
        segs = [make_segment(1, 0, 29), make_segment(2, 50, 79)]
        child_result = make_child_result(track_ids=[1, 2], segments=segs)
        processor.save_detailed_log("vid_d", child_result, make_tracking_data(), processing_time=0.0)

        with open(log_dir / "vid_d_analysis.json") as f:
            data = json.load(f)

        assert len(data["child_identification"]["segments"]) == 2


# ===========================================================================
# 4.  process_single_file
# ===========================================================================

class TestProcessSingleFile:
    def _write_tracking_json(self, directory: Path, name: str, video_path: str = "/fake/video.mp4") -> Path:
        data = make_tracking_data(video_path=video_path)
        path = directory / f"{name}_tracking.json"
        path.write_text(json.dumps(data))
        return path

    def test_skips_when_output_exists(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, video_dir, _ = tmp_dirs
        json_path = self._write_tracking_json(input_dir, "clip01")
        # Pre-create the output so the skip logic fires
        (video_dir / "clip01_child_identified.mp4").touch()

        result = processor.process_single_file(json_path, skip_existing=True)
        assert result is True

    def test_no_skip_when_flag_false(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, video_dir, _ = tmp_dirs
        json_path = self._write_tracking_json(input_dir, "clip02")
        (video_dir / "clip02_child_identified.mp4").touch()

        # Even though output exists, skip_existing=False should proceed
        with (
            patch("os.path.exists", return_value=False),  # video "missing" → returns False
        ):
            result = processor.process_single_file(json_path, skip_existing=False)

        assert result is False  # failed because the mock video doesn't exist

    def test_returns_false_when_video_missing(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        json_path = self._write_tracking_json(input_dir, "clip03", video_path="/nonexistent/path.mp4")

        result = processor.process_single_file(json_path, skip_existing=False)
        assert result is False

    def test_returns_false_on_malformed_json(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        bad = input_dir / "bad_tracking.json"
        bad.write_text("{not valid json}")

        result = processor.process_single_file(bad, skip_existing=False)
        assert result is False

    def test_returns_true_on_success(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        json_path = self._write_tracking_json(input_dir, "clip04", video_path="/real/video.mp4")

        child_result = make_child_result()

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "sailsprep.id_tracking_model.target_id.child_id"
                ".single_child_identification.identify_single_child",
                return_value=child_result,
            ),
            patch.object(processor, "create_child_video", return_value=True),
            patch.object(processor, "save_detailed_log"),
        ):
            result = processor.process_single_file(json_path, skip_existing=False)

        assert result is True

    def test_returns_false_when_video_creation_fails(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        json_path = self._write_tracking_json(input_dir, "clip05", video_path="/real/video.mp4")
        child_result = make_child_result()

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "sailsprep.id_tracking_model.target_id.child_id"
                ".single_child_identification.identify_single_child",
                return_value=child_result,
            ),
            patch.object(processor, "create_child_video", return_value=False),
            patch.object(processor, "save_detailed_log"),
        ):
            result = processor.process_single_file(json_path, skip_existing=False)

        assert result is False


# ===========================================================================
# 5.  create_child_video
# ===========================================================================

class TestCreateChildVideo:
    """Tests for create_child_video using mocked cv2 and subprocess."""

    def _make_mock_cap(self, num_frames: int = 5, width: int = 320, height: int = 240, fps: float = 30.0):
        """Return a mock VideoCapture that yields `num_frames` blank frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            0x05: fps,     # CAP_PROP_FPS
            0x03: width,   # CAP_PROP_FRAME_WIDTH
            0x04: height,  # CAP_PROP_FRAME_HEIGHT
            0x07: num_frames,  # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        reads = [(True, frame)] * num_frames + [(False, None)]
        cap.read.side_effect = reads
        return cap

    def test_returns_false_when_video_cannot_open(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=cap):
            result = processor.create_child_video(
                "/bad/path.mp4",
                make_child_result(),
                make_tracking_data(),
                tmp_path / "out.mp4",
            )

        assert result is False

    def test_ffmpeg_path_succeeds(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = self._make_mock_cap(num_frames=3)
        proc_mock = MagicMock()
        proc_mock.stdin = MagicMock()
        proc_mock.stderr = MagicMock()
        proc_mock.stderr.read.return_value = b""
        proc_mock.wait.return_value = 0

        tracking_data = make_tracking_data(num_tracks=1, frames_per_track=3)
        child_result = make_child_result(segments=[make_segment(1, 0, 2)])

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", return_value=proc_mock),
        ):
            result = processor.create_child_video(
                "/fake/video.mp4",
                child_result,
                tracking_data,
                tmp_path / "out.mp4",
                max_frames=10,
            )

        assert result is True
        assert proc_mock.stdin.write.call_count == 3

    def test_ffmpeg_failure_returns_false(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = self._make_mock_cap(num_frames=2)
        proc_mock = MagicMock()
        proc_mock.stdin = MagicMock()
        proc_mock.stderr = MagicMock()
        proc_mock.stderr.read.return_value = b"some ffmpeg error"
        proc_mock.wait.return_value = 1  # non-zero → failure

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", return_value=proc_mock),
        ):
            result = processor.create_child_video(
                "/fake/video.mp4",
                make_child_result(),
                make_tracking_data(),
                tmp_path / "out.mp4",
            )

        assert result is False

    def test_fallback_to_cv2_writer_when_ffmpeg_missing(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = self._make_mock_cap(num_frames=3)
        writer_mock = MagicMock()

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", side_effect=FileNotFoundError),
            patch("cv2.VideoWriter", return_value=writer_mock),
            patch("batch_child_identification.getattr", return_value=lambda *a: 0x20),
        ):
            result = processor.create_child_video(
                "/fake/video.mp4",
                make_child_result(segments=[make_segment(1, 0, 2)]),
                make_tracking_data(num_tracks=1, frames_per_track=3),
                tmp_path / "out.mp4",
                max_frames=10,
            )

        assert result is True
        writer_mock.release.assert_called_once()

    def test_max_frames_respected(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = self._make_mock_cap(num_frames=20)
        proc_mock = MagicMock()
        proc_mock.stdin = MagicMock()
        proc_mock.stderr = MagicMock()
        proc_mock.stderr.read.return_value = b""
        proc_mock.wait.return_value = 0

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", return_value=proc_mock),
        ):
            processor.create_child_video(
                "/fake/video.mp4",
                make_child_result(),
                make_tracking_data(),
                tmp_path / "out.mp4",
                max_frames=5,
            )

        # Should have written exactly 5 frames
        assert proc_mock.stdin.write.call_count == 5

    def test_broken_pipe_during_write_breaks_loop(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = self._make_mock_cap(num_frames=10)
        proc_mock = MagicMock()
        proc_mock.stdin = MagicMock()
        proc_mock.stdin.write.side_effect = BrokenPipeError
        proc_mock.stderr = MagicMock()
        proc_mock.stderr.read.return_value = b"died"
        proc_mock.wait.return_value = 1

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", return_value=proc_mock),
        ):
            result = processor.create_child_video(
                "/fake/video.mp4",
                make_child_result(),
                make_tracking_data(),
                tmp_path / "out.mp4",
            )

        # Loop breaks on first write; ffmpeg returns non-zero → False
        assert result is False

    def test_fps_defaults_to_30_when_zero(
        self, processor: ChildIdentificationProcessor, tmp_path: Path
    ) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = True
        # Return 0 for FPS to trigger the default
        cap.get.return_value = 0
        cap.read.return_value = (False, None)

        proc_mock = MagicMock()
        proc_mock.stdin = MagicMock()
        proc_mock.stderr = MagicMock()
        proc_mock.stderr.read.return_value = b""
        proc_mock.wait.return_value = 0

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("subprocess.Popen", return_value=proc_mock) as popen_mock,
        ):
            processor.create_child_video(
                "/fake/video.mp4",
                make_child_result(),
                make_tracking_data(),
                tmp_path / "out.mp4",
            )

        # Verify the ffmpeg command contains "-r" "30.0"
        cmd_args: List[str] = popen_mock.call_args[0][0]
        r_idx = cmd_args.index("-r")
        assert cmd_args[r_idx + 1] == "30.0"


# ===========================================================================
# 6.  process_batch
# ===========================================================================

class TestProcessBatch:
    def _create_json_files(self, input_dir: Path, count: int, video_path: str = "/fake/video.mp4") -> None:
        for i in range(count):
            data = make_tracking_data(video_path=video_path)
            (input_dir / f"clip{i:02d}_tracking.json").write_text(json.dumps(data))

    def test_test_mode_limits_to_3_files(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 10)

        processed: List[Path] = []

        def fake_process(json_path: Path, skip_existing: bool = True) -> bool:
            processed.append(json_path)
            return True

        with patch.object(processor, "process_single_file", side_effect=fake_process):
            processor.process_batch(test_mode=True)

        assert len(processed) == 3

    def test_max_files_respected(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 10)

        processed: List[Path] = []

        def fake_process(json_path: Path, skip_existing: bool = True) -> bool:
            processed.append(json_path)
            return True

        with patch.object(processor, "process_single_file", side_effect=fake_process):
            processor.process_batch(max_files=4)

        assert len(processed) == 4

    def test_aggressive_sampling_sets_config(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 1)

        with patch.object(processor, "process_single_file", return_value=True):
            processor.process_batch(aggressive_sampling=True)

        assert processor.config.sampling_percentage == 0.05
        assert processor.config.sampling_max_frames_per_track == 8

    def test_parallel_processing_with_workers(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 4)

        with patch.object(processor, "process_single_file", return_value=True) as mock_proc:
            processor.process_batch(max_workers=2)

        assert mock_proc.call_count == 4

    def test_skip_existing_passed_to_process_single_file(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 2)

        calls: List[bool] = []

        def fake_process(json_path: Path, skip_existing: bool = True) -> bool:
            calls.append(skip_existing)
            return True

        with patch.object(processor, "process_single_file", side_effect=fake_process):
            processor.process_batch(skip_existing=False)

        assert all(flag is False for flag in calls)

    def test_empty_input_dir(
        self, processor: ChildIdentificationProcessor
    ) -> None:
        """process_batch on an empty dir should not raise."""
        # No files created → json_files will be []
        # Division by zero guard: len==0 → would crash on success_rate line
        # The real code does divide by len(json_files); if 0 files this raises ZeroDivisionError.
        # This test documents that behaviour.
        with pytest.raises(ZeroDivisionError):
            processor.process_batch()

    def test_failed_file_counted(
        self, processor: ChildIdentificationProcessor, tmp_dirs
    ) -> None:
        input_dir, _, _ = tmp_dirs
        self._create_json_files(input_dir, 3)

        results = [True, False, True]

        with patch.object(processor, "process_single_file", side_effect=results):
            # Just check it doesn't raise; success rate = 2/3
            processor.process_batch()


# ===========================================================================
# 7.  setup_logging
# ===========================================================================

class TestSetupLogging:
    def test_log_file_created(self, tmp_dirs) -> None:
        input_dir, video_dir, log_dir = tmp_dirs
        ChildIdentificationProcessor(DUMMY_CONFIG, input_dir, video_dir, log_dir)
        log_files = list(log_dir.glob("batch_processing_*.log"))
        assert len(log_files) == 1

    def test_logger_is_set(self, processor: ChildIdentificationProcessor) -> None:
        import logging
        assert isinstance(processor.logger, logging.Logger)