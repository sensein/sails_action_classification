"""
Tests for visualize_tracking_from_child_id.py

Run with:
    poetry run pytest src/tests/test_visualize_tracking_from_child_id.py
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from sailsprep.id_tracking_model.visualize_tracking_from_child_id import (
    create_tracking_visualization,
    find_original_video_path,
    find_tracking_json_from_child_id_path,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_tracking_data(video_path: str = "/fake/video.mp4") -> Dict[str, Any]:
    """Return a minimal but valid tracking JSON structure."""
    return {
        "video_metadata": {
            "input_path": video_path,
        },
        "tracking_results": {
            "1": {
                "frames": {
                    "1": {"bbox": [10, 20, 100, 200]},
                    "2": {"bbox": [15, 25, 105, 205]},
                }
            },
            "2": {
                "frames": {
                    "1": {"bbox": [300, 400, 450, 550]},
                }
            },
        },
    }


@pytest.fixture()
def tmp_pipeline(tmp_path: Path) -> Path:
    """
    Build the expected directory tree under tmp_path:

    tmp_path/
      subset/
        tracking/
          123_ABC_tracking.json
        child_classifications/
          videos/
            123_ABC_child_identified.mp4   (empty placeholder)
          logs/
            123_ABC_analysis.json          (empty placeholder)
    """
    subset = tmp_path / "subset"
    tracking_dir = subset / "tracking"
    tracking_dir.mkdir(parents=True)

    videos_dir = subset / "child_classifications" / "videos"
    videos_dir.mkdir(parents=True)

    logs_dir = subset / "child_classifications" / "logs"
    logs_dir.mkdir(parents=True)

    # Write a real tracking JSON so the function can find it
    tracking_json = tracking_dir / "123_ABC_tracking.json"
    tracking_json.write_text(json.dumps(_make_tracking_data()))

    # Placeholder files (just need to exist)
    (videos_dir / "123_ABC_child_identified.mp4").write_bytes(b"")
    (logs_dir / "123_ABC_analysis.json").write_text("{}")

    return tmp_path


# ---------------------------------------------------------------------------
# find_tracking_json_from_child_id_path
# ---------------------------------------------------------------------------

class TestFindTrackingJson:

    def test_finds_json_from_child_identified_video(self, tmp_pipeline: Path) -> None:
        video = tmp_pipeline / "subset" / "child_classifications" / "videos" / "123_ABC_child_identified.mp4"
        result = find_tracking_json_from_child_id_path(video)
        expected = tmp_pipeline / "subset" / "tracking" / "123_ABC_tracking.json"
        assert result == expected

    def test_finds_json_from_analysis_log(self, tmp_pipeline: Path) -> None:
        log = tmp_pipeline / "subset" / "child_classifications" / "logs" / "123_ABC_analysis.json"
        result = find_tracking_json_from_child_id_path(log)
        expected = tmp_pipeline / "subset" / "tracking" / "123_ABC_tracking.json"
        assert result == expected

    def test_returns_none_when_tracking_json_missing(self, tmp_pipeline: Path) -> None:
        # Remove the tracking JSON so the file is not found
        (tmp_pipeline / "subset" / "tracking" / "123_ABC_tracking.json").unlink()
        video = tmp_pipeline / "subset" / "child_classifications" / "videos" / "123_ABC_child_identified.mp4"
        result = find_tracking_json_from_child_id_path(video)
        assert result is None

    def test_returns_none_when_no_tracking_dir(self, tmp_path: Path) -> None:
        # A path with no 'tracking' ancestor directory anywhere
        orphan_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
        orphan_dir.mkdir(parents=True)
        orphan_file = orphan_dir / "some_child_identified.mp4"
        orphan_file.write_bytes(b"")
        result = find_tracking_json_from_child_id_path(orphan_file)
        assert result is None

    def test_generic_stem_used_when_suffix_unknown(self, tmp_pipeline: Path) -> None:
        """A file with no recognised suffix falls back to using its full stem."""
        # Place a tracking JSON matching the raw stem
        tracking_dir = tmp_pipeline / "subset" / "tracking"
        (tracking_dir / "rawfile_tracking.json").write_text("{}")

        # Put the raw file somewhere inside subset
        child_dir = tmp_pipeline / "subset" / "child_classifications" / "videos"
        raw_file = child_dir / "rawfile.mp4"
        raw_file.write_bytes(b"")

        result = find_tracking_json_from_child_id_path(raw_file)
        assert result == tracking_dir / "rawfile_tracking.json"


# ---------------------------------------------------------------------------
# find_original_video_path
# ---------------------------------------------------------------------------

class TestFindOriginalVideoPath:

    def test_returns_path_when_file_exists(self, tmp_path: Path) -> None:
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"")
        data = _make_tracking_data(str(fake_video))
        result = find_original_video_path(data)
        assert result == str(fake_video)

    def test_returns_none_when_file_missing(self) -> None:
        data = _make_tracking_data("/nonexistent/path/video.mp4")
        result = find_original_video_path(data)
        assert result is None

    def test_returns_none_when_key_missing(self) -> None:
        result = find_original_video_path({"video_metadata": {}})
        assert result is None

    def test_returns_none_when_metadata_missing(self) -> None:
        result = find_original_video_path({})
        assert result is None


# ---------------------------------------------------------------------------
# Shared mock helpers for create_tracking_visualization
# ---------------------------------------------------------------------------

def _make_fake_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _base_cap_mock(width: int = 640, height: int = 480, fps: float = 30.0,
                   total_frames: int = 3) -> MagicMock:
    """Return a mock VideoCapture that yields `total_frames` blank frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        0x03: width,   # CAP_PROP_FPS  (actual int values don't matter; side_effect covers all)
        0x04: height,
        0x05: total_frames,
        # Use the real cv2 constants via their numeric value
    }.get(prop, {
        # fallback: map by the actual cv2 constant values
    })

    # Simpler: return correct values positionally via a function
    def cap_get(prop: int) -> float:
        import cv2
        mapping = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
        }
        return mapping.get(prop, 0.0)

    cap.get.side_effect = cap_get

    # Yield `total_frames` real frames then signal end
    frames = [_make_fake_frame(width, height) for _ in range(total_frames)]
    returns = [(True, f) for f in frames] + [(False, None)]
    cap.read.side_effect = returns
    return cap


# ---------------------------------------------------------------------------
# create_tracking_visualization — ffmpeg path
# ---------------------------------------------------------------------------

class TestCreateTrackingVisualizationFfmpeg:

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_returns_true_on_success(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock()

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        tracking_data = _make_tracking_data()
        output = tmp_path / "out.mp4"
        result = create_tracking_visualization("/fake/video.mp4", tracking_data, output)

        assert result is True
        assert proc.stdin.write.called

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_returns_false_when_ffmpeg_exits_nonzero(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock()

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b"some ffmpeg error"
        proc.wait.return_value = 1          # non-zero → failure
        mock_popen.return_value = proc

        tracking_data = _make_tracking_data()
        output = tmp_path / "out.mp4"
        result = create_tracking_visualization("/fake/video.mp4", tracking_data, output)

        assert result is False

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_max_frames_limits_processing(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        total = 10
        mock_cap_cls.return_value = _base_cap_mock(total_frames=total)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        tracking_data = _make_tracking_data()
        output = tmp_path / "out.mp4"
        result = create_tracking_visualization(
            "/fake/video.mp4", tracking_data, output, max_frames=3
        )

        assert result is True
        # stdin.write should be called at most 3 times (one per processed frame)
        assert proc.stdin.write.call_count <= 3

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_broken_pipe_on_write_breaks_loop_gracefully(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(total_frames=5)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdin.write.side_effect = BrokenPipeError
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b"pipe broken"
        proc.wait.return_value = 0          # ffmpeg itself exits 0
        mock_popen.return_value = proc

        tracking_data = _make_tracking_data()
        output = tmp_path / "out.mp4"
        # Should not raise; returns True because ffmpeg rc == 0
        result = create_tracking_visualization("/fake/video.mp4", tracking_data, output)
        assert isinstance(result, bool)

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_returns_false_when_video_cannot_open(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        result = create_tracking_visualization(
            "/fake/video.mp4", _make_tracking_data(), tmp_path / "out.mp4"
        )
        assert result is False

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_zero_fps_falls_back_to_30(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(fps=0.0, total_frames=2)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        result = create_tracking_visualization(
            "/fake/video.mp4", _make_tracking_data(), tmp_path / "out.mp4"
        )
        assert result is True
        # Verify ffmpeg was called with fps=30
        cmd_args = mock_popen.call_args[0][0]
        fps_index = cmd_args.index("-r") + 1
        assert float(cmd_args[fps_index]) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# create_tracking_visualization — cv2 fallback path (ffmpeg not found)
# ---------------------------------------------------------------------------

class TestCreateTrackingVisualizationCv2Fallback:

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoWriter")
    @patch(
        "sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen",
        side_effect=FileNotFoundError,
    )
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_falls_back_to_cv2_writer(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        mock_writer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(total_frames=2)

        writer = MagicMock()
        mock_writer_cls.return_value = writer

        result = create_tracking_visualization(
            "/fake/video.mp4", _make_tracking_data(), tmp_path / "out.mp4"
        )

        assert result is True
        assert writer.write.called
        writer.release.assert_called_once()

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoWriter")
    @patch(
        "sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen",
        side_effect=FileNotFoundError,
    )
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_cv2_fallback_respects_max_frames(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        mock_writer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(total_frames=10)
        writer = MagicMock()
        mock_writer_cls.return_value = writer

        create_tracking_visualization(
            "/fake/video.mp4", _make_tracking_data(), tmp_path / "out.mp4", max_frames=4
        )

        assert writer.write.call_count <= 4


# ---------------------------------------------------------------------------
# create_tracking_visualization — tracking overlay logic
# ---------------------------------------------------------------------------

class TestTrackingOverlay:

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_multiple_tracks_same_frame_all_drawn(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Two tracks share frame 1 — both bboxes should be drawn."""
        mock_cap_cls.return_value = _base_cap_mock(total_frames=1)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        tracking_data: Dict[str, Any] = {
            "video_metadata": {"input_path": "/fake/video.mp4"},
            "tracking_results": {
                "1": {"frames": {"1": {"bbox": [0, 0, 50, 50]}}},
                "2": {"frames": {"1": {"bbox": [100, 100, 200, 200]}}},
            },
        }

        with patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.rectangle") as mock_rect:
            result = create_tracking_visualization("/fake/video.mp4", tracking_data, tmp_path / "out.mp4")

        assert result is True
        # cv2.rectangle is called for bbox + label background per track → at least 4 calls for 2 tracks
        assert mock_rect.call_count >= 4

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_empty_tracking_results_still_succeeds(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(total_frames=2)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        tracking_data: Dict[str, Any] = {
            "video_metadata": {"input_path": "/fake/video.mp4"},
            "tracking_results": {},
        }

        result = create_tracking_visualization("/fake/video.mp4", tracking_data, tmp_path / "out.mp4")
        assert result is True

    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.subprocess.Popen")
    @patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.VideoCapture")
    def test_frame_counter_text_is_added(
        self,
        mock_cap_cls: MagicMock,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cap_cls.return_value = _base_cap_mock(total_frames=1)

        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stderr = MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        with patch("sailsprep.id_tracking_model.visualize_tracking_from_child_id.cv2.putText") as mock_text:
            create_tracking_visualization("/fake/video.mp4", _make_tracking_data(), tmp_path / "out.mp4")

        # At minimum the "Frame: X/Y" counter should be added each frame
        assert mock_text.call_count >= 1
        all_labels = [str(c.args[1]) for c in mock_text.call_args_list]
        assert any("Frame:" in lbl for lbl in all_labels)