"""
src/tests/tracking_pose_model_testing/test_openpose_video.py

Unit tests for the OpenPose batch video-processing helpers.
  Script under test : src/sailsprep/tracking_pose_model_testing/openpose_video.py
  This test file    : src/tests/tracking_pose_model_testing/test_openpose_video.py

Only unusual import is `from openpose import pyopenpose as op`, which is
stubbed since the real `openpose` bindings are not installed. All the logic
under test lives inside plain functions guarded by `if __name__ ==
"__main__":`, so the module is importable normally once `openpose` is
stubbed.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_openpose_video.py -v
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
import unittest.mock as mock
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


def _heal_real_module(name: str, required_attr: str) -> None:
    """
    Some other test files in this directory replace sys.modules[name] with a
    bare/incomplete stub without restoring it (e.g. test_hrnet.py permanently
    stubs cv2 with only a handful of attributes). If that happened before
    this file's tests run, drop the stub and re-import the real package so
    functions like cv2.VideoWriter_fourcc are available here.
    """
    mod = sys.modules.get(name)
    if mod is None or not hasattr(mod, required_attr):
        sys.modules.pop(name, None)
        globals()[name] = importlib.import_module(name)


_heal_real_module("cv2", "VideoWriter_fourcc")

# ─────────────────────────────────────────────────────────────────────────────
# Stub the `openpose` package before importing the module under test
# ─────────────────────────────────────────────────────────────────────────────


def _make_op_stub() -> types.ModuleType:
    op_mod = types.ModuleType("openpose.pyopenpose")
    op_mod.WrapperPython = mock.MagicMock
    op_mod.Datum = mock.MagicMock
    op_mod.VectorDatum = mock.MagicMock
    openpose_pkg = types.ModuleType("openpose")
    openpose_pkg.pyopenpose = op_mod
    return openpose_pkg, op_mod


_openpose_pkg, _op_mod = _make_op_stub()
sys.modules.setdefault("openpose", _openpose_pkg)
sys.modules["openpose"].pyopenpose = _op_mod
sys.modules.setdefault("openpose.pyopenpose", _op_mod)

from sailsprep.tracking_pose_model_testing.openpose_video import (  # noqa: E402
    check_ffmpeg_available,
    process_csv_videos,
    process_video,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
#
# NOTE: `mock.patch("cv2.X", ...)` below is intentionally NOT used to mock
# OpenCV calls -- it patches whatever object sys.modules["cv2"] currently
# points to, which can be a *different* module object than the one already
# bound inside sailsprep.tracking_pose_model_testing.openpose_video's own
# namespace (bound once, at that module's import time). If some other test
# file collected later permanently reassigns sys.modules["cv2"] (as
# test_hrnet.py does), patching the bare "cv2.X" path silently patches the
# wrong object and the mock never takes effect. Patching the fully-qualified
# `sailsprep...openpose_video.cv2.X` path always targets the exact object
# the function under test actually calls.
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_video(path: Path, n_frames: int = 5, size=(64, 64)) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: check_ffmpeg_available
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCheckFfmpegAvailable:

    def test_success(self):
        with mock.patch("subprocess.run", return_value=mock.MagicMock()):
            assert check_ffmpeg_available() is True

    def test_called_process_error(self):
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
        ):
            assert check_ffmpeg_available() is False

    def test_file_not_found(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_ffmpeg_available() is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: process_video
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestProcessVideo:

    def test_missing_input_returns_false(self, tmp_path):
        missing = tmp_path / "does_not_exist.mp4"
        result = process_video(str(missing), str(tmp_path / "out.mp4"))
        assert result is False

    def test_openpose_init_exception_returns_false(self, tmp_path):
        video_path = make_dummy_video(tmp_path / "in.mp4")
        with mock.patch(
            "sailsprep.tracking_pose_model_testing.openpose_video.op.WrapperPython",
            side_effect=RuntimeError("boom"),
        ):
            result = process_video(str(video_path), str(tmp_path / "out.mp4"), use_ffmpeg=False)
        assert result is False

    def test_cannot_open_video(self, tmp_path):
        video_path = make_dummy_video(tmp_path / "in.mp4")
        fake_wrapper = mock.MagicMock()
        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = False
        with (
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.WrapperPython",
                return_value=fake_wrapper,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.cv2.VideoCapture",
                return_value=fake_cap,
            ),
        ):
            result = process_video(str(video_path), str(tmp_path / "out.mp4"), use_ffmpeg=False)
        assert result is False

    def test_opencv_writer_fallback_path(self, tmp_path):
        video_path = make_dummy_video(tmp_path / "in.mp4", n_frames=2)
        fake_wrapper = mock.MagicMock()

        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = True
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
        fake_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 64,
            cv2.CAP_PROP_FRAME_COUNT: 2,
        }.get(prop, 0)

        fake_writer = mock.MagicMock()
        fake_writer.isOpened.return_value = True

        fake_datum = mock.MagicMock()
        fake_datum.cvOutputData = frame

        with (
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.WrapperPython",
                return_value=fake_wrapper,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.Datum",
                return_value=fake_datum,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.VectorDatum",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.cv2.VideoCapture",
                return_value=fake_cap,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.cv2.VideoWriter",
                return_value=fake_writer,
            ),
        ):
            result = process_video(str(video_path), str(tmp_path / "out.mp4"), use_ffmpeg=False)

        assert result is True
        assert fake_writer.write.call_count == 2

    def test_ffmpeg_path(self, tmp_path):
        video_path = make_dummy_video(tmp_path / "in.mp4", n_frames=1)
        fake_wrapper = mock.MagicMock()

        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = True
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_cap.read.side_effect = [(True, frame), (False, None)]
        fake_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 64,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }.get(prop, 0)

        fake_datum = mock.MagicMock()
        fake_datum.cvOutputData = frame

        fake_process = mock.MagicMock()
        fake_process.returncode = 0
        fake_process.poll.return_value = 0

        with (
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.WrapperPython",
                return_value=fake_wrapper,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.Datum",
                return_value=fake_datum,
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.op.VectorDatum",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "sailsprep.tracking_pose_model_testing.openpose_video.cv2.VideoCapture",
                return_value=fake_cap,
            ),
            mock.patch("subprocess.Popen", return_value=fake_process),
        ):
            result = process_video(str(video_path), str(tmp_path / "out.mp4"), use_ffmpeg=True)

        assert result is True
        fake_process.stdin.write.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: process_csv_videos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestProcessCsvVideos:

    def test_missing_csv_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            process_csv_videos(str(tmp_path / "nope.csv"), str(tmp_path / "out"))
        assert exc_info.value.code == 1

    def test_missing_column_exits(self, tmp_path):
        csv_path = tmp_path / "videos.csv"
        pd.DataFrame({"other_col": ["a.mp4"]}).to_csv(csv_path, index=False)
        with pytest.raises(SystemExit) as exc_info:
            process_csv_videos(str(csv_path), str(tmp_path / "out"))
        assert exc_info.value.code == 1

    def test_existing_output_is_skipped(self, tmp_path):
        csv_path = tmp_path / "videos.csv"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        video_name = "clip1.mp4"
        pd.DataFrame({"BidsProcessed": [video_name]}).to_csv(csv_path, index=False)

        # Pre-create the expected output file so it should be skipped.
        (out_dir / "clip1_openpose.mp4").write_bytes(b"x")

        with mock.patch(
            "sailsprep.tracking_pose_model_testing.openpose_video.process_video"
        ) as mock_process:
            process_csv_videos(str(csv_path), str(out_dir))

        mock_process.assert_not_called()

    def test_calls_process_video_for_each_row(self, tmp_path):
        csv_path = tmp_path / "videos.csv"
        out_dir = tmp_path / "out"

        pd.DataFrame({"BidsProcessed": ["clip1.mp4", "clip2.mp4"]}).to_csv(csv_path, index=False)

        with mock.patch(
            "sailsprep.tracking_pose_model_testing.openpose_video.process_video",
            return_value=True,
        ) as mock_process:
            process_csv_videos(str(csv_path), str(out_dir))

        assert mock_process.call_count == 2
