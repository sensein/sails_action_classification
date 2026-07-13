"""
src/tests/tracking_pose_model_testing/test_poseformer.py

Smoke test for the PoseFormer video-trimming/copy script.
  Script under test : src/sailsprep/tracking_pose_model_testing/poseformer.py
  This test file    : src/tests/tracking_pose_model_testing/test_poseformer.py

This script defines no functions -- it is pure top-level orchestration: it
makes an output directory, shells out to `ffmpeg` unconditionally to trim a
hardcoded input video, and then conditionally copies a hardcoded output path
if it exists. There is no business logic to unit test directly, so this
file verifies the module executes top-to-bottom without raising once
`os.makedirs` and `subprocess.run` are patched (so the unconditional ffmpeg
call at import time doesn't require a real ffmpeg binary or input file), and
that a couple of expected top-level constants are present.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_poseformer.py -v
"""

from __future__ import annotations

import importlib.util
import unittest.mock as mock
from pathlib import Path

import pytest


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "poseformer.py"


@pytest.mark.unit
class TestPoseformerSmoke:

    def test_module_loads_without_raising(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("poseformer_pipeline", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            mock.patch("os.makedirs"),
            mock.patch("subprocess.run", return_value=mock.MagicMock()),
            mock.patch("os.path.exists", return_value=False),
            mock.patch("shutil.copy"),
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert mod.input_folder == "/input"
        assert mod.output_folder == "/outputs/PoseFormer"
        assert mod.video_name == "video.mkv"

    def test_ffmpeg_invoked_with_expected_args(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("poseformer_pipeline2", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            mock.patch("os.makedirs"),
            mock.patch("subprocess.run", return_value=mock.MagicMock()) as mock_run,
            mock.patch("os.path.exists", return_value=False),
            mock.patch("shutil.copy"),
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert mod.input_path in cmd

    def test_copies_output_when_it_exists(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("poseformer_pipeline3", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            mock.patch("os.makedirs"),
            mock.patch("subprocess.run", return_value=mock.MagicMock()),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("shutil.copy") as mock_copy,
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        mock_copy.assert_called_once()
