"""Tests for batch_pose.py BatchProcessor."""
import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import sys
from unittest.mock import MagicMock

# Mock mmcv and mmpose before importing anything that touches them
sys.modules.setdefault("mmcv", MagicMock())
sys.modules.setdefault("mmpose", MagicMock())
sys.modules.setdefault("mmpose.apis", MagicMock())
sys.modules.setdefault("mmdet", MagicMock())
# Add any other heavy deps that appear in cache_pose.py imports
sys.modules.setdefault("sailsprep.id_tracking_model.pose.cache_pose", MagicMock())

from sailsprep.id_tracking_model.pose.batch_pose import BatchProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FAKE_CSV_CONTENT = (
    "SourceFile,FileName,ID,Coder\n"
    "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub1/vid1.mp4,vid1.mp4,001,coder1\n"
    "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub2/vid2.mp4,vid2.mp4,002,coder2\n"
)


@pytest.fixture()
def csv_file(tmp_path: Any) -> str:
    p = tmp_path / "videos.csv"
    p.write_text(FAKE_CSV_CONTENT)
    return str(p)


@pytest.fixture()
def processor(csv_file: str, tmp_path: Any) -> BatchProcessor:
    """Return a BatchProcessor with pipeline initialisation mocked out."""
    with patch(
        "sailsprep.id_tracking_model.pose.batch_pose.DetectionPosePipeline"
    ) as mock_pipeline_cls:
        mock_pipeline_cls.return_value = MagicMock()
        bp = BatchProcessor(
            csv_path=csv_file,
            output_base_dir=str(tmp_path / "output"),
            base_video_dir=str(tmp_path / "videos"),
            exp_id=None,
            reuse_pipeline=True,
            rmm=False,
            start_row=0,
            end_row=None,
        )
    return bp


# ---------------------------------------------------------------------------
# _read_video_list
# ---------------------------------------------------------------------------

class TestReadVideoList:
    def test_returns_two_entries(self, processor: BatchProcessor) -> None:
        videos = processor._read_video_list()
        assert len(videos) == 2

    def test_entry_fields(self, processor: BatchProcessor) -> None:
        videos = processor._read_video_list()
        v = videos[0]
        assert v["filename"] == "vid1.mp4"
        assert v["video_id"] == "001"
        assert v["coder"] == "coder1"

    def test_missing_csv_returns_empty(self, tmp_path: Any) -> None:
        with patch(
            "sailsprep.id_tracking_model.pose.batch_pose.DetectionPosePipeline"
        ):
            bp = BatchProcessor(
                csv_path=str(tmp_path / "nonexistent.csv"),
                output_base_dir=str(tmp_path / "out"),
                base_video_dir=str(tmp_path / "vid"),
            )
        assert bp._read_video_list() == []


# ---------------------------------------------------------------------------
# _convert_path
# ---------------------------------------------------------------------------

class TestConvertPath:
    def test_strips_volumes_prefix(self, processor: BatchProcessor) -> None:
        src = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub/vid.mp4"
        result = processor._convert_path(src)
        assert result.endswith("sub/vid.mp4")
        assert "/Volumes" not in result

    def test_rmm_mode_joins_directly(self, csv_file: str, tmp_path: Any) -> None:
        with patch(
            "sailsprep.id_tracking_model.pose.batch_pose.DetectionPosePipeline"
        ):
            bp = BatchProcessor(
                csv_path=csv_file,
                output_base_dir=str(tmp_path / "out"),
                base_video_dir="/base",
                rmm=True,
            )
        result = bp._convert_path("relative/path.mp4")
        assert result == "/base/relative/path.mp4"

    def test_fallback_uses_last_two_parts(self, processor: BatchProcessor) -> None:
        src = "/some/other/prefix/subdir/vid.mp4"
        result = processor._convert_path(src)
        assert result.endswith("subdir/vid.mp4")


# ---------------------------------------------------------------------------
# _load_progress / _save_progress
# ---------------------------------------------------------------------------

class TestProgress:
    def test_save_and_reload(self, processor: BatchProcessor) -> None:
        processor.completed_videos = {"vid1.mp4", "vid2.mp4"}
        processor._save_progress()

        assert os.path.exists(processor.progress_file)

        processor.completed_videos = set()
        processor._load_progress()
        assert processor.completed_videos == {"vid1.mp4", "vid2.mp4"}

    def test_load_handles_corrupt_file(self, processor: BatchProcessor) -> None:
        with open(processor.progress_file, "w") as f:
            f.write("not valid json{{{")

        processor._load_progress()  # should not raise
        assert processor.completed_videos == set()

    def test_load_missing_file_starts_empty(self, processor: BatchProcessor) -> None:
        if os.path.exists(processor.progress_file):
            os.remove(processor.progress_file)
        processor._load_progress()
        assert processor.completed_videos == set()


# ---------------------------------------------------------------------------
# _process_single_video
# ---------------------------------------------------------------------------

class TestProcessSingleVideo:
    def _make_video_info(self, filename: str = "vid1.mp4") -> dict[str, Any]:
        return {
            "source_file": "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub/vid1.mp4",
            "filename": filename,
            "video_id": "001",
            "coder": "coder1",
            "row_data": {},
        }

    def test_returns_false_when_source_missing(self, processor: BatchProcessor) -> None:
        info = self._make_video_info()
        # source file definitely doesn't exist on disk
        result = processor._process_single_video(info)
        assert result is False

    def test_returns_true_when_cache_exists(
        self, processor: BatchProcessor, tmp_path: Any
    ) -> None:
        # Create a fake source file
        src = tmp_path / "videos" / "sub" / "vid1.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fake")

        info = self._make_video_info()
        # Point source_file to the fake file via rmm-style (absolute path)
        info["source_file"] = str(src)
        processor.rmm = True

        with patch.object(processor, "_check_cache_exists", return_value=True):
            result = processor._process_single_video(info)

        assert result is True
        # pipeline.process_video should NOT have been called
        processor.pipeline.process_video.assert_not_called()  # type: ignore[union-attr]

    def test_returns_true_on_successful_processing(
        self, processor: BatchProcessor, tmp_path: Any
    ) -> None:
        src = tmp_path / "sub" / "vid1.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fake")

        info = self._make_video_info()
        info["source_file"] = str(src)
        processor.rmm = True

        with patch.object(processor, "_check_cache_exists", return_value=False):
            result = processor._process_single_video(info)

        assert result is True
        processor.pipeline.process_video.assert_called_once()  # type: ignore[union-attr]

    def test_returns_false_on_pipeline_exception(
        self, processor: BatchProcessor, tmp_path: Any
    ) -> None:
        src = tmp_path / "sub" / "vid1.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"fake")

        info = self._make_video_info()
        info["source_file"] = str(src)
        processor.rmm = True

        processor.pipeline.process_video.side_effect = RuntimeError("boom")  # type: ignore[union-attr]

        with patch.object(processor, "_check_cache_exists", return_value=False):
            result = processor._process_single_video(info)

        assert result is False


# ---------------------------------------------------------------------------
# process_all — row range + skip completed
# ---------------------------------------------------------------------------

class TestProcessAll:
    def test_skips_completed_videos(
        self, processor: BatchProcessor, tmp_path: Any
    ) -> None:
        processor.completed_videos = {"vid1.mp4", "vid2.mp4"}

        with patch.object(processor, "_process_single_video") as mock_proc:
            processor.process_all()

        mock_proc.assert_not_called()

    def test_row_range_limits_videos(
        self, processor: BatchProcessor, tmp_path: Any
    ) -> None:
        processor.start_row = 0
        processor.end_row = 1  # only vid1

        processed: list[str] = []

        def fake_process(info: dict[str, Any]) -> bool:
            processed.append(info["filename"])
            return True

        with patch.object(processor, "_process_single_video", side_effect=fake_process):
            with patch.object(processor, "_check_cache_exists", return_value=False):
                processor.process_all()

        assert processed == ["vid1.mp4"]

    def test_failed_video_not_added_to_completed(
        self, processor: BatchProcessor
    ) -> None:
        with patch.object(processor, "_process_single_video", return_value=False):
            processor.process_all()

        assert len(processor.completed_videos) == 0