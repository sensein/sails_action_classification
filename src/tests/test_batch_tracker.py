"""
Tests for BatchTracker in sailsprep.id_tracking_model.tracker.batch_tracker.
All external dependencies (MultiPersonTrackingPipeline, create_batch_config, etc.)
are mocked so these tests run without any GPU / model weights.

The module is loaded by file path (importlib) rather than by package name so the
tests work regardless of whether the real sailsprep package is installed.
"""

import importlib.util
import json
import os
import signal
import sys
import types
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Build fake leaf modules BEFORE loading batch_tracker
# ---------------------------------------------------------------------------

class _VisualizationConfig:
    enable_visualization: bool = True

class _ExportConfig:
    output_path: str = ""
    enable_export: bool = False
    export_hdf5: bool = False

class _CacheConfig:
    cache_base_path: str = "/tmp/cache"

class _FakePipelineConfig:
    visualization = _VisualizationConfig()
    export = _ExportConfig()
    cache = _CacheConfig()

def _fake_create_batch_config(json_output_path: str) -> _FakePipelineConfig:
    return _FakePipelineConfig()

_fake_pipeline_cls = MagicMock()
_fake_pipeline_cls.return_value = MagicMock()

_fake_tracker_module = types.ModuleType("sailsprep.id_tracking_model.tracker.clip.tracker_clip_new")
_fake_tracker_module.MultiPersonTrackingPipeline = _fake_pipeline_cls  # type: ignore[attr-defined]
_fake_tracker_module.PipelineConfig = _FakePipelineConfig              # type: ignore[attr-defined]
_fake_tracker_module.create_batch_config = _fake_create_batch_config   # type: ignore[attr-defined]

_fake_exporter_module = types.ModuleType("sailsprep.id_tracking_model.utils.tracking_exporter_new")
_fake_exporter_module.TrackingDataCollector = MagicMock()              # type: ignore[attr-defined]


def _ensure_pkg(name: str) -> types.ModuleType:
    """Guarantee a real ModuleType exists at every ancestor level."""
    if name in sys.modules and isinstance(sys.modules[name], types.ModuleType):
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Overwrite every ancestor with a real ModuleType unconditionally —
# the installed sailsprep package may have left stale non-package entries.
for _pkg in [
    "sailsprep",
    "sailsprep.id_tracking_model",
    "sailsprep.id_tracking_model.tracker",
    "sailsprep.id_tracking_model.tracker.clip",
    "sailsprep.id_tracking_model.utils",
]:
    _mod = types.ModuleType(_pkg)
    _mod.__path__ = []          # mark as package so sub-imports resolve
    _mod.__package__ = _pkg
    sys.modules[_pkg] = _mod

# Stub the two modules we know about from the original source
sys.modules["sailsprep.id_tracking_model.tracker.clip.tracker_clip_new"] = _fake_tracker_module
sys.modules["sailsprep.id_tracking_model.utils.tracking_exporter_new"] = _fake_exporter_module


class _AutoStubFinder:
    """Meta path finder: auto-stub any sailsprep.* module not already in sys.modules.

    This makes the tests resilient to the exact set of imports inside
    batch_tracker.py without needing to enumerate them all up front.
    The two hand-crafted fakes above take precedence because they are
    already in sys.modules before this finder is consulted.
    """

    def find_module(self, fullname: str, path: object = None) -> "Optional[_AutoStubFinder]":
        if fullname.startswith("sailsprep.") and fullname not in sys.modules:
            return self
        return None

    def load_module(self, fullname: str) -> types.ModuleType:
        if fullname in sys.modules:
            return sys.modules[fullname]
        stub = types.ModuleType(fullname)
        stub.__path__ = []      # type: ignore[attr-defined]
        stub.__package__ = fullname
        # Expose common names that batch_tracker imports from these modules
        stub.MultiPersonTrackingPipeline = _fake_pipeline_cls   # type: ignore[attr-defined]
        stub.PipelineConfig = _FakePipelineConfig               # type: ignore[attr-defined]
        stub.create_batch_config = _fake_create_batch_config    # type: ignore[attr-defined]
        stub.TrackingDataCollector = MagicMock()                # type: ignore[attr-defined]
        sys.modules[fullname] = stub
        return stub


sys.meta_path.insert(0, _AutoStubFinder())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Load batch_tracker.py directly by file path — no package import needed
# ---------------------------------------------------------------------------

def _load_batch_tracker() -> types.ModuleType:
    # Walk up from this file to find the repo src root, then locate the module
    here = Path(__file__).resolve()
    # Climb until we find the marker directory layout, or fall back to scanning
    for parent in here.parents:
        candidate = parent / "src" / "sailsprep" / "id_tracking_model" / "tracker" / "batch_tracker.py"
        if candidate.exists():
            break
    else:
        # Last resort: search from cwd
        cwd = Path.cwd()
        candidates = list(cwd.rglob("batch_tracker.py"))
        candidates = [c for c in candidates if "test" not in c.parts]
        if not candidates:
            raise FileNotFoundError("Cannot locate batch_tracker.py")
        candidate = candidates[0]

    spec = importlib.util.spec_from_file_location(
        "sailsprep.id_tracking_model.tracker.batch_tracker", candidate
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so any internal relative imports resolve
    sys.modules["sailsprep.id_tracking_model.tracker.batch_tracker"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_bt_module = _load_batch_tracker()
BatchTracker = _bt_module.BatchTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_HEADER = "SourceFile,FileName,ID,Coder\n"
CSV_ROW_1  = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub1/vid1.mov,vid1.mov,ID001,CoderA\n"
CSV_ROW_2  = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub2/vid2.mov,vid2.mov,ID002,CoderB\n"
CSV_ROW_3  = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub3/vid3.mov,vid3.mov,ID003,CoderC\n"

MINIMAL_CSV = CSV_HEADER + CSV_ROW_1 + CSV_ROW_2 + CSV_ROW_3


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _make_tracker(
    tmp_path: Path,
    csv_content: str = MINIMAL_CSV,
    reuse_pipeline: bool = False,
    enable_visualization: bool = False,
    filter_ids: Optional[list] = None,
    start_row: int = 0,
    end_row: Optional[int] = None,
    exp_id: Optional[str] = None,
    rmm: bool = False,
) -> BatchTracker:
    """Create a BatchTracker wired to tmp_path without touching the real FS."""
    csv_file = tmp_path / "videos.csv"
    _write_csv(csv_file, csv_content)

    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    # Suppress pipeline construction when reuse_pipeline=True
    _fake_pipeline_cls.reset_mock()
    _fake_pipeline_cls.return_value = MagicMock()

    tracker = BatchTracker(
        csv_path=str(csv_file),
        output_base_dir=str(output_dir),
        base_video_dir=str(video_dir),
        exp_id=exp_id,
        reuse_pipeline=reuse_pipeline,
        rmm=rmm,
        enable_visualization=enable_visualization,
        filter_ids=filter_ids,
        start_row=start_row,
        end_row=end_row,
    )
    return tracker


# ---------------------------------------------------------------------------
# __init__ / directory naming
# ---------------------------------------------------------------------------

class TestInit:
    def test_output_dir_uses_csv_stem(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        assert "videos" in t.output_dir  # csv stem is "videos"

    def test_output_dir_appends_exp_id(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, exp_id="run01")
        assert "run01" in t.output_dir

    def test_output_dir_appends_row_range(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, start_row=2, end_row=5)
        assert "rows2-5" in t.output_dir

    def test_output_dir_appends_rows_no_end(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, start_row=3)
        assert "rows3-end" in t.output_dir

    def test_output_dir_created(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        assert os.path.isdir(t.output_dir)

    def test_progress_file_path_inside_output_dir(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        assert t.progress_file.startswith(t.output_dir)

    def test_pipeline_none_when_not_reusing(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False)
        assert t.pipeline is None

    def test_pipeline_created_when_reusing(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=True)
        assert t.pipeline is not None

    def test_filter_ids_stored_as_set(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, filter_ids=["ID001", "ID002"])
        assert t.filter_ids == {"ID001", "ID002"}

    def test_no_filter_ids_is_none(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, filter_ids=None)
        assert t.filter_ids is None

    def test_completed_videos_empty_on_fresh_start(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        assert t.completed_videos == set()


# ---------------------------------------------------------------------------
# _load_progress / _save_progress
# ---------------------------------------------------------------------------

class TestProgress:
    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        t.completed_videos = {"vid1.mov", "vid2.mov"}
        t._save_progress()

        t2 = _make_tracker(tmp_path)
        # t2 reads the same progress_file because csv name and dirs match
        assert t2.completed_videos == {"vid1.mov", "vid2.mov"}

    def test_load_progress_prints_resume_message(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        t = _make_tracker(tmp_path)
        t.completed_videos = {"vid1.mov"}
        t._save_progress()

        _make_tracker(tmp_path)
        captured = capsys.readouterr()
        assert "Resumed from previous session" in captured.out

    def test_save_progress_writes_json(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        t.completed_videos = {"vid1.mov"}
        t._save_progress()

        with open(t.progress_file) as f:
            data = json.load(f)
        assert "vid1.mov" in data["completed_videos"]
        assert data["csv_path"] == t.csv_path
        assert data["output_dir"] == t.output_dir

    def test_load_progress_handles_corrupt_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        t = _make_tracker(tmp_path)
        Path(t.progress_file).write_text("NOT JSON", encoding="utf-8")
        t._load_progress()  # should not raise
        assert t.completed_videos == set()
        assert "Warning" in capsys.readouterr().out

    def test_no_progress_file_gives_empty_set(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        assert not os.path.exists(t.progress_file) or t.completed_videos == set()


# ---------------------------------------------------------------------------
# _read_video_list
# ---------------------------------------------------------------------------

class TestReadVideoList:
    def test_reads_all_rows(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        videos = t._read_video_list()
        assert len(videos) == 3

    def test_row_fields_populated(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        v = t._read_video_list()[0]
        assert v["filename"] == "vid1.mov"
        assert v["video_id"] == "ID001"
        assert v["coder"] == "CoderA"
        assert "source_file" in v

    def test_filter_ids_applied(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, filter_ids=["ID001"])
        videos = t._read_video_list()
        assert len(videos) == 1
        assert videos[0]["video_id"] == "ID001"

    def test_missing_source_or_filename_skipped(self, tmp_path: Path) -> None:
        bad_csv = "SourceFile,FileName,ID,Coder\n,vid1.mov,ID001,CoderA\n"
        t = _make_tracker(tmp_path, csv_content=bad_csv)
        assert t._read_video_list() == []

    def test_original_coder_fallback(self, tmp_path: Path) -> None:
        csv_content = "SourceFile,FileName,ID,Coder,Original_Coder\n"
        csv_content += "/path/to/vid.mov,vid.mov,ID001,,FallbackCoder\n"
        t = _make_tracker(tmp_path, csv_content=csv_content)
        videos = t._read_video_list()
        assert videos[0]["coder"] == "FallbackCoder"

    def test_returns_empty_on_bad_csv_path(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        t.csv_path = "/nonexistent/file.csv"
        assert t._read_video_list() == []


# ---------------------------------------------------------------------------
# _convert_path
# ---------------------------------------------------------------------------

class TestConvertPath:
    def test_standard_prefix_replaced(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        src = "/Volumes/T7 Shield/AMES_Phase_III/Phase_III_videos/sub/vid.mov"
        result = t._convert_path(src)
        assert result.startswith(t.base_video_dir)
        assert "sub/vid.mov" in result

    def test_rmm_mode_joins_directly(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, rmm=True)
        result = t._convert_path("relative/path/vid.mov")
        assert result == os.path.join(t.base_video_dir, "relative/path/vid.mov")

    def test_unknown_prefix_uses_last_two_parts(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, rmm=False)
        src = "/some/unknown/prefix/subdir/vid.mov"
        result = t._convert_path(src)
        assert "subdir/vid.mov" in result


# ---------------------------------------------------------------------------
# _create_output_paths
# ---------------------------------------------------------------------------

class TestCreateOutputPaths:
    def _video_info(self) -> dict:
        return {
            "filename": "vid1.mov",
            "video_id": "ID001",
            "coder": "CoderA",
            "source_file": "/path/vid1.mov",
            "row_data": "",
        }

    def test_tracking_path_contains_base_name(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=False)
        _, tracking = t._create_output_paths(self._video_info())
        assert "ID001_CoderA_vid1" in tracking

    def test_video_path_none_when_viz_disabled(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=False)
        vid, _ = t._create_output_paths(self._video_info())
        assert vid is None

    def test_video_path_set_when_viz_enabled(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=True)
        vid, _ = t._create_output_paths(self._video_info())
        assert vid is not None
        assert vid.endswith(".mp4")

    def test_tracking_dir_created(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=False)
        t._create_output_paths(self._video_info())
        assert os.path.isdir(os.path.join(t.output_dir, "tracking"))

    def test_video_dir_created_when_viz_enabled(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=True)
        t._create_output_paths(self._video_info())
        assert os.path.isdir(os.path.join(t.output_dir, "videos"))


# ---------------------------------------------------------------------------
# _process_single_video
# ---------------------------------------------------------------------------

class TestProcessSingleVideo:
    def _video_info(self, tmp_path: Path) -> dict:
        # Create a real source file so the existence check passes
        src = tmp_path / "videos" / "vid1.mov"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.touch()
        return {
            "filename": "vid1.mov",
            "video_id": "ID001",
            "coder": "CoderA",
            "source_file": str(src),
            "row_data": "",
        }

    def test_returns_false_when_source_missing(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, rmm=True)
        info = {
            "filename": "missing.mov",
            "video_id": "ID001",
            "coder": "CoderA",
            "source_file": "/nonexistent/missing.mov",
            "row_data": "",
        }
        result = t._process_single_video(info)
        assert result is False

    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, rmm=True, enable_visualization=False)
        mock_pipeline = MagicMock()
        _fake_pipeline_cls.return_value = mock_pipeline

        info = self._video_info(tmp_path)
        info["source_file"] = str(tmp_path / "videos" / "vid1.mov")

        result = t._process_single_video(info)
        assert result is True

    def test_skips_if_tracking_output_exists(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, rmm=True, enable_visualization=False)
        info = self._video_info(tmp_path)
        info["source_file"] = str(tmp_path / "videos" / "vid1.mov")

        _, tracking_path = t._create_output_paths(info)
        # Pre-create the tracking output to simulate already-processed
        Path(tracking_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tracking_path).touch()

        result = t._process_single_video(info)
        assert result is True
        _fake_pipeline_cls.assert_not_called()

    def test_cleans_up_partial_files_on_exception(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, rmm=True, enable_visualization=False)
        info = self._video_info(tmp_path)
        info["source_file"] = str(tmp_path / "videos" / "vid1.mov")

        _, tracking_path = t._create_output_paths(info)
        Path(tracking_path).parent.mkdir(parents=True, exist_ok=True)

        # Simulate pipeline writing a partial file mid-run then raising
        def _side_effect(src: str, vid: str) -> None:
            Path(tracking_path).touch()   # partial file created by pipeline
            raise RuntimeError("boom")

        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = _side_effect
        _fake_pipeline_cls.return_value = mock_pipeline

        result = t._process_single_video(info)
        assert result is False
        assert not os.path.exists(tracking_path)

    def test_returns_false_on_exception(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, rmm=True, enable_visualization=False)
        info = self._video_info(tmp_path)
        info["source_file"] = str(tmp_path / "videos" / "vid1.mov")

        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = RuntimeError("fail")
        _fake_pipeline_cls.return_value = mock_pipeline

        result = t._process_single_video(info)
        assert result is False


# ---------------------------------------------------------------------------
# process_all — row range and completed filtering
# ---------------------------------------------------------------------------

class TestProcessAll:
    def _tracker_with_mock_process(
        self, tmp_path: Path, **kwargs: Any
    ) -> tuple[BatchTracker, MagicMock]:
        t = _make_tracker(tmp_path, reuse_pipeline=False, enable_visualization=False, **kwargs)
        mock_process = MagicMock(return_value=True)
        t._process_single_video = mock_process  # type: ignore[method-assign]
        return t, mock_process

    def test_processes_all_videos_by_default(self, tmp_path: Path) -> None:
        t, mock_process = self._tracker_with_mock_process(tmp_path)
        t.process_all()
        assert mock_process.call_count == 3

    def test_skips_already_completed(self, tmp_path: Path) -> None:
        t, mock_process = self._tracker_with_mock_process(tmp_path)
        t.completed_videos = {"vid1.mov"}
        t.process_all()
        assert mock_process.call_count == 2

    def test_row_range_slices_video_list(self, tmp_path: Path) -> None:
        t, mock_process = self._tracker_with_mock_process(tmp_path, start_row=1, end_row=2)
        t.process_all()
        assert mock_process.call_count == 1

    def test_successful_video_added_to_completed(self, tmp_path: Path) -> None:
        t, _ = self._tracker_with_mock_process(tmp_path)
        t.process_all()
        assert len(t.completed_videos) == 3

    def test_failed_video_not_added_to_completed(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, enable_visualization=False)
        t._process_single_video = MagicMock(return_value=False)  # type: ignore[method-assign]
        t.process_all()
        assert len(t.completed_videos) == 0

    def test_interrupted_flag_stops_loop(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, reuse_pipeline=False, enable_visualization=False)
        call_count = 0

        def _side_effect(info: dict) -> bool:
            nonlocal call_count
            call_count += 1
            t.interrupted = True
            return True

        t._process_single_video = _side_effect  # type: ignore[method-assign]
        t.process_all()
        assert call_count == 1  # stops after first video sets interrupted=True

    def test_empty_csv_exits_early(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        t = _make_tracker(tmp_path, csv_content="SourceFile,FileName,ID,Coder\n")
        t._process_single_video = MagicMock()  # type: ignore[method-assign]
        t.process_all()
        assert "No videos found" in capsys.readouterr().out
        t._process_single_video.assert_not_called()

    def test_all_already_completed_exits_early(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        t, mock_process = self._tracker_with_mock_process(tmp_path)
        t.completed_videos = {"vid1.mov", "vid2.mov", "vid3.mov"}
        t.process_all()
        mock_process.assert_not_called()
        assert "All videos have already been processed" in capsys.readouterr().out

    def test_progress_saved_every_5_videos(self, tmp_path: Path) -> None:
        # Build a 6-row CSV so the mod-5 checkpoint fires once
        rows = CSV_HEADER
        for i in range(1, 7):
            rows += f"/vol/vid{i}.mov,vid{i}.mov,ID00{i},Coder{i}\n"
        t = _make_tracker(tmp_path, csv_content=rows, reuse_pipeline=False, enable_visualization=False)
        t._process_single_video = MagicMock(return_value=True)  # type: ignore[method-assign]
        save_spy = MagicMock(wraps=t._save_progress)
        t._save_progress = save_spy  # type: ignore[method-assign]
        t.process_all()
        # Called once at the 5th iteration checkpoint + once at the end
        assert save_spy.call_count >= 2


# ---------------------------------------------------------------------------
# _signal_handler
# ---------------------------------------------------------------------------

class TestSignalHandler:
    def test_sets_interrupted_flag(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        with pytest.raises(SystemExit):
            t._signal_handler(signal.SIGINT, None)
        assert t.interrupted is True

    def test_removes_current_video_from_completed(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        t.current_video = "vid1.mov"
        t.completed_videos = {"vid1.mov"}
        with pytest.raises(SystemExit):
            t._signal_handler(signal.SIGINT, None)
        assert "vid1.mov" not in t.completed_videos

    def test_progress_saved_on_interrupt(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        t.completed_videos = {"vid2.mov"}
        with pytest.raises(SystemExit):
            t._signal_handler(signal.SIGINT, None)
        with open(t.progress_file) as f:
            data = json.load(f)
        assert "vid2.mov" in data["completed_videos"]

    def test_partial_files_cleaned_on_interrupt(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path, enable_visualization=False)
        t.current_video = "vid1.mov"

        # Build the tracking path that _create_output_paths would produce
        video_info = {
            "filename": "vid1.mov",
            "video_id": "ID001",
            "coder": "CoderA",
            "source_file": "/path/vid1.mov",
            "row_data": "",
        }
        _, tracking_path = t._create_output_paths(video_info)
        Path(tracking_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tracking_path).touch()

        with pytest.raises(SystemExit):
            t._signal_handler(signal.SIGINT, None)

        assert not os.path.exists(tracking_path)