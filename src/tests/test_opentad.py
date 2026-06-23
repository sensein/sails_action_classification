# tests/test_opentad.py
"""
Combined pytest suite for:
  - end_to_end_custom.py  (PrepareVideoInfoAbsPath, LoadFramesAt15fps)
  - run_locomotion.py     (config generation, seed helpers, aggregation)

Run with:
    poetry run pytest tests/test_opentad.py -v
"""
from __future__ import annotations

import json
import os
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without a full GPU env
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_mmaction():
    """Minimal stubs for mmaction2 / mmcv registry used by end_to_end_custom."""
    mmaction = _make_stub("mmaction")
    datasets = _make_stub("mmaction.datasets")
    pipelines_mod = _make_stub("mmaction.datasets.pipelines")

    # Fake PIPELINES registry: just stores registered classes
    class _FakePipelines:
        _registry: dict = {}

        def register_module(self):
            def decorator(cls):
                self._registry[cls.__name__] = cls
                return cls
            return decorator

    PIPELINES = _FakePipelines()
    pipelines_mod.PIPELINES = PIPELINES

    builder_mod = _make_stub("mmaction.datasets.builder")
    builder_mod.PIPELINES = PIPELINES

    # Make "from ..builder import PIPELINES" work inside end_to_end_custom
    # by patching the package hierarchy that the relative import resolves to.
    # We expose it on the sailsprep package stub too.
    return PIPELINES


PIPELINES = _stub_mmaction()

# ---------------------------------------------------------------------------
# Dynamic import helpers
# ---------------------------------------------------------------------------

def _import_transforms():
    pkg_name = "sailsprep.action_model_testing.OpenTAD"
    for part in ["sailsprep", "sailsprep.action_model_testing", pkg_name]:
        if part not in sys.modules:
            sys.modules[part] = types.ModuleType(part)

    builder_stub = types.ModuleType(f"{pkg_name}.builder")
    builder_stub.PIPELINES = PIPELINES
    sys.modules[f"{pkg_name}.builder"] = builder_stub

    # __file__ = src/tests/test_opentad.py
    # .parent.parent = src/
    src = Path(__file__).parent.parent / "sailsprep/action_model_testing/OpenTAD/end_to_end_custom.py"
    assert src.exists(), f"Cannot find end_to_end_custom.py at: {src}"

    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.end_to_end_custom", src,
        submodule_search_locations=[]
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_run_locomotion():
    src = Path(__file__).parent.parent / "sailsprep/action_model_testing/OpenTAD/run_locomotion.py"
    assert src.exists(), f"Cannot find run_locomotion.py at: {src}"

    spec = importlib.util.spec_from_file_location("run_locomotion", src)
    mod = importlib.util.module_from_spec(spec)
    with patch("os.path.exists", return_value=False):
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def transforms_mod():
    return _import_transforms()


@pytest.fixture(scope="module")
def loco_mod():
    return _import_run_locomotion()


@pytest.fixture()
def tmp_video_map(tmp_path):
    """Write a tiny video_path_map JSON and return its path + contents."""
    mapping = {
        "vid_001": str(tmp_path / "vid_001.mp4"),
        "vid_002": str(tmp_path / "vid_002.mp4"),
    }
    map_file = tmp_path / "path_map.json"
    map_file.write_text(json.dumps(mapping))
    return map_file, mapping


@pytest.fixture()
def basic_results():
    """Minimal results dict for LoadFramesAt15fps tests."""
    return {
        "total_frames": 300,        # 10 s at 30 fps
        "gt_segments": np.array([[0.0, 5.0], [6.0, 9.0]], dtype=np.float32),
        "gt_labels":   np.array([0, 1]),
    }


# ===========================================================================
# Tests — PrepareVideoInfoAbsPath
# ===========================================================================

class TestPrepareVideoInfoAbsPath:

    def test_init_raises_if_map_missing(self, transforms_mod):
        with pytest.raises(AssertionError, match="video_path_map not found"):
            transforms_mod.PrepareVideoInfoAbsPath(
                video_path_map="/nonexistent/path.json"
            )

    def test_init_loads_map(self, transforms_mod, tmp_video_map):
        map_file, mapping = tmp_video_map
        t = transforms_mod.PrepareVideoInfoAbsPath(str(map_file))
        assert t.path_map == mapping
        assert t.modality == "RGB"

    def test_call_sets_filename_and_modality(self, transforms_mod, tmp_video_map):
        map_file, mapping = tmp_video_map
        t = transforms_mod.PrepareVideoInfoAbsPath(str(map_file))
        results = {"video_name": "vid_001"}
        out = t(results)
        assert out["filename"] == mapping["vid_001"]
        assert out["modality"] == "RGB"

    def test_call_raises_for_unknown_video(self, transforms_mod, tmp_video_map):
        map_file, _ = tmp_video_map
        t = transforms_mod.PrepareVideoInfoAbsPath(str(map_file))
        with pytest.raises(AssertionError, match="not found in video_path_map"):
            t({"video_name": "unknown_vid"})

    def test_custom_modality(self, transforms_mod, tmp_video_map):
        map_file, _ = tmp_video_map
        t = transforms_mod.PrepareVideoInfoAbsPath(str(map_file), modality="Flow")
        out = t({"video_name": "vid_001"})
        assert out["modality"] == "Flow"

    def test_video_name_cast_to_str(self, transforms_mod, tmp_video_map):
        """video_name may arrive as an int-like; must be cast to str."""
        map_file, mapping = tmp_video_map
        # Add an int-keyed entry
        mapping["42"] = "/some/path.mp4"
        map_file.write_text(json.dumps(mapping))
        t = transforms_mod.PrepareVideoInfoAbsPath(str(map_file))
        out = t({"video_name": 42})
        assert out["filename"] == "/some/path.mp4"


# ===========================================================================
# Tests — LoadFramesAt15fps
# ===========================================================================

class TestLoadFramesAt15fps:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_init_default(self, transforms_mod):
        t = transforms_mod.LoadFramesAt15fps(clip_len=16)
        assert t.frame_stride == 2       # 30 / 15
        assert t.source_fps == 30.0
        assert t.target_fps == 15.0

    def test_init_bad_fps_ratio(self, transforms_mod):
        with pytest.raises(AssertionError):
            transforms_mod.LoadFramesAt15fps(clip_len=16, source_fps=25, target_fps=15)

    def test_init_custom_fps(self, transforms_mod):
        t = transforms_mod.LoadFramesAt15fps(clip_len=16, source_fps=60, target_fps=15)
        assert t.frame_stride == 4

    # ------------------------------------------------------------------
    # random_trunc method
    # ------------------------------------------------------------------

    def _make_trunc(self, transforms_mod, trunc_len=50, trunc_thresh=0.5):
        return transforms_mod.LoadFramesAt15fps(
            clip_len=16,
            method="random_trunc",
            trunc_len=trunc_len,
            trunc_thresh=trunc_thresh,
            crop_ratio=[0.9, 1.0],
        )

    def test_random_trunc_output_shape(self, transforms_mod, basic_results):
        t = self._make_trunc(transforms_mod, trunc_len=50)
        out = t(basic_results.copy())
        assert len(out["frame_inds"]) == 50
        assert out["masks"].shape == (50,)
        assert out["masks"].dtype == torch.bool

    def test_random_trunc_masks_all_true_long_video(self, transforms_mod):
        """Video longer than trunc_len → all masks should be True."""
        t = self._make_trunc(transforms_mod, trunc_len=50)
        results = {
            "total_frames": 600,           # 300 snippets at 15fps
            "gt_segments": np.array([[0.0, 100.0]], dtype=np.float32),
            "gt_labels":   np.array([0]),
        }
        out = t(results)
        assert out["masks"].all()

    def test_sliding_window_clamps_start_idx(self, transforms_mod):
        """feature_end_idx past video end gets clamped; result is padded to window_size."""
        t = self._make_sliding(transforms_mod)
        results = {
            "total_frames": 120,      
            "window_size": 30,
            "feature_start_idx": 10,  
            "feature_end_idx": 20,   
        }
        out = t(results)
        assert len(out["frame_inds"]) == 30
        assert out["masks"][0].item()
        assert not out["masks"][-1].item()

    def test_random_trunc_frame_inds_in_range(self, transforms_mod, basic_results):
        t = self._make_trunc(transforms_mod, trunc_len=50)
        out = t(basic_results.copy())
        assert out["frame_inds"].min() >= 0
        assert out["frame_inds"].max() < basic_results["total_frames"]

    def test_random_trunc_gt_segments_updated(self, transforms_mod):
        t = self._make_trunc(transforms_mod, trunc_len=50, trunc_thresh=0.0)
        results = {
            "total_frames": 300,
            "gt_segments": np.array([[0.0, 40.0]], dtype=np.float32),
            "gt_labels":   np.array([2]),
        }
        out = t(results)
        # gt_segments must be shifted relative to window start
        assert out["gt_segments"].ndim == 2
        assert out["gt_labels"].shape[0] == out["gt_segments"].shape[0]

    def test_effective_fps_set(self, transforms_mod, basic_results):
        t = self._make_trunc(transforms_mod, trunc_len=50)
        out = t(basic_results.copy())
        assert out["effective_fps"] == 15.0

    def test_num_clips_and_clip_len(self, transforms_mod, basic_results):
        t = self._make_trunc(transforms_mod, trunc_len=50)
        out = t(basic_results.copy())
        assert out["num_clips"] == 1
        assert out["clip_len"] == 50

    # ------------------------------------------------------------------
    # sliding_window method
    # ------------------------------------------------------------------

    def _make_sliding(self, transforms_mod, window_size=30):
        return transforms_mod.LoadFramesAt15fps(
            clip_len=16,
            method="sliding_window",
        )
    def test_sliding_window_basic(self, transforms_mod):
        t = self._make_sliding(transforms_mod)
        results = {
            "total_frames": 300,
            "window_size": 30,
            "feature_start_idx": 0,
            "feature_end_idx": 29,
        }
        out = t(results)
        assert len(out["frame_inds"]) == 30
        assert out["masks"].all()

    def test_sliding_window_past_end_pads(self, transforms_mod):
        t = self._make_sliding(transforms_mod)
        results = {
            "total_frames": 20,           # only 10 snippets at 15fps
            "window_size": 30,
            "feature_start_idx": 0,
            "feature_end_idx": 29,
        }
        out = t(results)
        assert len(out["frame_inds"]) == 30
        assert out["masks"][:10].all()
        assert not out["masks"][10:].any()


    # ------------------------------------------------------------------
    # Invalid method
    # ------------------------------------------------------------------

    def test_invalid_method_raises(self, transforms_mod):
        t = transforms_mod.LoadFramesAt15fps(clip_len=16, method="unknown")
        with pytest.raises(ValueError, match="unsupported method"):
            t({"total_frames": 100})

    def test_missing_total_frames_raises(self, transforms_mod):
        t = transforms_mod.LoadFramesAt15fps(
            clip_len=16, method="random_trunc", trunc_len=50, trunc_thresh=0.5
        )
        with pytest.raises(AssertionError, match="total_frames must be in results"):
            t({"gt_segments": np.array([[0.0, 1.0]]), "gt_labels": np.array([0])})


# ===========================================================================
# Tests — run_locomotion helpers
# ===========================================================================

class TestRunLocomotionHelpers:

    def test_task_config_keys(self, loco_mod):
        for task in ("locomotion", "rmm"):
            cfg = loco_mod.TASK_CONFIG[task]
            assert "num_classes" in cfg
            assert "ann_file" in cfg
            assert "class_map" in cfg

    def test_backbone_config_keys(self, loco_mod):
        for bb in ("vjepa", "i3d", "r2plus1d", "pose"):
            cfg = loco_mod.BACKBONE_CONFIG[bb]
            assert "dim" in cfg
            assert "feat_dir" in cfg
            assert cfg["dim"] > 0

    def test_set_global_seed_reproducible(self, loco_mod):
        loco_mod.set_global_seed(42)
        a = np.random.randint(0, 1000)
        loco_mod.set_global_seed(42)
        b = np.random.randint(0, 1000)
        assert a == b

    def test_aggregate_no_results(self, loco_mod, capsys):
        """aggregate_seed_results prints a sensible message when nothing found."""
        loco_mod.aggregate_seed_results("actionformer", "i3d", "locomotion", [42])
        captured = capsys.readouterr()
        assert "No results found" in captured.out or "seed" in captured.out


class TestGenerateConfig:

    def test_generates_files(self, loco_mod, tmp_path, monkeypatch):
        """generate_config writes two files and returns the model config path."""
        monkeypatch.chdir(tmp_path)
        cfg_path = loco_mod.generate_config("actionformer", "i3d", "locomotion", seed=42)
        assert os.path.exists(cfg_path)
        # dataset config should also exist
        ds_path = tmp_path / "configs/_base_/datasets/locomotion/features_i3d_pad.py"
        assert ds_path.exists()

    def test_config_contains_seed(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = loco_mod.generate_config("tridet", "pose", "rmm", seed=123)
        content = Path(cfg_path).read_text()
        assert "random_seed = 123" in content

    def test_config_contains_num_classes(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = loco_mod.generate_config("dyfadet", "vjepa", "locomotion", seed=42)
        content = Path(cfg_path).read_text()
        assert "num_classes=5" in content

    def test_config_contains_feat_dim(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        expected_dim = loco_mod.BACKBONE_CONFIG["i3d"]["dim"]
        cfg_path = loco_mod.generate_config("actionformer", "i3d", "locomotion", seed=42)
        content = Path(cfg_path).read_text()
        assert f"in_channels={expected_dim}" in content

    def test_all_model_backbone_combos(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for model in loco_mod.ALL_MODELS:
            for bb in loco_mod.ALL_BACKBONES:
                path = loco_mod.generate_config(model, bb, "locomotion", seed=42)
                assert os.path.exists(path), f"Missing config for {model}/{bb}"


class TestFindBestCheckpoint:

    def test_returns_none_if_no_workdir(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = loco_mod.find_best_checkpoint("actionformer", "i3d", "locomotion", 42)
        assert result is None

    def test_finds_checkpoint(self, loco_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt_dir = tmp_path / "exps/locomotion/actionformer_i3d/seed_42/run1/checkpoint"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best.pth"
        ckpt.write_text("dummy")
        result = loco_mod.find_best_checkpoint("actionformer", "i3d", "locomotion", 42)
        assert os.path.abspath(result) == str(ckpt)


class TestAggregateResults:

    def _write_result(self, path: Path, map_val: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mAP": map_val}))

    def test_two_seeds_produce_summary(self, loco_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        for seed, val in [(42, 0.45), (123, 0.55)]:
            p = tmp_path / f"exps/locomotion/actionformer_i3d/seed_{seed}/eval/test_results.json"
            self._write_result(p, val)

        loco_mod.aggregate_seed_results("actionformer", "i3d", "locomotion", [42, 123])

        summary = tmp_path / "exps/locomotion/actionformer_i3d/seed_summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert abs(data["mean_mAP"] - 0.50) < 1e-6
        assert "ci95_lower" in data
        assert "ci95_upper" in data

    def test_one_seed_no_summary_file(self, loco_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "exps/locomotion/tridet_pose/seed_42/eval/test_results.json"
        self._write_result(p, 0.60)

        loco_mod.aggregate_seed_results("tridet", "pose", "locomotion", [42])
        out = capsys.readouterr().out
        assert "Only 1 seed" in out

    def test_alternative_map_keys(self, loco_mod, tmp_path, monkeypatch):
        """Aggregator should parse 'map', 'average_mAP', 'mAP@0.5' too."""
        monkeypatch.chdir(tmp_path)
        for seed, key, val in [(42, "map", 0.4), (123, "average_mAP", 0.6)]:
            p = tmp_path / f"exps/locomotion/dyfadet_r2plus1d/seed_{seed}/eval/test_results.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({key: val}))

        loco_mod.aggregate_seed_results("dyfadet", "r2plus1d", "locomotion", [42, 123])
        summary = tmp_path / "exps/locomotion/dyfadet_r2plus1d/seed_summary.json"
        assert summary.exists()