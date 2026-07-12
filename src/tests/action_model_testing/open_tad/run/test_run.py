"""
Tests for src/sailsprep/action_model_testing/open_tad/run/run.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "open_tad" / "run" / "run.py"
)


def _load_run_module():
    spec = importlib.util.spec_from_file_location("opentad_run", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_mod():
    return _load_run_module()


class TestConfigs:
    def test_task_config_keys(self, run_mod):
        for task in ("locomotion", "rmm"):
            cfg = run_mod.TASK_CONFIG[task]
            assert "num_classes" in cfg
            assert "ann_file" in cfg
            assert "class_map" in cfg

    def test_backbone_config_keys(self, run_mod):
        for bb in ("vjepa", "i3d", "r2plus1d", "pose"):
            cfg = run_mod.BACKBONE_CONFIG[bb]
            assert "dim" in cfg
            assert "feat_dir" in cfg
            assert cfg["dim"] > 0

    def test_all_models_and_backbones(self, run_mod):
        assert run_mod.ALL_MODELS == ["actionformer", "tridet", "dyfadet"]
        assert set(run_mod.ALL_BACKBONES) == {"vjepa", "i3d", "r2plus1d", "pose"}


class TestSetGlobalSeed:
    def test_reproducible(self, run_mod):
        run_mod.set_global_seed(42)
        a = np.random.randint(0, 1000)
        run_mod.set_global_seed(42)
        b = np.random.randint(0, 1000)
        assert a == b


class TestGenerateConfig:
    def test_generates_files(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = run_mod.generate_config("actionformer", "i3d", "locomotion", seed=42)
        assert os.path.exists(cfg_path)
        ds_path = tmp_path / "configs/_base_/datasets/locomotion/features_i3d_pad.py"
        assert ds_path.exists()

    def test_config_contains_seed(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = run_mod.generate_config("tridet", "pose", "rmm", seed=123)
        content = Path(cfg_path).read_text()
        assert "seed=123" in content or "123" in content

    def test_config_contains_num_classes(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = run_mod.generate_config("dyfadet", "vjepa", "locomotion", seed=42)
        content = Path(cfg_path).read_text()
        assert "num_classes=5" in content

    def test_config_contains_feat_dim(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        expected_dim = run_mod.BACKBONE_CONFIG["i3d"]["dim"]
        cfg_path = run_mod.generate_config("actionformer", "i3d", "locomotion", seed=42)
        content = Path(cfg_path).read_text()
        assert f"{expected_dim}" in content

    def test_all_model_backbone_combos(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for model in run_mod.ALL_MODELS:
            for bb in run_mod.ALL_BACKBONES:
                path = run_mod.generate_config(model, bb, "locomotion", seed=42)
                assert os.path.exists(path), f"Missing config for {model}/{bb}"


class TestFindBestCheckpoint:
    def test_returns_none_if_no_workdir(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_mod.find_best_checkpoint("actionformer", "i3d", "locomotion", 42)
        assert result is None

    def test_finds_checkpoint(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt_dir = tmp_path / "exps/locomotion/actionformer_i3d/seed_42/run1/checkpoint"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best.pth"
        ckpt.write_text("dummy")
        result = run_mod.find_best_checkpoint("actionformer", "i3d", "locomotion", 42)
        assert os.path.abspath(result) == str(ckpt)


class TestAggregateSeedResults:
    def _write_result(self, path: Path, map_val: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mAP": map_val}))

    def test_two_seeds_produce_summary(self, run_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for seed, val in [(42, 0.45), (123, 0.55)]:
            p = tmp_path / f"exps/locomotion/actionformer_i3d/seed_{seed}/eval/test_results.json"
            self._write_result(p, val)

        run_mod.aggregate_seed_results("actionformer", "i3d", "locomotion", [42, 123])

        summary = tmp_path / "exps/locomotion/actionformer_i3d/seed_summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert abs(data["mean_mAP"] - 0.50) < 1e-6
        assert "ci95_lower" in data
        assert "ci95_upper" in data

    def test_one_seed_no_summary_file(self, run_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "exps/locomotion/tridet_pose/seed_42/eval/test_results.json"
        self._write_result(p, 0.60)

        run_mod.aggregate_seed_results("tridet", "pose", "locomotion", [42])
        out = capsys.readouterr().out
        assert "Only 1 seed" in out

    def test_alternative_map_keys(self, run_mod, tmp_path, monkeypatch):
        """Aggregator should parse 'map', 'average_mAP', 'mAP@0.5' too."""
        monkeypatch.chdir(tmp_path)
        for seed, key, val in [(42, "map", 0.4), (123, "average_mAP", 0.6)]:
            p = tmp_path / f"exps/locomotion/dyfadet_r2plus1d/seed_{seed}/eval/test_results.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({key: val}))

        run_mod.aggregate_seed_results("dyfadet", "r2plus1d", "locomotion", [42, 123])
        summary = tmp_path / "exps/locomotion/dyfadet_r2plus1d/seed_summary.json"
        assert summary.exists()

    def test_no_results_found_message(self, run_mod, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        run_mod.aggregate_seed_results("actionformer", "i3d", "locomotion", [42])
        captured = capsys.readouterr()
        assert "no test_results.json found" in captured.out
