"""
Tests for src/sailsprep/action_model_testing/open_tad/evaluate/common.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "open_tad" / "evaluate" / "common.py"
)


def _load_common_module():
    spec = importlib.util.spec_from_file_location("opentad_eval_common", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def common_mod():
    return _load_common_module()


class TestIou1d:
    def test_full_overlap(self, common_mod):
        pred = np.array([0.0, 10.0])
        gt = np.array([[0.0, 10.0]])
        assert common_mod.iou_1d(pred, gt)[0] == pytest.approx(1.0)

    def test_no_overlap(self, common_mod):
        pred = np.array([0.0, 5.0])
        gt = np.array([[10.0, 15.0]])
        assert common_mod.iou_1d(pred, gt)[0] == pytest.approx(0.0)

    def test_partial_overlap(self, common_mod):
        pred = np.array([0.0, 10.0])
        gt = np.array([[5.0, 15.0]])
        # inter = 5, union = 10 + 10 - 5 = 15
        assert common_mod.iou_1d(pred, gt)[0] == pytest.approx(5.0 / 15.0)

    def test_multiple_gt_segments(self, common_mod):
        pred = np.array([0.0, 10.0])
        gt = np.array([[0.0, 10.0], [20.0, 30.0]])
        ious = common_mod.iou_1d(pred, gt)
        assert ious.shape == (2,)
        assert ious[0] == pytest.approx(1.0)
        assert ious[1] == pytest.approx(0.0)


class TestDiscoverResults:
    def test_no_matches_returns_empty(self, common_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        groups = common_mod.discover_results()
        assert groups == {}

    def test_groups_by_task_and_model(self, common_mod, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for seed in ("seed_42", "seed_123"):
            d = tmp_path / "exps" / "locomotion" / "actionformer_i3d" / seed / "eval"
            d.mkdir(parents=True)
            (d / "result_detection.json").write_text(json.dumps({}))

        groups = common_mod.discover_results()
        assert ("locomotion", "actionformer_i3d") in groups
        assert len(groups[("locomotion", "actionformer_i3d")]) == 2
