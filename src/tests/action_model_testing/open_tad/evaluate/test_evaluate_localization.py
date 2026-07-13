"""
Tests for src/sailsprep/action_model_testing/open_tad/evaluate/evaluate_localization.py

Same import pattern as test_evaluate.py: `evaluate_localization.py` does
`from common import iou_1d, discover_results` and runs report logic at
import time, so we import it from an empty tmp cwd with `evaluate/` on
sys.path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_EVAL_DIR = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "open_tad" / "evaluate"
)
_MODULE_PATH = _EVAL_DIR / "evaluate_localization.py"


def _load_module():
    # Other action_model_testing suites (video_swin, vlm_models, etc.) may
    # have already cached an unrelated `common` *package* under this name
    # during collection — purge it so `from common import ...` below
    # resolves fresh to open_tad/evaluate/common.py.
    sys.modules.pop("common", None)
    sys.path.insert(0, str(_EVAL_DIR))
    try:
        spec = importlib.util.spec_from_file_location("opentad_evaluate_localization", _MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(_EVAL_DIR))
        # `evaluate_localization.py` does `from common import iou_1d,
        # discover_results` — a plain top-level `common` module local to
        # open_tad/evaluate/. Purge it so it can't shadow the unrelated
        # `common` *packages* used by other action_model_testing suites.
        sys.modules.pop("common", None)


@pytest.fixture(scope="module")
def loc_mod(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("opentad_evaluate_localization_import")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return _load_module()
    finally:
        os.chdir(cwd)


class TestLoadGtAndPred:
    def test_load_gt_filters_test_subset_with_annotations(self, loc_mod, tmp_path):
        db = {
            "vid1": {"subset": "test", "annotations": [{"segment": [0.0, 5.0]}]},
            "vid2": {"subset": "test", "annotations": []},
            "vid3": {"subset": "train", "annotations": [{"segment": [1.0, 2.0]}]},
        }
        p = tmp_path / "anno.json"
        p.write_text(json.dumps({"database": db}))
        gt = loc_mod.load_gt(str(p))
        assert list(gt.keys()) == ["vid1"]

    def test_load_pred_sorted_by_score_desc(self, loc_mod, tmp_path):
        p = tmp_path / "pred.json"
        p.write_text(json.dumps({
            "results": {
                "vid1": [
                    {"score": 0.2, "segment": [0.0, 1.0]},
                    {"score": 0.9, "segment": [1.0, 2.0]},
                ]
            }
        }))
        pred = loc_mod.load_pred(str(p))
        scores = [s for s, _ in pred["vid1"]]
        assert scores == sorted(scores, reverse=True)


class TestComputeRecall:
    def test_full_recall(self, loc_mod):
        gt = {"vid1": np.array([[0.0, 10.0]], dtype=np.float32)}
        pred = {"vid1": [(1.0, [0.0, 10.0])]}
        assert loc_mod.compute_recall(pred, gt, 0.5) == pytest.approx(1.0)

    def test_zero_recall_when_no_predictions(self, loc_mod):
        gt = {"vid1": np.array([[0.0, 10.0]], dtype=np.float32)}
        pred = {}
        assert loc_mod.compute_recall(pred, gt, 0.5) == pytest.approx(0.0)


class TestAgnosticMap:
    def test_perfect_prediction_gives_map_one(self, loc_mod):
        gt = {"vid1": np.array([[0.0, 10.0]], dtype=np.float32)}
        pred = {"vid1": [(1.0, [0.0, 10.0])]}
        assert loc_mod.agnostic_map(pred, gt, 0.5) == pytest.approx(1.0)

    def test_no_gt_returns_zero(self, loc_mod):
        gt = {}
        pred = {"vid1": [(1.0, [0.0, 10.0])]}
        assert loc_mod.agnostic_map(pred, gt, 0.5) == 0.0


class TestFmtSpread:
    def test_mean_and_per_seed_values(self, loc_mod):
        summary, per_seed = loc_mod.fmt_spread([40.0, 60.0])
        assert "50.00" in summary
        assert "40.00%" in per_seed and "60.00%" in per_seed
