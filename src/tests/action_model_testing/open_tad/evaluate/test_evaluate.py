"""
Tests for src/sailsprep/action_model_testing/open_tad/evaluate/evaluate.py

`evaluate.py` is a top-level script: it does `from common import iou_1d,
discover_results` (sibling import) and runs its aggregation/report logic at
import time. We add the `evaluate/` dir to sys.path so the sibling import
resolves, and import it inside an empty tmp cwd so `discover_results()`
finds nothing and the script body is a no-op beyond writing empty report
files.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_EVAL_DIR = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "open_tad" / "evaluate"
)
_MODULE_PATH = _EVAL_DIR / "evaluate.py"


def _load_evaluate_module(tmp_path):
    # Other action_model_testing suites (video_swin, vlm_models, etc.) may
    # have already cached an unrelated `common` *package* under this name
    # during collection — purge it so `from common import ...` below
    # resolves fresh to open_tad/evaluate/common.py.
    sys.modules.pop("common", None)
    sys.path.insert(0, str(_EVAL_DIR))
    try:
        spec = importlib.util.spec_from_file_location("opentad_evaluate", _MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(_EVAL_DIR))
        # `evaluate.py` does `from common import iou_1d, discover_results` — a
        # plain top-level `common` module local to open_tad/evaluate/. Purge it
        # so it can't shadow the unrelated `common` *packages* used by other
        # action_model_testing suites (video_swin, vlm_models, etc).
        sys.modules.pop("common", None)


@pytest.fixture(scope="module")
def eval_mod(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("opentad_evaluate_import")
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return _load_evaluate_module(tmp_path)
    finally:
        os.chdir(cwd)


class TestLoadClassMap:
    def test_reads_nonempty_lines(self, eval_mod, tmp_path):
        p = tmp_path / "classes.txt"
        p.write_text("Walking\nCruising\n\nCrawling\n")
        assert eval_mod.load_class_map(str(p)) == ["Walking", "Cruising", "Crawling"]


class TestLoadGtAndPred:
    def _anno(self, tmp_path):
        db = {
            "vid1": {
                "subset": "test",
                "annotations": [
                    {"segment": [0.0, 5.0], "label": "Walking"},
                    {"segment": [6.0, 8.0], "label": "Unknown"},
                ],
            },
            "vid2": {"subset": "train", "annotations": []},
        }
        p = tmp_path / "anno.json"
        p.write_text(json.dumps({"database": db}))
        return p

    def test_load_gt_filters_test_subset_and_unknown_labels(self, eval_mod, tmp_path):
        anno = self._anno(tmp_path)
        gt = eval_mod.load_gt(str(anno), ["Walking", "Cruising"])
        assert "vid1" in gt and "vid2" not in gt
        assert len(gt["vid1"]) == 1
        assert gt["vid1"][0]["label_id"] == 0

    def test_load_pred_filters_unknown_labels(self, eval_mod, tmp_path):
        p = tmp_path / "pred.json"
        p.write_text(json.dumps({
            "results": {
                "vid1": [
                    {"segment": [0.0, 5.0], "label": "Walking", "score": 0.9},
                    {"segment": [1.0, 2.0], "label": "Unknown", "score": 0.5},
                ]
            }
        }))
        pred = eval_mod.load_pred(str(p), ["Walking", "Cruising"])
        assert len(pred["vid1"]) == 1
        assert pred["vid1"][0]["label_id"] == 0


class TestComputeMapPerThresh:
    def test_perfect_prediction_gives_map_one(self, eval_mod):
        gt = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0}]}
        pred = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0, "score": 1.0}]}
        aps = eval_mod.compute_map_per_thresh(pred, gt, [0.5], num_classes=1)
        assert aps == pytest.approx([1.0])

    def test_no_predictions_gives_zero_map(self, eval_mod):
        gt = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0}]}
        pred = {"vid1": []}
        aps = eval_mod.compute_map_per_thresh(pred, gt, [0.5], num_classes=1)
        assert aps == pytest.approx([0.0])


class TestComputeRecallPerThresh:
    def test_full_recall(self, eval_mod):
        gt = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0}]}
        pred = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0, "score": 1.0}]}
        recalls = eval_mod.compute_recall_per_thresh(pred, gt, [0.5], num_classes=1)
        assert recalls == pytest.approx([1.0])

    def test_zero_recall_no_predictions(self, eval_mod):
        gt = {"vid1": [{"segment": [0.0, 10.0], "label_id": 0}]}
        pred = {"vid1": []}
        recalls = eval_mod.compute_recall_per_thresh(pred, gt, [0.5], num_classes=1)
        assert recalls == pytest.approx([0.0])


class TestFmtSpread:
    def test_three_seeds_includes_ci(self, eval_mod):
        per_seed = {"seed_42": 48.0, "seed_123": 52.0, "seed_456": 50.0}
        s, ci, ps = eval_mod.fmt_spread(50.0, 5.0, 2.5, 3, per_seed)
        assert "50.00" in s
        assert ci == "[47.50, 52.50]"

    def test_single_seed_ci_is_na(self, eval_mod):
        s, ci, ps = eval_mod.fmt_spread(50.0, 0.0, 0.0, 1, {"seed_42": 50.0})
        assert "1 seed" in s
        assert ci == "N/A"
