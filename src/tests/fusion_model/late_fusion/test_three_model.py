"""
Tests for src/sailsprep/fusion_model/late_fusion/three_model.py

Run with:
    poetry run pytest src/tests/tests_three_model.py
"""

import json
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from sailsprep.fusion_model.late_fusion.three_model import (
    LABEL_MAPS,
    SEEDS,
    build_unified_labels,
    find_matching_windows,
    fmt,
    fuse_one_video,
    vjepa_to_frame_scores,
    windows_to_frame_scores,
)


# ===========================================================================
# Fixtures
# ===========================================================================

LOCO_LABEL_MAP: dict[int, str] = {
    0: "Crawling", 1: "Cruising", 2: "None",
    3: "Running",  4: "Vehicle",  5: "Walking",
}
RMM_LABEL_MAP: dict[int, str] = {
    0: "Hands_flapping", 1: "Jumping", 2: "None",
    3: "Rocking",        4: "Spinning",
}

ALL_LOCO_LABELS = sorted(LOCO_LABEL_MAP.values())
ALL_RMM_LABELS  = sorted(RMM_LABEL_MAP.values())


def _make_vjepa_df(
    labels: list[str],
    confidences: list[float] | None = None,
    true_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Helper: build a minimal V-JEPA per-frame DataFrame."""
    T = len(labels)
    if confidences is None:
        confidences = [0.9] * T
    if true_labels is None:
        true_labels = labels
    return pd.DataFrame({
        "frame":           list(range(T)),
        "predicted_label": labels,
        "true_label":      true_labels,
        "confidence":      confidences,
    })


def _make_windows(
    starts_ends_scores: list[tuple[int, int, list[float]]],
) -> list[dict[str, Any]]:
    return [
        {"start": s, "end": e, "scores": np.array(sc, dtype=np.float64)}
        for s, e, sc in starts_ends_scores
    ]


# ===========================================================================
# 1. build_unified_labels
# ===========================================================================

class TestBuildUnifiedLabels:

    def _vjepa_map_from_label_map(self, label_map: dict[int, str]) -> dict[str, Any]:
        """Simulate the vjepa_label_map JSON structure (str → int index)."""
        return {lbl: i for i, lbl in sorted(label_map.items())}

    def test_locomotion_labels_match(self):
        vjepa_map = self._vjepa_map_from_label_map(LOCO_LABEL_MAP)
        all_labels, vs2u, ps2u, po2u = build_unified_labels(
            vjepa_map, LOCO_LABEL_MAP, LOCO_LABEL_MAP
        )
        assert set(all_labels) == set(LOCO_LABEL_MAP.values())

    def test_unified_index_is_consistent(self):
        vjepa_map = self._vjepa_map_from_label_map(LOCO_LABEL_MAP)
        all_labels, vs2u, ps2u, po2u = build_unified_labels(
            vjepa_map, LOCO_LABEL_MAP, LOCO_LABEL_MAP
        )
        # Every pyskl int maps to the correct unified index
        for pyskl_idx, lbl in LOCO_LABEL_MAP.items():
            assert ps2u[pyskl_idx] == all_labels.index(lbl)

    def test_vjepa_str_to_unified_correct(self):
        vjepa_map = self._vjepa_map_from_label_map(LOCO_LABEL_MAP)
        all_labels, vs2u, _, _ = build_unified_labels(
            vjepa_map, LOCO_LABEL_MAP, LOCO_LABEL_MAP
        )
        for lbl in LOCO_LABEL_MAP.values():
            assert lbl in vs2u
            assert vs2u[lbl] == all_labels.index(lbl)

    def test_disjoint_label_sets_are_merged(self):
        vjepa_map   = {"A": 0, "B": 1}
        pyskl_map   = {0: "C", 1: "D"}
        posec3d_map = {0: "E"}
        all_labels, _, _, _ = build_unified_labels(vjepa_map, pyskl_map, posec3d_map)
        assert set(all_labels) == {"A", "B", "C", "D", "E"}

    def test_returns_sorted_labels(self):
        vjepa_map = self._vjepa_map_from_label_map(LOCO_LABEL_MAP)
        all_labels, _, _, _ = build_unified_labels(
            vjepa_map, LOCO_LABEL_MAP, LOCO_LABEL_MAP
        )
        assert all_labels == sorted(all_labels)


# ===========================================================================
# 2. windows_to_frame_scores
# ===========================================================================

class TestWindowsToFrameScores:

    def test_empty_windows(self):
        scores, counts = windows_to_frame_scores([], T=5, num_classes=3)
        assert scores.shape == (5, 3)
        assert counts.sum() == 0
        assert scores.sum() == 0

    def test_single_window_covers_all_frames(self):
        raw_scores = [0.1, 0.7, 0.2]
        windows = _make_windows([(0, 4, raw_scores)])
        scores, counts = windows_to_frame_scores(windows, T=5, num_classes=3)
        assert counts.tolist() == [1, 1, 1, 1, 1]
        np.testing.assert_allclose(scores[0], raw_scores)

    def test_overlapping_windows_are_averaged(self):
        # Two windows both covering frame 0
        w = _make_windows([
            (0, 0, [0.0, 1.0, 0.0]),
            (0, 0, [1.0, 0.0, 0.0]),
        ])
        scores, counts = windows_to_frame_scores(w, T=1, num_classes=3)
        assert counts[0] == 2
        np.testing.assert_allclose(scores[0], [0.5, 0.5, 0.0])

    def test_window_clipped_to_T(self):
        """A window that extends past T should not cause index errors."""
        w = _make_windows([(0, 100, [1.0, 0.0])])
        scores, counts = windows_to_frame_scores(w, T=5, num_classes=2)
        assert scores.shape == (5, 2)
        assert counts.tolist() == [1, 1, 1, 1, 1]

    def test_non_overlapping_windows(self):
        w = _make_windows([
            (0, 1, [1.0, 0.0]),
            (2, 3, [0.0, 1.0]),
        ])
        scores, counts = windows_to_frame_scores(w, T=4, num_classes=2)
        assert counts[0] == 1
        assert counts[2] == 1
        assert counts[3] == 1


# ===========================================================================
# 3. vjepa_to_frame_scores
# ===========================================================================

class TestVjepaToFrameScores:

    def _setup(self):
        all_labels = ALL_LOCO_LABELS
        vs2u = {lbl: i for i, lbl in enumerate(all_labels)}
        return all_labels, vs2u

    def test_scores_shape(self):
        all_labels, vs2u = self._setup()
        T = 10
        df = _make_vjepa_df(["Walking"] * T, [0.9] * T)
        scores = vjepa_to_frame_scores(df, all_labels, vs2u)
        assert scores.shape == (T, len(all_labels))

    def test_predicted_class_gets_confidence(self):
        all_labels, vs2u = self._setup()
        df = _make_vjepa_df(["Walking"], [0.8])
        scores = vjepa_to_frame_scores(df, all_labels, vs2u)
        walking_idx = vs2u["Walking"]
        assert abs(scores[0, walking_idx] - 0.8) < 1e-9

    def test_remaining_probability_distributed(self):
        all_labels, vs2u = self._setup()
        df = _make_vjepa_df(["Walking"], [0.6])
        scores = vjepa_to_frame_scores(df, all_labels, vs2u)
        assert abs(scores[0].sum() - 1.0) < 1e-9

    def test_unknown_label_leaves_row_zero(self):
        all_labels, vs2u = self._setup()
        df = _make_vjepa_df(["UNKNOWN_LABEL"], [0.99])
        scores = vjepa_to_frame_scores(df, all_labels, vs2u)
        assert scores[0].sum() == 0.0


# ===========================================================================
# 4. find_matching_windows
# ===========================================================================

class TestFindMatchingWindows:

    def _windows(self):
        return [{"start": 0, "end": 10, "scores": np.zeros(3)}]

    def test_exact_match(self):
        by_video = {"video_001": self._windows()}
        result = find_matching_windows("video_001", by_video)
        assert result == by_video["video_001"]

    def test_partial_match_vjepa_in_pyskl(self):
        by_video = {"video_001_extra": self._windows()}
        result = find_matching_windows("video_001", by_video)
        assert result == by_video["video_001_extra"]

    def test_partial_match_pyskl_in_vjepa(self):
        by_video = {"vid": self._windows()}
        result = find_matching_windows("vid_001", by_video)
        assert result == by_video["vid"]

    def test_no_match_returns_empty(self):
        by_video = {"completely_different": self._windows()}
        result = find_matching_windows("video_001", by_video)
        assert result == []

    def test_empty_dict_returns_empty(self):
        result = find_matching_windows("video_001", {})
        assert result == []


# ===========================================================================
# 5. fuse_one_video
# ===========================================================================

class TestFuseOneVideo:

    def _full_setup(self, task: str = "locomotion"):
        label_map  = LABEL_MAPS[task]
        all_labels = sorted(label_map.values())
        vs2u       = {lbl: i for i, lbl in enumerate(all_labels)}
        ps2u       = {k: all_labels.index(v) for k, v in label_map.items()}
        po2u       = {k: all_labels.index(v) for k, v in label_map.items()}
        return all_labels, vs2u, ps2u, po2u, len(label_map)

    def test_output_columns(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T = 5
        df = _make_vjepa_df(["Walking"] * T, [0.9] * T, ["Walking"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert set(result.columns) >= {"frame", "true_label", "predicted_label",
                                       "confidence", "correct", "source"}

    def test_output_length_equals_input(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T = 8
        df     = _make_vjepa_df(["Running"] * T, [0.85] * T, ["Running"] * T)
        w      = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert len(result) == T

    def test_no_skeleton_falls_back_to_vjepa(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T = 3
        df     = _make_vjepa_df(["Walking"] * T, [0.9] * T, ["Walking"] * T)
        result = fuse_one_video(
            df, [], [], all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert (result["source"] == "vjepa").all()
        assert (result["predicted_label"] == "Walking").all()

    def test_source_tag_three_models(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T  = 4
        df = _make_vjepa_df(["Running"] * T, [0.9] * T, ["Running"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert (result["source"] == "vjepa_pyskl_posec3d").all()

    def test_source_tag_vjepa_pyskl_only(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T  = 4
        df = _make_vjepa_df(["Running"] * T, [0.9] * T, ["Running"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, [], all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert (result["source"] == "vjepa_pyskl").all()

    def test_source_tag_vjepa_posec3d_only(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T  = 4
        df = _make_vjepa_df(["Running"] * T, [0.9] * T, ["Running"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, [], w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert (result["source"] == "vjepa_posec3d").all()

    def test_correct_column_values(self):
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        walking_idx = all_labels.index("Walking")
        # Force all models to predict "Walking"
        scores = [0.0] * nc
        scores[walking_idx] = 1.0
        T  = 3
        df = _make_vjepa_df(["Walking"] * T, [0.95] * T, ["Walking"] * T)
        w  = _make_windows([(0, T - 1, scores)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.5, beta=0.3
        )
        assert (result["correct"] == 1).all()

    def test_alpha_beta_boundary(self):
        """alpha + beta == 1.0 means posec3d weight is 0 — should not crash."""
        all_labels, vs2u, ps2u, po2u, nc = self._full_setup()
        T  = 2
        df = _make_vjepa_df(["Running"] * T, [0.9] * T, ["Running"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.5, beta=0.5
        )
        assert len(result) == T

    def test_rmm_task(self):
        label_map  = LABEL_MAPS["rmm"]
        all_labels = sorted(label_map.values())
        vs2u       = {lbl: i for i, lbl in enumerate(all_labels)}
        ps2u       = {k: all_labels.index(v) for k, v in label_map.items()}
        po2u       = {k: all_labels.index(v) for k, v in label_map.items()}
        nc = len(label_map)
        T  = 5
        df = _make_vjepa_df(["Jumping"] * T, [0.8] * T, ["Jumping"] * T)
        w  = _make_windows([(0, T - 1, [1/nc] * nc)])
        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert len(result) == T


# ===========================================================================
# 6. fmt helper
# ===========================================================================

class TestFmt:

    def test_basic_formatting(self):
        result = fmt(0.9123, 0.0456)
        assert "91.2" in result
        assert "4.6" in result

    def test_zero_std(self):
        result = fmt(0.5, 0.0)
        assert "50.0" in result
        assert "0.0" in result

    def test_returns_string(self):
        assert isinstance(fmt(0.75, 0.05), str)


# ===========================================================================
# 7. LABEL_MAPS / constants sanity checks
# ===========================================================================

class TestConstants:

    def test_label_maps_keys(self):
        assert "locomotion" in LABEL_MAPS
        assert "rmm" in LABEL_MAPS

    def test_locomotion_has_six_classes(self):
        assert len(LABEL_MAPS["locomotion"]) == 6

    def test_rmm_has_five_classes(self):
        assert len(LABEL_MAPS["rmm"]) == 5

    def test_none_class_present(self):
        assert "None" in LABEL_MAPS["locomotion"].values()
        assert "None" in LABEL_MAPS["rmm"].values()

    def test_seeds_are_three_ints(self):
        assert len(SEEDS) == 3
        assert all(isinstance(s, int) for s in SEEDS)


# ===========================================================================
# 8. Integration-style smoke test (no I/O)
# ===========================================================================

class TestIntegration:

    def test_full_pipeline_locomotion(self):
        """End-to-end through build_unified_labels → vjepa_to_frame_scores →
        windows_to_frame_scores → fuse_one_video without any disk access."""
        task      = "locomotion"
        label_map = LABEL_MAPS[task]
        vjepa_map = {lbl: i for i, lbl in sorted(label_map.items())}

        all_labels, vs2u, ps2u, po2u = build_unified_labels(
            vjepa_map, label_map, label_map
        )
        nc = len(label_map)
        T  = 20
        true_lbl = "Running"
        df = _make_vjepa_df([true_lbl] * T, [0.88] * T, [true_lbl] * T)

        running_idx = all_labels.index(true_lbl)
        scores = [0.0] * nc
        scores[running_idx] = 1.0
        w = _make_windows([(0, T - 1, scores)])

        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.4, beta=0.3
        )
        assert len(result) == T
        assert (result["predicted_label"] == true_lbl).all()
        assert (result["correct"] == 1).all()

    def test_full_pipeline_rmm(self):
        task      = "rmm"
        label_map = LABEL_MAPS[task]
        vjepa_map = {lbl: i for i, lbl in sorted(label_map.items())}
        all_labels, vs2u, ps2u, po2u = build_unified_labels(
            vjepa_map, label_map, label_map
        )
        nc = len(label_map)
        T  = 15
        true_lbl = "Rocking"
        df = _make_vjepa_df([true_lbl] * T, [0.75] * T, [true_lbl] * T)

        rocking_idx = all_labels.index(true_lbl)
        scores = [0.0] * nc
        scores[rocking_idx] = 1.0
        w = _make_windows([(0, T - 1, scores)])

        result = fuse_one_video(
            df, w, w, all_labels, vs2u, ps2u, po2u, nc, nc, alpha=0.5, beta=0.2
        )
        assert len(result) == T
        assert (result["correct"] == 1).all()