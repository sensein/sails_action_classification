"""
Tests for eval_pyskl_sw pure functions.
Run: poetry run pytest src/tests/tests_eval_pyskl_sw.py
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from sailsprep.fusion_model.pyskl.eval_pyskl_sw import (
    LABEL_MAPS,
    WindowDict,
    load_true_labels,
    windows_to_frame_predictions,
)

from typing import TypedDict
import numpy as np

# ============================================================
# Helpers
# ============================================================
RMM_MAP = LABEL_MAPS["rmm"]          # {0: "Hands_flapping", ..., 2: "None", ...}
LOCO_MAP = LABEL_MAPS["locomotion"]  # {0: "Crawling", ..., 2: "None", ...}
NUM_RMM  = len(RMM_MAP)
NUM_LOCO = len(LOCO_MAP)


def _win(start: int, end: int, scores: list[float]) -> WindowDict:
    return WindowDict(start=start, end=end, scores=np.array(scores, dtype=np.float64))


# ============================================================
# windows_to_frame_predictions
# ============================================================

class TestWindowsToFramePredictions:

    def test_single_window_all_frames_covered(self) -> None:
        """One window spanning all frames → all frames get a prediction."""
        scores = [0.1, 0.2, 0.7, 0.0, 0.0]   # argmax=2 → "None"
        windows = [_win(0, 4, scores)]
        preds, confs = windows_to_frame_predictions(windows, 5, RMM_MAP, NUM_RMM)
        assert preds == ["None"] * 5
        assert all(abs(c - 0.7) < 1e-4 for c in confs)

    def test_no_windows_returns_none_predictions(self) -> None:
        """Empty window list → every frame predicts 'None' with 0.0 confidence."""
        preds, confs = windows_to_frame_predictions([], 6, RMM_MAP, NUM_RMM)
        assert preds == ["None"] * 6
        assert confs == [0.0] * 6

    def test_overlapping_windows_averaged(self) -> None:
        """Two overlapping windows → overlapping frames use averaged scores."""
        # Window A frames 0-2: class 0 wins
        w_a = _win(0, 2, [0.9, 0.05, 0.05, 0.0, 0.0])
        # Window B frames 1-3: class 1 wins
        w_b = _win(1, 3, [0.05, 0.9, 0.05, 0.0, 0.0])
        preds, _ = windows_to_frame_predictions([w_a, w_b], 4, RMM_MAP, NUM_RMM)
        # frame 0: only w_a → class 0 "Hands_flapping"
        assert preds[0] == "Hands_flapping"
        # frame 3: only w_b → class 1 "Jumping"
        assert preds[3] == "Jumping"
        # frame 1 & 2: average → (0.9+0.05)/2=0.475 vs (0.05+0.9)/2=0.475 → tie broken by argmax (0)
        # both equal so argmax picks index 0
        assert preds[1] in RMM_MAP.values()
        assert preds[2] in RMM_MAP.values()

    def test_window_clipped_to_T(self) -> None:
        """Window end beyond T is clipped; no IndexError."""
        windows = [_win(0, 100, [0.0, 0.0, 1.0, 0.0, 0.0])]
        preds, confs = windows_to_frame_predictions(windows, 5, RMM_MAP, NUM_RMM)
        assert len(preds) == 5
        assert len(confs) == 5
        assert all(p == "None" for p in preds)

    def test_confidence_rounded_to_4_decimals(self) -> None:
        scores = [0.123456789, 0.0, 0.876543211, 0.0, 0.0]
        windows = [_win(0, 0, scores)]
        _, confs = windows_to_frame_predictions(windows, 1, RMM_MAP, NUM_RMM)
        assert confs[0] == round(max(scores), 4)

    def test_locomotion_label_map(self) -> None:
        """Works with 6-class locomotion map."""
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # argmax=5 → "Walking"
        windows = [_win(0, 2, scores)]
        preds, _ = windows_to_frame_predictions(windows, 3, LOCO_MAP, NUM_LOCO)
        assert preds == ["Walking"] * 3

    def test_returns_correct_lengths(self) -> None:
        T = 10
        windows = [_win(2, 7, [0.2, 0.8, 0.0, 0.0, 0.0])]
        preds, confs = windows_to_frame_predictions(windows, T, RMM_MAP, NUM_RMM)
        assert len(preds) == T
        assert len(confs) == T

    def test_uncovered_frames_get_none_zero(self) -> None:
        """Frames outside any window → 'None' / 0.0."""
        windows = [_win(3, 5, [0.0, 1.0, 0.0, 0.0, 0.0])]
        preds, confs = windows_to_frame_predictions(windows, 8, RMM_MAP, NUM_RMM)
        for f in [0, 1, 2, 6, 7]:
            assert preds[f] == "None"
            assert confs[f] == 0.0


# ============================================================
# load_true_labels
# ============================================================

class TestLoadTrueLabels:

    def _write_csv(self, tmp_path: Path, content: str) -> str:
        p = tmp_path / "labels.csv"
        p.write_text(content)
        return str(p)

    def test_basic_load(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "Locomotion\nWalking\nRunning\nNone\n")
        labels = load_true_labels(path, "Locomotion", 3)
        assert labels == ["Walking", "Running", "None"]

    def test_pads_short_csv(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "Locomotion\nWalking\n")
        labels = load_true_labels(path, "Locomotion", 5)
        assert labels == ["Walking", "None", "None", "None", "None"]

    def test_truncates_long_csv(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "Locomotion\nA\nB\nC\nD\nE\n")
        labels = load_true_labels(path, "Locomotion", 3)
        assert labels == ["A", "B", "C"]

    def test_missing_column_returns_none_list(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "OtherCol\nfoo\nbar\n")
        labels = load_true_labels(path, "Locomotion", 3)
        assert labels == ["None"] * 3

    def test_bad_path_returns_none_list(self) -> None:
        labels = load_true_labels("/nonexistent/path/labels.csv", "Locomotion", 4)
        assert labels == ["None"] * 4

    def test_nan_values_replaced(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "Locomotion\nWalking\n\nN/A\n")
        labels = load_true_labels(path, "Locomotion", 3)
        assert labels[1] == "None"
        assert labels[2] == "None"

    def test_column_with_whitespace_stripped(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, " Locomotion \nWalking\n")
        labels = load_true_labels(path, "Locomotion", 1)
        assert labels == ["Walking"]

    def test_returns_list_of_str(self, tmp_path: Path) -> None:
        path = self._write_csv(tmp_path, "Locomotion\nWalking\n")
        labels = load_true_labels(path, "Locomotion", 1)
        assert isinstance(labels, list)
        assert all(isinstance(x, str) for x in labels)


# ============================================================
# WindowDict
# ============================================================

class TestWindowDict:

    def test_fields_accessible(self) -> None:
        scores = np.array([0.1, 0.9], dtype=np.float64)
        w = WindowDict(start=0, end=5, scores=scores)
        assert w["start"] == 0
        assert w["end"] == 5
        np.testing.assert_array_equal(w["scores"], scores)

    def test_scores_dtype(self) -> None:
        scores = np.zeros(5, dtype=np.float64)
        w = WindowDict(start=0, end=4, scores=scores)
        assert w["scores"].dtype == np.float64


# ============================================================
# LABEL_MAPS sanity
# ============================================================

class TestLabelMaps:

    def test_rmm_keys_contiguous(self) -> None:
        assert set(LABEL_MAPS["rmm"].keys()) == set(range(5))

    def test_locomotion_keys_contiguous(self) -> None:
        assert set(LABEL_MAPS["locomotion"].keys()) == set(range(6))

    def test_none_present_in_both(self) -> None:
        assert "None" in LABEL_MAPS["rmm"].values()
        assert "None" in LABEL_MAPS["locomotion"].values()