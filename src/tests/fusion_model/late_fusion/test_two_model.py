"""Tests for sailsprep.fusion_model.late_fusion.two_model."""
from __future__ import annotations

import json
import os
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sailsprep.fusion_model.late_fusion.two_model import (
    build_unified_labels,
    find_matching_windows,
    fmt,
    fuse_one_video,
    load_pyskl_window_preds,
    load_vjepa_label_map,
    load_vjepa_predictions,
    vjepa_to_frame_scores,
    windows_to_frame_scores,
)

MODULE = "sailsprep.fusion_model.late_fusion.two_model"


# ============================================================
# fmt
# ============================================================
class TestFmt:
    def test_basic(self) -> None:
        assert fmt(0.856, 0.023) == "85.6 ± 2.3"

    def test_zero(self) -> None:
        assert fmt(0.0, 0.0) == "0.0 ± 0.0"

    def test_one(self) -> None:
        assert fmt(1.0, 0.0) == "100.0 ± 0.0"

    def test_rounding(self) -> None:
        result = fmt(0.1234, 0.0056)
        assert "12.3" in result
        assert "0.6" in result


# ============================================================
# windows_to_frame_scores
# ============================================================
class TestWindowsToFrameScores:
    def test_empty_windows(self) -> None:
        scores, counts = windows_to_frame_scores([], T=5, num_classes=3)
        assert scores.shape == (5, 3)
        assert np.all(scores == 0)
        assert np.all(counts == 0)

    def test_single_window(self) -> None:
        w = [{"start": 1, "end": 3, "scores": np.array([0.1, 0.7, 0.2])}]
        scores, counts = windows_to_frame_scores(w, T=5, num_classes=3)
        np.testing.assert_array_almost_equal(scores[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(scores[1], [0.1, 0.7, 0.2])
        np.testing.assert_array_almost_equal(scores[2], [0.1, 0.7, 0.2])
        np.testing.assert_array_almost_equal(scores[3], [0.1, 0.7, 0.2])
        np.testing.assert_array_almost_equal(scores[4], [0, 0, 0])
        assert counts[1] == 1
        assert counts[0] == 0

    def test_overlapping_windows_averaged(self) -> None:
        windows = [
            {"start": 0, "end": 2, "scores": np.array([1.0, 0.0, 0.0])},
            {"start": 1, "end": 3, "scores": np.array([0.0, 1.0, 0.0])},
        ]
        scores, counts = windows_to_frame_scores(windows, T=4, num_classes=3)
        np.testing.assert_array_almost_equal(scores[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(scores[1], [0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(scores[2], [0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(scores[3], [0.0, 1.0, 0.0])
        assert counts[1] == 2

    def test_window_clipped_at_T(self) -> None:
        w = [{"start": 3, "end": 10, "scores": np.array([0.5, 0.5])}]
        scores, counts = windows_to_frame_scores(w, T=5, num_classes=2)
        assert counts[3] == 1
        assert counts[4] == 1

    def test_counts_shape(self) -> None:
        w = [{"start": 0, "end": 1, "scores": np.array([0.3, 0.7])}]
        _, counts = windows_to_frame_scores(w, T=3, num_classes=2)
        assert counts.shape == (3,)
        assert counts[0] == 1
        assert counts[1] == 1
        assert counts[2] == 0


# ============================================================
# build_unified_labels
# ============================================================
class TestBuildUnifiedLabels:
    def test_union_of_labels(self) -> None:
        vjepa_map = {"Walking": 0, "Running": 1}
        pyskl_map = {0: "Walking", 1: "Crawling"}
        labels, _, _ = build_unified_labels(vjepa_map, pyskl_map)
        assert set(labels) == {"Crawling", "Running", "Walking"}

    def test_labels_sorted(self) -> None:
        vjepa_map = {"Walking": 0, "Running": 1}
        pyskl_map = {0: "Crawling"}
        labels, _, _ = build_unified_labels(vjepa_map, pyskl_map)
        assert labels == sorted(labels)

    def test_vjepa_index_mapping(self) -> None:
        vjepa_map = {"Walking": 0, "Running": 1}
        pyskl_map = {0: "Walking"}
        labels, v2u, _ = build_unified_labels(vjepa_map, pyskl_map)
        assert v2u["Walking"] == labels.index("Walking")
        assert v2u["Running"] == labels.index("Running")

    def test_pyskl_index_mapping(self) -> None:
        vjepa_map = {"Walking": 0}
        pyskl_map = {0: "Walking", 1: "Crawling"}
        labels, _, p2u = build_unified_labels(vjepa_map, pyskl_map)
        assert p2u[0] == labels.index("Walking")
        assert p2u[1] == labels.index("Crawling")

    def test_disjoint_labels(self) -> None:
        vjepa_map = {"A": 0}
        pyskl_map = {0: "B"}
        labels, v2u, p2u = build_unified_labels(vjepa_map, pyskl_map)
        assert "A" in labels
        assert "B" in labels
        assert "A" in v2u
        assert 0 in p2u

    def test_identical_labels(self) -> None:
        vjepa_map = {"Walking": 0}
        pyskl_map = {0: "Walking"}
        labels, v2u, p2u = build_unified_labels(vjepa_map, pyskl_map)
        assert labels == ["Walking"]
        assert v2u["Walking"] == 0
        assert p2u[0] == 0


# ============================================================
# vjepa_to_frame_scores
# ============================================================
class TestVjepaToFrameScores:
    @staticmethod
    def _make_df(
        labels: list[str],
        confs: list[float],
        true_labels: list[str] | None = None,
    ) -> pd.DataFrame:
        if true_labels is None:
            true_labels = labels
        return pd.DataFrame([
            {"frame": i, "predicted_label": lbl, "confidence": conf, "true_label": tl}
            for i, (lbl, conf, tl) in enumerate(zip(labels, confs, true_labels))
        ])

    def test_correct_class_gets_confidence(self) -> None:
        df = self._make_df(["Walking"], [0.9])
        scores = vjepa_to_frame_scores(df, ["Running", "Walking"], {"Running": 0, "Walking": 1})
        assert scores[0, 1] == pytest.approx(0.9)

    def test_remaining_distributed_to_other_class(self) -> None:
        df = self._make_df(["Walking"], [0.9])
        scores = vjepa_to_frame_scores(df, ["Running", "Walking"], {"Running": 0, "Walking": 1})
        assert scores[0, 0] == pytest.approx(0.1)

    def test_multiple_frames(self) -> None:
        df = self._make_df(["Walking", "Running"], [0.9, 0.8])
        scores = vjepa_to_frame_scores(df, ["Running", "Walking"], {"Running": 0, "Walking": 1})
        assert scores[0, 1] == pytest.approx(0.9)
        assert scores[1, 0] == pytest.approx(0.8)

    def test_unknown_label_gives_zero_scores(self) -> None:
        df = self._make_df(["Unknown"], [0.9])
        scores = vjepa_to_frame_scores(df, ["Running", "Walking"], {"Running": 0, "Walking": 1})
        np.testing.assert_array_equal(scores[0], [0.0, 0.0])

    def test_output_shape(self) -> None:
        df = self._make_df(["Walking", "Running", "Walking"], [0.9, 0.8, 0.7])
        scores = vjepa_to_frame_scores(
            df,
            ["Crawling", "Running", "Walking"],
            {"Crawling": 0, "Running": 1, "Walking": 2},
        )
        assert scores.shape == (3, 3)

    def test_three_classes_remaining_split_evenly(self) -> None:
        df = self._make_df(["Walking"], [0.6])
        labels = ["Crawling", "Running", "Walking"]
        v2u = {"Crawling": 0, "Running": 1, "Walking": 2}
        scores = vjepa_to_frame_scores(df, labels, v2u)
        assert scores[0, 2] == pytest.approx(0.6)
        assert scores[0, 0] == pytest.approx(0.2)
        assert scores[0, 1] == pytest.approx(0.2)


# ============================================================
# find_matching_windows
# ============================================================
class TestFindMatchingWindows:
    @staticmethod
    def _pyskl() -> dict[str, list[dict]]:
        return {"video_001": [{"start": 0, "end": 30, "scores": np.zeros(3)}]}

    def test_exact_match(self) -> None:
        result = find_matching_windows("video_001", self._pyskl())
        assert len(result) == 1

    def test_vjepa_name_contains_pyskl_name(self) -> None:
        result = find_matching_windows("prefix_video_001_suffix", self._pyskl())
        assert len(result) == 1

    def test_pyskl_name_contains_vjepa_name(self) -> None:
        pyskl = {"prefix_video_001_suffix": [{"start": 0, "end": 30, "scores": np.zeros(3)}]}
        result = find_matching_windows("video_001", pyskl)
        assert len(result) == 1

    def test_no_match_returns_empty_list(self) -> None:
        result = find_matching_windows("video_999", self._pyskl())
        assert result == []

    def test_multiple_windows_returned(self) -> None:
        pyskl = {
            "video_001": [
                {"start": 0, "end": 15, "scores": np.zeros(3)},
                {"start": 15, "end": 30, "scores": np.zeros(3)},
            ]
        }
        result = find_matching_windows("video_001", pyskl)
        assert len(result) == 2


# ============================================================
# fuse_one_video
# ============================================================
class TestFuseOneVideo:
    _LABELS = ["Running", "Walking"]
    _V2U = {"Running": 0, "Walking": 1}
    _P2U = {0: 0, 1: 1}

    @staticmethod
    def _vjepa_df(n: int = 3, pred: str = "Walking", conf: float = 0.8) -> pd.DataFrame:
        return pd.DataFrame([
            {"frame": i, "predicted_label": pred, "confidence": conf, "true_label": "Walking"}
            for i in range(n)
        ])

    def test_no_pyskl_windows_source_is_vjepa(self) -> None:
        df = self._vjepa_df(2)
        result = fuse_one_video(df, [], self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        assert list(result["source"]) == ["vjepa", "vjepa"]
        assert list(result["predicted_label"]) == ["Walking", "Walking"]

    def test_with_pyskl_windows_source_is_fused(self) -> None:
        df = self._vjepa_df(2)
        windows = [{"start": 0, "end": 1, "scores": np.array([0.9, 0.1])}]
        result = fuse_one_video(df, windows, self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        assert list(result["source"]) == ["fused", "fused"]

    def test_correct_flag(self) -> None:
        df = pd.DataFrame([
            {"frame": 0, "predicted_label": "Walking", "confidence": 0.9, "true_label": "Walking"},
            {"frame": 1, "predicted_label": "Walking", "confidence": 0.9, "true_label": "Running"},
        ])
        result = fuse_one_video(df, [], self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        assert result.iloc[0]["correct"] == 1
        assert result.iloc[1]["correct"] == 0

    def test_output_columns(self) -> None:
        df = self._vjepa_df(2)
        result = fuse_one_video(df, [], self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        expected = {"frame", "true_label", "predicted_label", "confidence", "correct", "source"}
        assert expected.issubset(set(result.columns))

    def test_output_length_matches_input(self) -> None:
        df = self._vjepa_df(5)
        result = fuse_one_video(df, [], self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        assert len(result) == 5

    def test_alpha_zero_uses_only_vjepa(self) -> None:
        df = pd.DataFrame([{
            "frame": 0, "predicted_label": "Walking", "confidence": 1.0, "true_label": "Walking",
        }])
        # PySkl strongly predicts Running
        windows = [{"start": 0, "end": 0, "scores": np.array([0.99, 0.01])}]
        result = fuse_one_video(df, windows, self._LABELS, self._V2U, self._P2U, 2, alpha=0.0)
        assert result.iloc[0]["predicted_label"] == "Walking"

    def test_alpha_one_uses_only_pyskl(self) -> None:
        df = pd.DataFrame([{
            "frame": 0, "predicted_label": "Walking", "confidence": 1.0, "true_label": "Walking",
        }])
        windows = [{"start": 0, "end": 0, "scores": np.array([0.99, 0.01])}]
        result = fuse_one_video(df, windows, self._LABELS, self._V2U, self._P2U, 2, alpha=1.0)
        assert result.iloc[0]["predicted_label"] == "Running"

    def test_confidence_rounded_to_4_places(self) -> None:
        df = self._vjepa_df(1)
        result = fuse_one_video(df, [], self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        conf = result.iloc[0]["confidence"]
        assert conf == round(conf, 4)

    def test_mixed_fused_and_vjepa_sources(self) -> None:
        """Frames covered by a window get 'fused'; uncovered frames get 'vjepa'."""
        df = self._vjepa_df(4)
        # Window covers only frames 0–1
        windows = [{"start": 0, "end": 1, "scores": np.array([0.5, 0.5])}]
        result = fuse_one_video(df, windows, self._LABELS, self._V2U, self._P2U, 2, alpha=0.5)
        assert result.iloc[0]["source"] == "fused"
        assert result.iloc[1]["source"] == "fused"
        assert result.iloc[2]["source"] == "vjepa"
        assert result.iloc[3]["source"] == "vjepa"


# ============================================================
# load_vjepa_label_map
# ============================================================
class TestLoadVjepaLabelMap:
    def test_loads_json(self, tmp_path: pytest.TempPathFactory) -> None:
        label_map = {"Walking": 0, "Running": 1}
        label_dir = tmp_path / "locomotion" / "seed_42"
        label_dir.mkdir(parents=True)
        (label_dir / "label_mapping.json").write_text(json.dumps(label_map))

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_label_map("locomotion", 42)

        assert result == label_map

    def test_returns_all_keys(self, tmp_path: pytest.TempPathFactory) -> None:
        label_map = {"Walking": 0, "Running": 1, "Crawling": 2, "None": 3}
        label_dir = tmp_path / "rmm" / "seed_123"
        label_dir.mkdir(parents=True)
        (label_dir / "label_mapping.json").write_text(json.dumps(label_map))

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_label_map("rmm", 123)

        assert set(result.keys()) == set(label_map.keys())


# ============================================================
# load_vjepa_predictions
# ============================================================
class TestLoadVjepaPredictions:
    def test_returns_empty_dict_if_dir_missing(self, tmp_path: pytest.TempPathFactory) -> None:
        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_predictions("locomotion", 42)
        assert result == {}

    def test_loads_csv_files(self, tmp_path: pytest.TempPathFactory) -> None:
        pred_dir = tmp_path / "locomotion" / "seed_42" / "per_video_predictions"
        pred_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "frame": [0, 1],
            "predicted_label": ["Walking", "Running"],
            "confidence": [0.9, 0.8],
            "true_label": ["Walking", "Walking"],
        })
        df.to_csv(pred_dir / "video_001_predictions.csv", index=False)

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_predictions("locomotion", 42)

        assert "video_001" in result
        assert len(result["video_001"]) == 2

    def test_ignores_non_prediction_files(self, tmp_path: pytest.TempPathFactory) -> None:
        pred_dir = tmp_path / "locomotion" / "seed_42" / "per_video_predictions"
        pred_dir.mkdir(parents=True)
        (pred_dir / "README.txt").write_text("ignore me")

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_predictions("locomotion", 42)

        assert result == {}

    def test_sorted_by_frame(self, tmp_path: pytest.TempPathFactory) -> None:
        pred_dir = tmp_path / "locomotion" / "seed_42" / "per_video_predictions"
        pred_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "frame": [2, 0, 1],
            "predicted_label": ["Walking", "Running", "None"],
            "confidence": [0.7, 0.9, 0.8],
            "true_label": ["Walking", "Walking", "Walking"],
        })
        df.to_csv(pred_dir / "video_001_predictions.csv", index=False)

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_predictions("locomotion", 42)

        frames = list(result["video_001"]["frame"])
        assert frames == sorted(frames)

    def test_multiple_videos(self, tmp_path: pytest.TempPathFactory) -> None:
        pred_dir = tmp_path / "locomotion" / "seed_42" / "per_video_predictions"
        pred_dir.mkdir(parents=True)
        for name in ("alpha", "beta", "gamma"):
            pd.DataFrame({
                "frame": [0], "predicted_label": ["Walking"],
                "confidence": [0.9], "true_label": ["Walking"],
            }).to_csv(pred_dir / f"{name}_predictions.csv", index=False)

        with patch(f"{MODULE}.VJEPA_BASE", str(tmp_path)):
            result = load_vjepa_predictions("locomotion", 42)

        assert set(result.keys()) == {"alpha", "beta", "gamma"}


# ============================================================
# load_pyskl_window_preds
# ============================================================
class TestLoadPysklWindowPreds:
    def _setup_files(
        self,
        tmp_path: pytest.TempPathFactory,
        test_ids: list[str],
        scores_list: list[np.ndarray],
    ) -> tuple[str, str]:
        """Create fake pred pkl and data pkl; return (pyskl_base, pkl_base)."""
        pred_dir = tmp_path / "stgcnpp_locomotion_sw" / "b_s42"
        pred_dir.mkdir(parents=True)
        with open(pred_dir / "test_pred.pkl", "wb") as f:
            pickle.dump(scores_list, f)

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pkl_data = {"split": {"test": test_ids}}
        with open(data_dir / "locomotion_slidingwindow_pyskl.pkl", "wb") as f:
            pickle.dump(pkl_data, f)

        return str(tmp_path), str(data_dir)

    def test_loads_and_parses_windows(self, tmp_path: pytest.TempPathFactory) -> None:
        fake_scores = np.array([0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
        pyskl_base, pkl_base = self._setup_files(
            tmp_path,
            test_ids=["video_001_0_30_w0"],
            scores_list=[fake_scores],
        )
        with patch(f"{MODULE}.PYSKL_BASE", pyskl_base), \
             patch(f"{MODULE}.PKL_BASE", pkl_base):
            result = load_pyskl_window_preds("locomotion", 42)

        assert "video_001" in result
        assert len(result["video_001"]) == 1
        w = result["video_001"][0]
        assert w["start"] == 0
        assert w["end"] == 30
        np.testing.assert_array_almost_equal(w["scores"], fake_scores)

    def test_multiple_windows_same_video(self, tmp_path: pytest.TempPathFactory) -> None:
        s = np.zeros(6)
        pyskl_base, pkl_base = self._setup_files(
            tmp_path,
            test_ids=["video_001_0_15_w0", "video_001_15_30_w1"],
            scores_list=[s, s],
        )
        with patch(f"{MODULE}.PYSKL_BASE", pyskl_base), \
             patch(f"{MODULE}.PKL_BASE", pkl_base):
            result = load_pyskl_window_preds("locomotion", 42)

        assert len(result["video_001"]) == 2

    def test_multiple_videos(self, tmp_path: pytest.TempPathFactory) -> None:
        s = np.zeros(6)
        pyskl_base, pkl_base = self._setup_files(
            tmp_path,
            test_ids=["video_001_0_30_w0", "video_002_0_30_w0"],
            scores_list=[s, s],
        )
        with patch(f"{MODULE}.PYSKL_BASE", pyskl_base), \
             patch(f"{MODULE}.PKL_BASE", pkl_base):
            result = load_pyskl_window_preds("locomotion", 42)

        assert "video_001" in result
        assert "video_002" in result

    def test_malformed_ids_skipped(self, tmp_path: pytest.TempPathFactory) -> None:
        pyskl_base, pkl_base = self._setup_files(
            tmp_path,
            test_ids=["bad_id_no_window_suffix"],
            scores_list=[np.zeros(6)],
        )
        with patch(f"{MODULE}.PYSKL_BASE", pyskl_base), \
             patch(f"{MODULE}.PKL_BASE", pkl_base):
            result = load_pyskl_window_preds("locomotion", 42)

        total = sum(len(v) for v in result.values())
        assert total == 0

    def test_window_start_end_parsed_correctly(self, tmp_path: pytest.TempPathFactory) -> None:
        pyskl_base, pkl_base = self._setup_files(
            tmp_path,
            test_ids=["subj01_walk_100_250_w3"],
            scores_list=[np.zeros(6)],
        )
        with patch(f"{MODULE}.PYSKL_BASE", pyskl_base), \
             patch(f"{MODULE}.PKL_BASE", pkl_base):
            result = load_pyskl_window_preds("locomotion", 42)

        key = "subj01_walk"
        assert key in result
        w = result[key][0]
        assert w["start"] == 100
        assert w["end"] == 250
        