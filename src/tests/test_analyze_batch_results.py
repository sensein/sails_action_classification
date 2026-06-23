#!/usr/bin/env python3
"""
Tests for analyze_batch_results.py
"""

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results import (
    analyze_tracking_quality,
    calculate_bbox_similarity,
    create_tracking_visualizations,
    create_visualizations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(
    track_id: int = 1,
    age_prob: float = 0.8,
    evidence_flags: list[str] | None = None,
    start_frame: int = 0,
    end_frame: int = 30,
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "age_prob": age_prob,
        "evidence_flags": evidence_flags or ["small_size"],
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def make_row(
    filename: str = "video_001.mp4",
    confidence: float = 0.9,
    nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if nodes is None:
        nodes = [make_node()]
    return {
        "filename": filename,
        "confidence": confidence,
        "num_segments": 1,
        "total_duration": 10.0,
        "selected_tracks": [1],
        "processing_time": 2.5,
        "fps": 30.0,
        "total_frames": 300,
        "total_nodes": len(nodes),
        "total_edges": 0,
        "age_probs": [n["age_prob"] for n in nodes if n["age_prob"] is not None],
        "evidence_flags": [f for n in nodes for f in n["evidence_flags"]],
        "tracking_data": {},
        "detailed_nodes": nodes,
        "detailed_edges": [],
    }


@pytest.fixture()
def single_row_df() -> pd.DataFrame:
    return pd.DataFrame([make_row()])


@pytest.fixture()
def multi_row_df() -> pd.DataFrame:
    rows = [
        make_row("video_001.mp4", 0.95, [make_node(1, 0.9, ["small_size"], 0, 60)]),
        make_row("video_002.mp4", 0.75, [make_node(2, 0.6, ["low_height"], 0, 45)]),
        make_row(
            "video_003.mp4",
            0.85,
            [
                make_node(1, 0.85, ["small_size"], 0, 30),
                make_node(2, 0.80, ["small_size"], 31, 60),
            ],
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_analysis_json(tmp_path: Path) -> Path:
    """Write a minimal analysis JSON file and return its path."""
    data = {
        "video_info": {
            "filename": "test_video.mp4",
            "processing_time_seconds": 3.0,
            "fps": 30.0,
            "total_frames": 900,
        },
        "child_identification": {
            "confidence": 0.88,
            "num_segments": 1,
            "total_duration_seconds": 30.0,
            "selected_track_ids": [1],
        },
        "detailed_analysis": {
            "total_nodes": 1,
            "total_edges": 0,
            "nodes": [
                {
                    "track_id": 1,
                    "age_prob": 0.82,
                    "evidence_flags": ["small_size"],
                    "start_frame": 0,
                    "end_frame": 900,
                }
            ],
            "edges": [],
        },
    }
    file_path = tmp_path / "test_video_analysis.json"
    file_path.write_text(json.dumps(data))
    return file_path


# ---------------------------------------------------------------------------
# calculate_bbox_similarity
# ---------------------------------------------------------------------------

class TestCalculateBboxSimilarity:
    def test_identical_boxes_iou_is_one(self) -> None:
        bbox = (0.0, 0.0, 100.0, 100.0)
        result = calculate_bbox_similarity(bbox, bbox)
        assert result["iou"] == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self) -> None:
        bbox1 = (0.0, 0.0, 10.0, 10.0)
        bbox2 = (20.0, 20.0, 30.0, 30.0)
        result = calculate_bbox_similarity(bbox1, bbox2)
        assert result["iou"] == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        bbox1 = (0.0, 0.0, 10.0, 10.0)
        bbox2 = (5.0, 5.0, 15.0, 15.0)
        result = calculate_bbox_similarity(bbox1, bbox2)
        assert 0.0 < result["iou"] < 1.0

    def test_center_distance_identical_boxes(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = calculate_bbox_similarity(bbox, bbox)
        assert result["center_distance"] == pytest.approx(0.0)

    def test_size_ratio_identical_boxes(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = calculate_bbox_similarity(bbox, bbox)
        assert result["size_ratio"] == pytest.approx(1.0)

    def test_size_ratio_different_sizes(self) -> None:
        bbox1 = (0.0, 0.0, 10.0, 10.0)
        bbox2 = (0.0, 0.0, 20.0, 20.0)
        result = calculate_bbox_similarity(bbox1, bbox2)
        assert result["size_ratio"] == pytest.approx(0.5)

    def test_returns_dict_with_expected_keys(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        result = calculate_bbox_similarity(bbox, bbox)
        assert set(result.keys()) == {"iou", "center_distance", "size_ratio"}

    def test_zero_area_box(self) -> None:
        """Degenerate case: zero-area box should not crash."""
        bbox1 = (5.0, 5.0, 5.0, 5.0)
        bbox2 = (0.0, 0.0, 10.0, 10.0)
        result = calculate_bbox_similarity(bbox1, bbox2)
        assert result["iou"] == pytest.approx(0.0)
        assert result["size_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analyze_tracking_quality
# ---------------------------------------------------------------------------

class TestAnalyzeTrackingQuality:
    def test_returns_expected_keys(self, single_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(single_row_df)
        expected_keys = {
            "total_videos",
            "potential_merges",
            "potential_splits",
            "track_fragmentations",
            "merge_probabilities",
            "quality_scores",
        }
        assert expected_keys == set(result.keys())

    def test_total_videos_count(self, multi_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(multi_row_df)
        assert result["total_videos"] == len(multi_row_df)

    def test_quality_scores_length_matches_df(self, multi_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(multi_row_df)
        assert len(result["quality_scores"]) == len(multi_row_df)

    def test_quality_scores_clamped_between_0_and_1(self, multi_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(multi_row_df)
        for score in result["quality_scores"]:
            assert 0.0 <= score <= 1.0

    def test_single_track_no_merges(self, single_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(single_row_df)
        assert result["potential_merges"] == []

    def test_two_similar_tracks_detected_as_merge(self) -> None:
        nodes = [
            make_node(1, 0.85, ["small_size", "low_height"], 0, 30),
            make_node(2, 0.80, ["small_size", "low_height"], 0, 30),
        ]
        df = pd.DataFrame([make_row("video.mp4", 0.9, nodes)])
        result = analyze_tracking_quality(df)
        assert len(result["potential_merges"]) > 0

    def test_fragmented_track_detected(self) -> None:
        """Same track_id appearing in two segments with a small gap → potential split."""
        nodes = [
            make_node(1, 0.8, ["small_size"], 0, 10),
            make_node(1, 0.8, ["small_size"], 15, 25),  # gap of 5 frames
        ]
        df = pd.DataFrame([make_row("video.mp4", 0.9, nodes)])
        result = analyze_tracking_quality(df)
        assert len(result["potential_splits"]) > 0

    def test_fragmentation_score_zero_for_single_segment(self, single_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(single_row_df)
        frag = result["track_fragmentations"][0]
        assert frag["fragmentation_score"] == 0

    def test_merge_probability_within_bounds(self) -> None:
        nodes = [
            make_node(1, 0.85, ["small_size", "low_height"], 0, 30),
            make_node(2, 0.80, ["small_size", "low_height"], 0, 30),
        ]
        df = pd.DataFrame([make_row("video.mp4", 0.9, nodes)])
        result = analyze_tracking_quality(df)
        for merge in result["potential_merges"]:
            assert 0.0 <= merge["merge_probability"] <= 1.0

    def test_empty_nodes_list(self) -> None:
        df = pd.DataFrame([make_row("video.mp4", 0.9, [])])
        result = analyze_tracking_quality(df)
        assert result["quality_scores"][0] == pytest.approx(0.9)

    def test_track_fragmentation_entry_per_video(self, multi_row_df: pd.DataFrame) -> None:
        result = analyze_tracking_quality(multi_row_df)
        assert len(result["track_fragmentations"]) == len(multi_row_df)


# ---------------------------------------------------------------------------
# create_visualizations (smoke tests — just check it doesn't crash)
# ---------------------------------------------------------------------------

class TestCreateVisualizations:
    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.plt")
    def test_does_not_raise(self, mock_plt: MagicMock, multi_row_df: pd.DataFrame, tmp_path: Path) -> None:
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)
        create_visualizations(multi_row_df, tmp_path)

    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.plt")
    def test_saves_plot_file(self, mock_plt: MagicMock, multi_row_df: pd.DataFrame, tmp_path: Path) -> None:
        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        create_visualizations(multi_row_df, tmp_path)
        mock_plt.savefig.assert_called_once()


# ---------------------------------------------------------------------------
# create_tracking_visualizations (smoke tests)
# ---------------------------------------------------------------------------

class TestCreateTrackingVisualizations:
    def _make_tracking_analysis(self, quality_scores: list[float] | None = None) -> dict[str, Any]:
        quality_scores = quality_scores or [0.9, 0.75, 0.85]
        return {
            "total_videos": len(quality_scores),
            "potential_merges": [],
            "potential_splits": [],
            "track_fragmentations": [
                {"video": f"video_{i}.mp4", "num_tracks": 1, "fragmentation_score": 0, "quality_score": q}
                for i, q in enumerate(quality_scores)
            ],
            "merge_probabilities": [],
            "quality_scores": quality_scores,
        }

    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.plt")
    def test_does_not_raise(self, mock_plt: MagicMock, multi_row_df: pd.DataFrame, tmp_path: Path) -> None:
        quality_scores = [0.9, 0.75, 0.85]
        tracking_analysis = {
            "total_videos": len(multi_row_df),
            "potential_merges": [],
            "potential_splits": [],
            "track_fragmentations": [
                {"video": row["filename"], "num_tracks": 1, "fragmentation_score": 0, "quality_score": q}
                for (_, row), q in zip(multi_row_df.iterrows(), quality_scores)
            ],
            "merge_probabilities": [],
            "quality_scores": quality_scores,
        }
        mock_plt.subplot.return_value = MagicMock()
        create_tracking_visualizations(multi_row_df, tracking_analysis, tmp_path)

    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.plt")
    def test_saves_png(self, mock_plt: MagicMock, multi_row_df: pd.DataFrame, tmp_path: Path) -> None:
        quality_scores = [0.9, 0.75, 0.85]
        tracking_analysis = {
            "total_videos": len(multi_row_df),
            "potential_merges": [],
            "potential_splits": [],
            "track_fragmentations": [
                {"video": row["filename"], "num_tracks": 1, "fragmentation_score": 0, "quality_score": q}
                for (_, row), q in zip(multi_row_df.iterrows(), quality_scores)
            ],
            "merge_probabilities": [],
            "quality_scores": quality_scores,
        }
        mock_plt.subplot.return_value = MagicMock()
        create_tracking_visualizations(multi_row_df, tracking_analysis, tmp_path)
        mock_plt.savefig.assert_called_once()

    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.plt")
    def test_exports_csv_when_data_present(
        self, mock_plt: MagicMock, multi_row_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        tracking_analysis = self._make_tracking_analysis(
            [row["confidence"] for _, row in multi_row_df.iterrows()]
        )
        # Assign video names to match df
        for i, entry in enumerate(tracking_analysis["track_fragmentations"]):
            entry["video"] = multi_row_df.iloc[i]["filename"]

        mock_plt.subplot.return_value = MagicMock()
        create_tracking_visualizations(multi_row_df, tracking_analysis, tmp_path)
        csv_path = tmp_path / "tracking_analysis_detailed.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# analyze_batch_results (integration — mocks filesystem)
# ---------------------------------------------------------------------------

class TestAnalyzeBatchResultsIntegration:
    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.create_tracking_visualizations")
    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.create_visualizations")
    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.LOG_DIR")
    def test_runs_with_single_json(
        self,
        mock_log_dir: MagicMock,
        mock_create_vis: MagicMock,
        mock_create_tracking: MagicMock,
        sample_analysis_json: Path,
        tmp_path: Path,
    ) -> None:
        from sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results import analyze_batch_results

        mock_log_dir.__truediv__ = lambda self, other: tmp_path / other
        mock_log_dir.glob.return_value = [sample_analysis_json]

        analyze_batch_results()

        mock_create_vis.assert_called_once()
        mock_create_tracking.assert_called_once()

    @patch("sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results.LOG_DIR")
    def test_no_files_returns_early(self, mock_log_dir: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results import analyze_batch_results

        mock_log_dir.glob.return_value = []
        analyze_batch_results()
        captured = capsys.readouterr()
        assert "No analysis files found" in captured.out