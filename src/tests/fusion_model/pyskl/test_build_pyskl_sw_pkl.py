"""
Tests for build_pyskl_sw_pkl.py

Run:
    poetry run pytest src/tests/tests_build_pyskl_sw_pkl.py
"""

import json
import os
import pickle
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent.parent / "sailsprep" / "fusion_model" / "pyskl"),
)

import build_pyskl_sw_pkl as M  # noqa: E402  (after sys.path patch)


# ===========================================================================
# Helpers
# ===========================================================================

COCO_KPS = M.COCO_KEYPOINTS  # 17 keypoint names


def _make_vitpose_json(tmp_dir: str, n_frames: int = 40) -> str:
    """Write a minimal vitpose JSON with n_frames frames."""
    frames: dict[str, Any] = {}
    for i in range(n_frames):
        frame: dict[str, Any] = {}
        for kp in COCO_KPS:
            frame[kp] = {"x": float(i + 1), "y": float(i + 2), "confidence": 0.9}
        frames[str(i)] = frame

    data = {"frames": frames}
    path = os.path.join(tmp_dir, "pose.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_label_csv(tmp_dir: str, n_rows: int, task: str = "locomotion") -> str:
    """Write a minimal annotation CSV."""
    col = M.TASK_COLUMN[task]
    labels = list(M.LABEL_MAPS[task].keys())
    rows = [labels[i % len(labels)] for i in range(n_rows)]
    df = pd.DataFrame({col: rows})
    path = os.path.join(tmp_dir, "labels.csv")
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# load_pose_json
# ===========================================================================

class TestLoadPoseJson:
    def test_returns_correct_shape(self, tmp_path: Path) -> None:
        n = 10
        p = _make_vitpose_json(str(tmp_path), n_frames=n)
        kp, sc, T = M.load_pose_json(p)
        assert T == n
        assert kp.shape == (n, 17, 2)
        assert sc.shape == (n, 17)

    def test_dtype_float32(self, tmp_path: Path) -> None:
        p = _make_vitpose_json(str(tmp_path), n_frames=5)
        kp, sc, _ = M.load_pose_json(p)
        assert kp.dtype == np.float32
        assert sc.dtype == np.float32

    def test_values_populated(self, tmp_path: Path) -> None:
        p = _make_vitpose_json(str(tmp_path), n_frames=3)
        kp, sc, T = M.load_pose_json(p)
        # All confidences should be 0.9
        assert np.allclose(sc, 0.9)
        # x of frame 0 keypoint 0 should be 1.0
        assert kp[0, 0, 0] == pytest.approx(1.0)

    def test_missing_keypoint_defaults_zero(self, tmp_path: Path) -> None:
        """If a keypoint is absent from a frame it should default to 0."""
        frames: dict[str, Any] = {
            "0": {}  # empty frame — no keypoints
        }
        data = {"frames": frames}
        p = str(tmp_path / "pose_missing.json")
        with open(p, "w") as f:
            json.dump(data, f)
        kp, sc, T = M.load_pose_json(p)
        assert T == 1
        assert np.all(kp == 0)
        assert np.all(sc == 0)

    def test_frame_ordering(self, tmp_path: Path) -> None:
        """Frames must be sorted numerically not lexicographically."""
        frames: dict[str, Any] = {}
        for i in [0, 9, 10, 1]:
            frame = {kp: {"x": float(i), "y": 0.0, "confidence": 1.0} for kp in COCO_KPS}
            frames[str(i)] = frame
        data = {"frames": frames}
        p = str(tmp_path / "pose_order.json")
        with open(p, "w") as f:
            json.dump(data, f)
        kp, _, T = M.load_pose_json(p)
        assert T == 4
        # Sorted order: 0,1,9,10 → x values at t=0..3 should be 0,1,9,10
        assert kp[0, 0, 0] == pytest.approx(0.0)
        assert kp[1, 0, 0] == pytest.approx(1.0)
        assert kp[2, 0, 0] == pytest.approx(9.0)
        assert kp[3, 0, 0] == pytest.approx(10.0)


# ===========================================================================
# load_annotations
# ===========================================================================

class TestLoadAnnotations:
    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        p = _make_label_csv(str(tmp_path), n_rows=20)
        result = M.load_annotations(p, "Locomotion", 20)
        assert result is not None
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)

    def test_length_matches_T(self, tmp_path: Path) -> None:
        p = _make_label_csv(str(tmp_path), n_rows=20)
        result = M.load_annotations(p, "Locomotion", 25)
        assert result is not None
        assert len(result) == 25

    def test_pads_with_none_when_short(self, tmp_path: Path) -> None:
        p = _make_label_csv(str(tmp_path), n_rows=5)
        result = M.load_annotations(p, "Locomotion", 10)
        assert result is not None
        assert len(result) == 10
        assert result[5] == "None"

    def test_truncates_when_long(self, tmp_path: Path) -> None:
        p = _make_label_csv(str(tmp_path), n_rows=30)
        result = M.load_annotations(p, "Locomotion", 10)
        assert result is not None
        assert len(result) == 10

    def test_returns_none_on_missing_file(self) -> None:
        result = M.load_annotations("/nonexistent/path.csv", "Locomotion", 10)
        assert result is None

    def test_returns_none_on_missing_column(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"WrongCol": ["A", "B"]})
        p = str(tmp_path / "bad.csv")
        df.to_csv(p, index=False)
        result = M.load_annotations(p, "Locomotion", 2)
        assert result is None

    def test_nan_replaced_with_none(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"Locomotion": [float("nan"), "Walking"]})
        p = str(tmp_path / "nan.csv")
        df.to_csv(p, index=False)
        result = M.load_annotations(p, "Locomotion", 2)
        assert result is not None
        assert result[0] == "None"

    def test_rmm_task(self, tmp_path: Path) -> None:
        p = _make_label_csv(str(tmp_path), n_rows=10, task="rmm")
        result = M.load_annotations(p, "Repetitive_Motor_Movements", 10)
        assert result is not None
        assert len(result) == 10


# ===========================================================================
# build_windows_one_video
# ===========================================================================

class TestBuildWindowsOneVideo:
    def _make_inputs(
        self, T: int = 60, task: str = "locomotion"
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
        kp = np.random.rand(T, 17, 2).astype(np.float32)
        sc = np.random.rand(T, 17).astype(np.float32)
        label_map = M.LABEL_MAPS[task]
        labels = ["Walking"] * T
        return kp, sc, labels, label_map

    def test_number_of_windows(self) -> None:
        T = 60
        kp, sc, labels, label_map = self._make_inputs(T)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        # floor((T - W) / S) + 1
        expected = (T - M.WINDOW_FRAMES) // M.STRIDE_FRAMES + 1
        assert len(windows) == expected

    def test_window_shape(self) -> None:
        kp, sc, labels, label_map = self._make_inputs(60)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        w = windows[0]
        assert w["keypoint"].shape == (1, M.WINDOW_FRAMES, 17, 2)
        assert w["keypoint_score"].shape == (1, M.WINDOW_FRAMES, 17)

    def test_label_is_int(self) -> None:
        kp, sc, labels, label_map = self._make_inputs(60)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        for w in windows:
            assert isinstance(w["label"], int)

    def test_majority_vote(self) -> None:
        T = 30
        kp = np.zeros((T, 17, 2), dtype=np.float32)
        sc = np.zeros((T, 17), dtype=np.float32)
        label_map = M.LABEL_MAPS["locomotion"]
        # 20 Walking, 10 Running in first window
        labels = ["Walking"] * 20 + ["Running"] * 10
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        assert windows[0]["label"] == label_map["Walking"]

    def test_unknown_label_falls_back_to_none(self) -> None:
        T = 30
        kp = np.zeros((T, 17, 2), dtype=np.float32)
        sc = np.zeros((T, 17), dtype=np.float32)
        label_map = M.LABEL_MAPS["locomotion"]
        labels = ["UNKNOWN_LABEL"] * T
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        assert windows[0]["label"] == label_map["None"]

    def test_frame_dir_unique(self) -> None:
        kp, sc, labels, label_map = self._make_inputs(90)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        ids = [w["frame_dir"] for w in windows]
        assert len(ids) == len(set(ids))

    def test_total_frames_field(self) -> None:
        kp, sc, labels, label_map = self._make_inputs(60)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        for w in windows:
            assert w["total_frames"] == M.WINDOW_FRAMES

    def test_no_windows_when_video_too_short(self) -> None:
        T = M.WINDOW_FRAMES - 1
        kp = np.zeros((T, 17, 2), dtype=np.float32)
        sc = np.zeros((T, 17), dtype=np.float32)
        labels = ["Walking"] * T
        label_map = M.LABEL_MAPS["locomotion"]
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, (480, 640)
        )
        assert windows == []

    def test_img_shape_stored(self) -> None:
        kp, sc, labels, label_map = self._make_inputs(30)
        img_shape = (720, 1280)
        windows = M.build_windows_one_video(
            kp, sc, labels, "vid", label_map, img_shape
        )
        for w in windows:
            assert w["img_shape"] == img_shape
            assert w["original_shape"] == img_shape


# ===========================================================================
# save_pkl
# ===========================================================================

class TestSavePkl:
    def _dummy_annotations(self, n: int = 4) -> list[dict[str, Any]]:
        kp = np.zeros((1, M.WINDOW_FRAMES, 17, 2), dtype=np.float32)
        sc = np.zeros((1, M.WINDOW_FRAMES, 17), dtype=np.float32)
        return [
            {
                "frame_dir": f"vid_{i}",
                "label": i % 3,
                "img_shape": (480, 640),
                "original_shape": (480, 640),
                "total_frames": M.WINDOW_FRAMES,
                "keypoint": kp.copy(),
                "keypoint_score": sc.copy(),
            }
            for i in range(n)
        ]

    def test_file_created(self, tmp_path: Path) -> None:
        anns = self._dummy_annotations()
        splits: dict[str, list[str]] = {
            "train": ["vid_0", "vid_1"],
            "val": ["vid_2"],
            "test": ["vid_3"],
        }
        out = M.save_pkl(anns, splits, "locomotion", str(tmp_path))
        assert os.path.exists(out)

    def test_pkl_structure(self, tmp_path: Path) -> None:
        anns = self._dummy_annotations()
        splits: dict[str, list[str]] = {
            "train": ["vid_0"],
            "val": ["vid_1"],
            "test": ["vid_2"],
        }
        out = M.save_pkl(anns, splits, "locomotion", str(tmp_path))
        with open(out, "rb") as f:
            data = pickle.load(f)
        assert "split" in data
        assert "annotations" in data
        assert data["split"] == splits
        loaded = data["annotations"]
        assert len(loaded) == len(anns)
        for loaded_ann, orig_ann in zip(loaded, anns):
            assert loaded_ann["frame_dir"] == orig_ann["frame_dir"]
            assert loaded_ann["label"] == orig_ann["label"]
            assert loaded_ann["img_shape"] == orig_ann["img_shape"]
            assert loaded_ann["original_shape"] == orig_ann["original_shape"]
            assert loaded_ann["total_frames"] == orig_ann["total_frames"]
            np.testing.assert_array_equal(loaded_ann["keypoint"], orig_ann["keypoint"])
            np.testing.assert_array_equal(loaded_ann["keypoint_score"], orig_ann["keypoint_score"])

    def test_filename_contains_task(self, tmp_path: Path) -> None:
        anns = self._dummy_annotations(1)
        splits: dict[str, list[str]] = {"train": ["vid_0"], "val": [], "test": []}
        out = M.save_pkl(anns, splits, "rmm", str(tmp_path))
        assert "rmm" in os.path.basename(out)


# ===========================================================================
# Config generators — just check files are created and contain key strings
# ===========================================================================

class TestConfigGenerators:
    def test_posec3d_config_created(self, tmp_path: Path) -> None:
        out = M.generate_posec3d_config(
            "locomotion", "/data/loco.pkl", 6, str(tmp_path)
        )
        assert os.path.exists(out)
        content = Path(out).read_text()
        assert "num_classes=6" in content
        assert "locomotion" in content
        assert "posec3d" in out

    def test_ctrgcn_config_created(self, tmp_path: Path) -> None:
        out = M.generate_ctrgcn_config(
            "locomotion", "/data/loco.pkl", 6, "b", str(tmp_path)
        )
        assert os.path.exists(out)
        content = Path(out).read_text()
        assert "num_classes=6" in content
        assert "CTRGCN" in content

    def test_stgcnpp_config_created(self, tmp_path: Path) -> None:
        out = M.generate_stgcnpp_config(
            "rmm", "/data/rmm.pkl", 5, "jm", str(tmp_path)
        )
        assert os.path.exists(out)
        content = Path(out).read_text()
        assert "num_classes=5" in content
        assert "STGCN" in content

    def test_ctrgcn_feat_in_filename(self, tmp_path: Path) -> None:
        out = M.generate_ctrgcn_config(
            "rmm", "/data/rmm.pkl", 5, "jm", str(tmp_path)
        )
        assert "jm.py" in out

    def test_stgcnpp_feat_in_filename(self, tmp_path: Path) -> None:
        out = M.generate_stgcnpp_config(
            "locomotion", "/data/loco.pkl", 6, "b", str(tmp_path)
        )
        assert "b.py" in out

    def test_posec3d_num_classes_rmm(self, tmp_path: Path) -> None:
        out = M.generate_posec3d_config(
            "rmm", "/data/rmm.pkl", 5, str(tmp_path)
        )
        content = Path(out).read_text()
        assert "num_classes=5" in content

    def test_ann_file_in_posec3d_config(self, tmp_path: Path) -> None:
        ann = "/some/path/rmm.pkl"
        out = M.generate_posec3d_config("rmm", ann, 5, str(tmp_path))
        content = Path(out).read_text()
        assert ann in content

    def test_ann_file_in_ctrgcn_config(self, tmp_path: Path) -> None:
        ann = "/some/path/loco.pkl"
        out = M.generate_ctrgcn_config("locomotion", ann, 6, "b", str(tmp_path))
        content = Path(out).read_text()
        assert ann in content


# ===========================================================================
# Constants / label maps sanity
# ===========================================================================

class TestLabelMaps:
    def test_locomotion_classes(self) -> None:
        lm = M.LABEL_MAPS["locomotion"]
        assert set(lm.keys()) == {"Crawling", "Cruising", "None", "Running", "Vehicle", "Walking"}
        assert sorted(lm.values()) == list(range(6))

    def test_rmm_classes(self) -> None:
        lm = M.LABEL_MAPS["rmm"]
        assert set(lm.keys()) == {"Hands_flapping", "Jumping", "None", "Rocking", "Spinning"}
        assert sorted(lm.values()) == list(range(5))

    def test_coco_keypoints_count(self) -> None:
        assert len(M.COCO_KEYPOINTS) == 17

    def test_window_stride_positive(self) -> None:
        assert M.WINDOW_FRAMES > 0
        assert M.STRIDE_FRAMES > 0
        assert M.STRIDE_FRAMES <= M.WINDOW_FRAMES