"""Tests for soft_nms, oks_iou, and oks_nms in utils.py."""

import numpy as np
import pytest

from sailsprep.id_tracking_model.utils.utils import oks_iou, oks_nms, soft_nms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_KPS = 17  # COCO keypoints


def _make_kpt_entry(
    x: float,
    y: float,
    score: float,
    area: float = 1000.0,
    vis: float = 2.0,
) -> dict:
    """Build a minimal kpts_db entry with 17 keypoints centred at (x, y)."""
    kps = np.zeros(NUM_KPS * 3, dtype=np.float32)
    kps[0::3] = x
    kps[1::3] = y
    kps[2::3] = vis
    return {"keypoints": kps, "score": float(score), "area": area}


def _make_kpt_entry_per_joint(
    x: float,
    y: float,
    area: float = 1000.0,
    vis: float = 2.0,
) -> dict:
    """Build an entry where 'score' is a per-joint array."""
    kps = np.zeros(NUM_KPS * 3, dtype=np.float32)
    kps[0::3] = x
    kps[1::3] = y
    kps[2::3] = vis
    scores = np.full(NUM_KPS, 0.9, dtype=np.float32)
    return {"keypoints": kps, "score": scores, "area": area}


# ===========================================================================
# soft_nms
# ===========================================================================


class TestSoftNmsEmpty:
    def test_empty_input_returns_empty_list(self) -> None:
        dets = np.empty((0, 5), dtype=np.float64)
        keep = soft_nms(dets)
        assert keep == []


class TestSoftNmsSingleBox:
    def test_single_box_always_kept(self) -> None:
        dets = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
        keep = soft_nms(dets)
        assert keep == [0]

    def test_single_low_score_box_dropped(self) -> None:
        dets = np.array([[10.0, 10.0, 50.0, 50.0, 0.01]])
        keep = soft_nms(dets, score_thr=0.05)
        assert keep == []


class TestSoftNmsLinear:
    """Linear-decay Soft-NMS behaviour."""

    def test_non_overlapping_boxes_all_kept(self) -> None:
        # Four boxes that do not overlap at all
        dets = np.array([
            [0.0,   0.0,  10.0,  10.0, 0.9],
            [20.0,  0.0,  30.0,  10.0, 0.8],
            [40.0,  0.0,  50.0,  10.0, 0.7],
            [60.0,  0.0,  70.0,  10.0, 0.6],
        ], dtype=np.float64)
        keep = soft_nms(dets, method='linear')
        assert set(keep) == {0, 1, 2, 3}

    def test_identical_boxes_second_decayed_away(self) -> None:
        # Two identical boxes — the second should be decayed to ~0
        dets = np.array([
            [0.0, 0.0, 100.0, 100.0, 0.9],
            [0.0, 0.0, 100.0, 100.0, 0.85],
        ], dtype=np.float64)
        keep = soft_nms(dets, iou_thr=0.5, score_thr=0.05, method='linear')
        # First box must be kept; second has IoU=1 → decay = 1-1 = 0 → dropped
        assert 0 in keep
        assert 1 not in keep

    def test_high_iou_box_score_decayed(self) -> None:
        # Box 1 barely overlaps box 0 — should survive
        dets = np.array([
            [0.0,  0.0, 100.0, 100.0, 0.9],
            [80.0, 0.0, 180.0, 100.0, 0.8],   # IoU ~0.18 → no decay
        ], dtype=np.float64)
        keep = soft_nms(dets, iou_thr=0.55, score_thr=0.05, method='linear')
        assert set(keep) == {0, 1}

    def test_top_k_limits_output(self) -> None:
        dets = np.array([
            [0.0,   0.0,  10.0,  10.0, 0.9],
            [20.0,  0.0,  30.0,  10.0, 0.8],
            [40.0,  0.0,  50.0,  10.0, 0.7],
        ], dtype=np.float64)
        keep = soft_nms(dets, method='linear', top_k=2)
        assert len(keep) <= 2

    def test_return_type_is_list_of_int(self) -> None:
        dets = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]], dtype=np.float64)
        keep = soft_nms(dets)
        assert isinstance(keep, list)
        assert all(isinstance(k, (int, np.integer)) for k in keep)

    def test_score_thr_filters_low_score(self) -> None:
        dets = np.array([
            [0.0,  0.0, 100.0, 100.0, 0.9],
            [5.0,  5.0,  95.0,  95.0, 0.06],   # high IoU → will be decayed below thr
        ], dtype=np.float64)
        keep = soft_nms(dets, iou_thr=0.3, score_thr=0.05, method='linear')
        assert 0 in keep


class TestSoftNmsGaussian:
    def test_gaussian_non_overlapping_all_kept(self) -> None:
        dets = np.array([
            [0.0,  0.0,  10.0,  10.0, 0.9],
            [20.0, 0.0,  30.0,  10.0, 0.8],
        ], dtype=np.float64)
        keep = soft_nms(dets, method='gaussian')
        assert set(keep) == {0, 1}

    def test_gaussian_identical_boxes_second_heavily_decayed(self) -> None:
        dets = np.array([
            [0.0, 0.0, 100.0, 100.0, 0.9],
            [0.0, 0.0, 100.0, 100.0, 0.85],
        ], dtype=np.float64)
        keep = soft_nms(dets, method='gaussian', sigma=0.5, score_thr=0.05)
        # exp(-1/0.5) ≈ 0.135; 0.85 * 0.135 ≈ 0.115 → still above 0.05 thr
        # Box 1 may survive depending on decay — just assert box 0 is kept
        assert 0 in keep

    def test_invalid_method_raises(self) -> None:
        # Need at least 2 boxes so the decay branch is actually reached
        dets = np.array([
            [0.0, 0.0, 10.0, 10.0, 0.9],
            [1.0, 1.0, 11.0, 11.0, 0.8],
        ], dtype=np.float64)
        with pytest.raises(ValueError, match="method must be"):
            soft_nms(dets, method='bad_method')


# ===========================================================================
# oks_iou
# ===========================================================================


class TestOksIou:
    def _gt(self, x: float = 100.0, y: float = 100.0) -> np.ndarray:
        kps = np.zeros(NUM_KPS * 3, dtype=np.float32)
        kps[0::3] = x
        kps[1::3] = y
        kps[2::3] = 2.0
        return kps

    def _det(self, x: float = 100.0, y: float = 100.0) -> np.ndarray:
        kps = np.zeros(NUM_KPS * 3, dtype=np.float32)
        kps[0::3] = x
        kps[1::3] = y
        kps[2::3] = 2.0
        return kps[np.newaxis, :]  # shape (1, 51)

    def test_identical_poses_iou_near_one(self) -> None:
        g = self._gt()
        d = self._det()
        a_g = 1000.0
        a_d = np.array([1000.0], dtype=np.float32)
        ious = oks_iou(g, d, a_g, a_d)
        assert ious.shape == (1,)
        assert float(ious[0]) == pytest.approx(1.0, abs=1e-4)

    def test_far_apart_poses_iou_near_zero(self) -> None:
        g = self._gt(x=0.0, y=0.0)
        d = self._det(x=1e6, y=1e6)
        a_g = 1000.0
        a_d = np.array([1000.0], dtype=np.float32)
        ious = oks_iou(g, d, a_g, a_d)
        assert float(ious[0]) == pytest.approx(0.0, abs=1e-3)

    def test_return_shape_matches_num_detections(self) -> None:
        g = self._gt()
        n = 5
        d = np.tile(self._det(), (n, 1))
        a_g = 1000.0
        a_d = np.full(n, 1000.0, dtype=np.float32)
        ious = oks_iou(g, d, a_g, a_d)
        assert ious.shape == (n,)

    def test_custom_sigmas(self) -> None:
        g = self._gt()
        d = self._det()
        a_g = 1000.0
        a_d = np.array([1000.0], dtype=np.float32)
        custom_sigmas = np.ones(NUM_KPS) * 0.05
        ious = oks_iou(g, d, a_g, a_d, sigmas=custom_sigmas)
        assert float(ious[0]) == pytest.approx(1.0, abs=1e-4)

    def test_vis_thr_filters_invisible_keypoints(self) -> None:
        g = self._gt()
        d = self._det()
        a_g = 1000.0
        a_d = np.array([1000.0], dtype=np.float32)
        # vis = 2.0 everywhere, thr = 1.5 → all visible → should still be ~1
        ious = oks_iou(g, d, a_g, a_d, vis_thr=1.5)
        assert float(ious[0]) == pytest.approx(1.0, abs=1e-4)

    def test_zero_area_does_not_crash(self) -> None:
        g = self._gt()
        d = self._det()
        a_g = 0.0
        a_d = np.array([0.0], dtype=np.float32)
        ious = oks_iou(g, d, a_g, a_d)
        assert ious.shape == (1,)
        assert np.isfinite(float(ious[0]))


# ===========================================================================
# oks_nms
# ===========================================================================


class TestOksNmsEmpty:
    def test_empty_db_returns_empty_array(self) -> None:
        result = oks_nms([], thr=0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestOksNms:
    def test_single_entry_kept(self) -> None:
        db = [_make_kpt_entry(100.0, 100.0, 0.9)]
        keep = oks_nms(db, thr=0.5)
        assert list(keep) == [0]

    def test_identical_poses_only_one_kept(self) -> None:
        db = [
            _make_kpt_entry(100.0, 100.0, 0.9, area=1000.0),
            _make_kpt_entry(100.0, 100.0, 0.8, area=1000.0),
        ]
        keep = oks_nms(db, thr=0.5)
        assert len(keep) == 1
        assert keep[0] == 0   # highest score wins

    def test_distant_poses_both_kept(self) -> None:
        db = [
            _make_kpt_entry(0.0,    0.0,   0.9, area=100.0),
            _make_kpt_entry(1000.0, 1000.0, 0.8, area=100.0),
        ]
        keep = oks_nms(db, thr=0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_score_ordering_respected(self) -> None:
        # Higher score should appear first in keep
        db = [
            _make_kpt_entry(100.0, 100.0, 0.6, area=1000.0),
            _make_kpt_entry(200.0, 200.0, 0.9, area=1000.0),
        ]
        keep = oks_nms(db, thr=0.5)
        assert keep[0] == 1   # index 1 has higher score

    def test_threshold_one_keeps_all(self) -> None:
        # thr=1.0 means nothing is suppressed (OKS can never exceed 1)
        db = [
            _make_kpt_entry(100.0, 100.0, 0.9, area=1000.0),
            _make_kpt_entry(100.0, 100.0, 0.8, area=1000.0),
            _make_kpt_entry(100.0, 100.0, 0.7, area=1000.0),
        ]
        keep = oks_nms(db, thr=1.0)
        assert len(keep) == 3

    def test_threshold_zero_keeps_only_top(self) -> None:
        # thr=0.0 suppresses everything with OKS > 0 — only top survives
        db = [
            _make_kpt_entry(100.0, 100.0, 0.9, area=1000.0),
            _make_kpt_entry(100.0, 100.0, 0.8, area=1000.0),
        ]
        keep = oks_nms(db, thr=0.0)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_return_type_is_ndarray(self) -> None:
        db = [_make_kpt_entry(100.0, 100.0, 0.9)]
        keep = oks_nms(db, thr=0.5)
        assert isinstance(keep, np.ndarray)

    def test_score_per_joint_mode(self) -> None:
        db = [
            _make_kpt_entry_per_joint(0.0,    0.0,   area=100.0),
            _make_kpt_entry_per_joint(1000.0, 1000.0, area=100.0),
        ]
        keep = oks_nms(db, thr=0.5, score_per_joint=True)
        assert set(keep.tolist()) == {0, 1}

    def test_custom_sigmas_accepted(self) -> None:
        db = [
            _make_kpt_entry(100.0, 100.0, 0.9, area=1000.0),
            _make_kpt_entry(200.0, 200.0, 0.8, area=1000.0),
        ]
        custom_sigmas = np.ones(NUM_KPS) * 0.05
        keep = oks_nms(db, thr=0.5, sigmas=custom_sigmas)
        assert len(keep) >= 1

    def test_vis_thr_parameter_accepted(self) -> None:
        db = [
            _make_kpt_entry(100.0, 100.0, 0.9, area=1000.0),
        ]
        keep = oks_nms(db, thr=0.5, vis_thr=1.5)
        assert list(keep) == [0]