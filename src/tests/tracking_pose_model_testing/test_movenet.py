"""
src/tests/tracking_pose_model_testing/test_movenet.py

Unit tests for the MoveNet pose-estimation geometry helpers.
  Script under test : src/sailsprep/tracking_pose_model_testing/movenet.py
  This test file    : src/tests/tracking_pose_model_testing/test_movenet.py

movenet.py performs top-level TF-Hub model loading (tensorflow / tensorflow_hub /
tensorflow_docs / IPython are not installed in this environment) and opens a
hardcoded video file at import time, so it cannot be safely imported or
exec'd. Following the fallback pattern already used by test_yolo_pose.py, the
pure numpy-only geometry functions are copied verbatim from the source file
and unit-tested directly here. These functions contain no TF/cv2 calls.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_movenet.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Verbatim copies of the pure functions from movenet.py
# ─────────────────────────────────────────────────────────────────────────────

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

MIN_CROP_KEYPOINT_SCORE = 0.2


def _keypoints_and_edges_for_display(keypoints_with_scores,
                                      height,
                                      width,
                                      keypoint_threshold=0.11):
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def init_crop_region(image_height, image_width):
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }


def torso_visible(keypoints):
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
             MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
             MIN_CROP_KEYPOINT_SCORE))


def determine_torso_and_body_range(
        keypoints, target_keypoints, center_y, center_x):
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def determine_crop_region(keypoints, image_height, image_width):
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] +
                    target_keypoints['right_hip'][0]) / 2
        center_x = (target_keypoints['left_hip'][1] +
                    target_keypoints['right_hip'][1]) / 2

        (max_torso_yrange, max_torso_xrange,
         max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x)

        crop_length_half = np.amax(
            [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
             max_body_yrange * 1.2, max_body_xrange * 1.2])

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                          crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                         crop_corner[1] / image_width
            }
    else:
        return init_crop_region(image_height, image_width)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kps(scores: dict[str, float] | None = None,
         coords: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """Build a [1, 1, 17, 3] keypoints_with_scores array (all zero by default)."""
    arr = np.zeros((1, 1, 17, 3), dtype=np.float64)
    if scores:
        for joint, score in scores.items():
            arr[0, 0, KEYPOINT_DICT[joint], 2] = score
    if coords:
        for joint, (y, x) in coords.items():
            arr[0, 0, KEYPOINT_DICT[joint], 0] = y
            arr[0, 0, KEYPOINT_DICT[joint], 1] = x
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Tests: init_crop_region
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestInitCropRegion:

    def test_landscape_image(self):
        region = init_crop_region(image_height=100, image_width=200)
        assert region['width'] == pytest.approx(1.0)
        assert region['height'] == pytest.approx(2.0)
        assert region['x_min'] == pytest.approx(0.0)

    def test_portrait_image(self):
        region = init_crop_region(image_height=200, image_width=100)
        assert region['height'] == pytest.approx(1.0)
        assert region['width'] == pytest.approx(2.0)
        assert region['y_min'] == pytest.approx(0.0)

    def test_square_image(self):
        region = init_crop_region(image_height=100, image_width=100)
        assert region['height'] == pytest.approx(1.0)
        assert region['width'] == pytest.approx(1.0)
        assert region['x_min'] == pytest.approx(0.0)
        assert region['y_min'] == pytest.approx(0.0)

    def test_x_max_y_max_consistency(self):
        region = init_crop_region(image_height=100, image_width=300)
        assert region['x_max'] == pytest.approx(region['x_min'] + region['width'])
        assert region['y_max'] == pytest.approx(region['y_min'] + region['height'])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: torso_visible
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTorsoVisible:

    def test_true_when_hip_and_shoulder_confident(self):
        kps = _kps(scores={'left_hip': 0.5, 'left_shoulder': 0.5})
        assert torso_visible(kps) is True or bool(torso_visible(kps)) is True

    def test_false_when_no_hip_confident(self):
        kps = _kps(scores={'left_shoulder': 0.9, 'right_shoulder': 0.9})
        assert not torso_visible(kps)

    def test_false_when_no_shoulder_confident(self):
        kps = _kps(scores={'left_hip': 0.9, 'right_hip': 0.9})
        assert not torso_visible(kps)

    def test_true_with_right_side_only(self):
        kps = _kps(scores={'right_hip': 0.5, 'right_shoulder': 0.5})
        assert bool(torso_visible(kps)) is True

    def test_false_when_all_scores_low(self):
        kps = _kps(scores={
            'left_hip': 0.05, 'right_hip': 0.05,
            'left_shoulder': 0.05, 'right_shoulder': 0.05,
        })
        assert not torso_visible(kps)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: determine_torso_and_body_range
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDetermineTorsoAndBodyRange:

    def test_zero_range_when_all_at_center(self):
        kps = _kps(scores={j: 0.9 for j in KEYPOINT_DICT})
        target_keypoints = {j: [10.0, 10.0] for j in KEYPOINT_DICT}
        result = determine_torso_and_body_range(kps, target_keypoints, 10.0, 10.0)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_nonzero_range_detected(self):
        kps = _kps(scores={j: 0.9 for j in KEYPOINT_DICT})
        target_keypoints = {j: [10.0, 10.0] for j in KEYPOINT_DICT}
        target_keypoints['left_shoulder'] = [50.0, 10.0]
        result = determine_torso_and_body_range(kps, target_keypoints, 10.0, 10.0)
        max_torso_yrange = result[0]
        assert max_torso_yrange == pytest.approx(40.0)

    def test_low_confidence_joint_excluded_from_body_range(self):
        kps = _kps(scores={j: 0.9 for j in KEYPOINT_DICT})
        # Push nose confidence below threshold so it is excluded from body range
        kps[0, 0, KEYPOINT_DICT['nose'], 2] = 0.01
        target_keypoints = {j: [0.0, 0.0] for j in KEYPOINT_DICT}
        target_keypoints['nose'] = [1000.0, 1000.0]
        result = determine_torso_and_body_range(kps, target_keypoints, 0.0, 0.0)
        max_body_yrange, max_body_xrange = result[2], result[3]
        assert max_body_yrange == 0.0
        assert max_body_xrange == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: determine_crop_region
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDetermineCropRegion:

    def test_falls_back_to_default_when_torso_not_visible(self):
        kps = _kps()  # all scores zero -> torso not visible
        region = determine_crop_region(kps, image_height=100, image_width=100)
        default = init_crop_region(100, 100)
        assert region == default

    def test_returns_region_when_torso_visible(self):
        kps = _kps(
            scores={j: 0.9 for j in KEYPOINT_DICT},
            coords={
                'left_hip': (0.5, 0.4),
                'right_hip': (0.5, 0.6),
                'left_shoulder': (0.3, 0.4),
                'right_shoulder': (0.3, 0.6),
            },
        )
        region = determine_crop_region(kps, image_height=200, image_width=200)
        assert set(region.keys()) == {'y_min', 'x_min', 'y_max', 'x_max', 'height', 'width'}
        assert region['height'] > 0
        assert region['width'] > 0

    def test_result_keys_present(self):
        kps = _kps()
        region = determine_crop_region(kps, image_height=50, image_width=80)
        assert set(region.keys()) == {'y_min', 'x_min', 'y_max', 'x_max', 'height', 'width'}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _keypoints_and_edges_for_display
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestKeypointsAndEdgesForDisplay:

    def test_no_keypoints_above_threshold(self):
        kps = _kps()  # all scores 0
        keypoints_xy, edges_xy, edge_colors = _keypoints_and_edges_for_display(
            kps, height=100, width=100)
        assert keypoints_xy.shape[0] == 0
        assert edges_xy.shape[0] == 0
        assert edge_colors == []

    def test_single_high_confidence_keypoint(self):
        kps = _kps(scores={'nose': 0.9})
        keypoints_xy, edges_xy, edge_colors = _keypoints_and_edges_for_display(
            kps, height=100, width=100)
        assert keypoints_xy.shape[0] == 1

    def test_edge_constructed_when_both_endpoints_confident(self):
        kps = _kps(scores={'nose': 0.9, 'left_eye': 0.9})
        _, edges_xy, edge_colors = _keypoints_and_edges_for_display(
            kps, height=100, width=100)
        assert edges_xy.shape[0] == 1
        assert edge_colors == ['m']  # (0, 1) -> 'm'

    def test_edge_thresholding_respected(self):
        kps = _kps(scores={'nose': 0.9, 'left_eye': 0.05})
        _, edges_xy, edge_colors = _keypoints_and_edges_for_display(
            kps, height=100, width=100, keypoint_threshold=0.11)
        assert edges_xy.shape[0] == 0

    def test_absolute_coords_scaled_by_width_height(self):
        kps = _kps(scores={'nose': 0.9}, coords={'nose': (0.5, 0.25)})
        keypoints_xy, _, _ = _keypoints_and_edges_for_display(
            kps, height=100, width=200)
        # x = width * kpts_x (col 1 -> x), y = height * kpts_y (col 0 -> y)
        np.testing.assert_allclose(keypoints_xy[0], [50.0, 50.0])
