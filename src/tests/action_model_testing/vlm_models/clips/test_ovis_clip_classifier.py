"""
Tests for src/sailsprep/action_model_testing/vlm_models/clips/ovis_clip_classifier.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from clips.ovis_clip_classifier import ClipActionClassifier


def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


class TestOvisClipClassifier:
    @pytest.fixture()
    def clf(self):
        obj = object.__new__(ClipActionClassifier)
        obj.task = "loco"
        obj.class_names = ["Crawling", "Cruising", "Walking", "Running", "Vehicle"]
        obj.random_frames = False
        obj.seed = 42
        obj.num_sample_frames = 8
        obj.max_partition = 9
        return obj

    @pytest.fixture()
    def clf_rmm(self, clf):
        clf.task = "rmm"
        clf.class_names = ["Jumping", "Hands_flapping", "Rocking", "Spinning"]
        return clf

    # --- _parse_action ---
    def test_parse_exact_action_tag(self, clf):
        assert clf._parse_action("ACTION: Walking") == "Walking"

    def test_parse_case_insensitive(self, clf):
        assert clf._parse_action("ACTION: walking") == "Walking"

    def test_parse_fallback_keyword(self, clf):
        assert clf._parse_action("The child is crawling.") == "Crawling"

    def test_parse_returns_none_unknown(self, clf):
        assert clf._parse_action("I have no idea") is None

    def test_parse_rmm_flap_alias_in_tag(self, clf_rmm):
        assert clf_rmm._parse_action("ACTION: flapping") == "Hands_flapping"

    def test_parse_rmm_flap_alias_keyword(self, clf_rmm):
        assert clf_rmm._parse_action("hands flap observed") == "Hands_flapping"

    # --- classify_clip majority vote ---
    def test_majority_vote_correct(self, clf):
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf.classify_frame = MagicMock(side_effect=[
            ("Walking", "ACTION: Walking"),
            ("Walking", "ACTION: Walking"),
            ("Running", "ACTION: Running"),
        ])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred == "Walking"
        assert conf == pytest.approx(2 / 3)
        assert fpreds.count("Walking") == 2

    def test_all_same_vote_confidence_one(self, clf):
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 2)
        clf.classify_frame = MagicMock(return_value=("Running", "ACTION: Running"))
        pred, _, conf = clf.classify_clip("fake.mp4")
        assert pred == "Running"
        assert conf == 1.0

    def test_empty_video_returns_none(self, clf):
        clf._sample_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None
        assert fpreds == []
        assert conf == 0.0

    def test_all_unparseable_returns_none(self, clf):
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 2)
        clf.classify_frame = MagicMock(return_value=(None, "garbage"))
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None

    # --- frame sampling indices ---
    def test_uniform_sampling_indices_count(self, clf):
        """Uniform sampling: number of indices <= num_sample_frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 100 if prop == cv2.CAP_PROP_FRAME_COUNT else 30
        cap.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=cap):
            frames = clf._sample_frames("fake.mp4")
        assert len(frames) <= clf.num_sample_frames

    def test_random_sampling_reproducible(self, clf):
        clf.random_frames = True
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 50 if prop == cv2.CAP_PROP_FRAME_COUNT else 15
        cap.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=cap):
            frames1 = clf._sample_frames("fake.mp4", clip_index=0)
        with patch("cv2.VideoCapture", return_value=cap):
            frames2 = clf._sample_frames("fake.mp4", clip_index=0)
        assert len(frames1) == len(frames2)  # same seed -> same count
