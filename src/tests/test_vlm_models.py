"""Tests for vlm_models — clips, window_classification.

All heavy dependencies (torch, transformers, cv2, qwen_vl_utils) are mocked
so the suite runs without a GPU or downloaded model weights.

Location: src/tests/test_vlm_models.py
Run:      poetry run pytest src/tests/ -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Stub heavy dependencies before any project import touches them
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


# torch — stub must include Tensor class so scipy's is_torch_array() doesn't crash
try:
    import torch as _torch  # use real torch if available (GPU CI)
except ImportError:
    _torch = _stub("torch")

# Always ensure these attrs exist (real torch already has them)
if not hasattr(_torch, "Tensor"):
    class _FakeTensor:
        pass
    _torch.Tensor = _FakeTensor

if not hasattr(_torch, "bfloat16"):
    _torch.bfloat16 = "bfloat16"
if not hasattr(_torch, "float16"):
    _torch.float16 = "float16"
if not hasattr(_torch, "inference_mode"):
    _torch.inference_mode = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
if not hasattr(_torch, "no_grad"):
    _torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
if not hasattr(_torch, "ne"):
    _torch.ne = MagicMock(return_value=MagicMock())
if not hasattr(_torch, "cuda"):
    _torch.cuda = MagicMock()
    _torch.cuda.is_available = MagicMock(return_value=False)

sys.modules["torch"] = _torch

# transformers
_tf = _stub("transformers")
_tf.AutoConfig = MagicMock()
_tf.AutoModelForCausalLM = MagicMock()
_tf.AutoProcessor = MagicMock()
_tf.Qwen2_5_VLForConditionalGeneration = MagicMock()

# qwen_vl_utils
_qvu = _stub("qwen_vl_utils")
_qvu.process_vision_info = MagicMock(return_value=(None, None))

# cv2 — only used in _sample_frames / analyze_video; mocked per-test
_cv2 = _stub("cv2")
_cv2.VideoCapture = MagicMock()
_cv2.CAP_PROP_FRAME_COUNT = 7
_cv2.CAP_PROP_FPS = 5
_cv2.CAP_PROP_POS_FRAMES = 1
_cv2.COLOR_BGR2RGB = 4
_cv2.cvtColor = MagicMock(return_value=np.zeros((64, 64, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


# ============================================================
#  clips — ovis_clip_classifier
# ============================================================

class TestOvisClipClassifier:
    """Tests for clips/ovis_clip_classifier.py."""

    @pytest.fixture()
    def clf(self):
        from clips.ovis_clip_classifier import ClipActionClassifier
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
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf._parse_action("ACTION: Walking") == "Walking"

    def test_parse_case_insensitive(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf._parse_action("ACTION: walking") == "Walking"

    def test_parse_fallback_keyword(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf._parse_action("The child is crawling.") == "Crawling"

    def test_parse_returns_none_unknown(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf._parse_action("I have no idea") is None

    def test_parse_rmm_flap_alias_in_tag(self, clf_rmm):
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf_rmm._parse_action("ACTION: flapping") == "Hands_flapping"

    def test_parse_rmm_flap_alias_keyword(self, clf_rmm):
        from clips.ovis_clip_classifier import ClipActionClassifier
        assert clf_rmm._parse_action("hands flap observed") == "Hands_flapping"

    # --- classify_clip majority vote ---
    def test_majority_vote_correct(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
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
        from clips.ovis_clip_classifier import ClipActionClassifier
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 2)
        clf.classify_frame = MagicMock(return_value=("Running", "ACTION: Running"))
        pred, _, conf = clf.classify_clip("fake.mp4")
        assert pred == "Running"
        assert conf == 1.0

    def test_empty_video_returns_none(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        clf._sample_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None
        assert fpreds == []
        assert conf == 0.0

    def test_all_unparseable_returns_none(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 2)
        clf.classify_frame = MagicMock(return_value=(None, "garbage"))
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None

    # --- frame sampling indices ---
    def test_uniform_sampling_indices_count(self, clf):
        """Uniform sampling: number of indices ≤ num_sample_frames."""
        from clips.ovis_clip_classifier import ClipActionClassifier
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 100 if prop == _cv2.CAP_PROP_FRAME_COUNT else 30
        cap.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=cap):
            frames = clf._sample_frames("fake.mp4")
        assert len(frames) <= clf.num_sample_frames

    def test_random_sampling_reproducible(self, clf):
        from clips.ovis_clip_classifier import ClipActionClassifier
        clf.random_frames = True
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 50 if prop == _cv2.CAP_PROP_FRAME_COUNT else 15
        cap.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=cap):
            frames1 = clf._sample_frames("fake.mp4", clip_index=0)
        with patch("cv2.VideoCapture", return_value=cap):
            frames2 = clf._sample_frames("fake.mp4", clip_index=0)
        assert len(frames1) == len(frames2)  # same seed → same count


# ============================================================
# clips — qwen_clip_classifier
# ============================================================

class TestQwenClipClassifier:
    """Tests for clips/qwen_clip_classifier.py."""

    @pytest.fixture()
    def clf(self):
        from clips.qwen_clip_classifier import ClipActionClassifier
        obj = object.__new__(ClipActionClassifier)
        obj.task = "loco"
        obj.class_names = ["Crawling", "Cruising", "Walking", "Running", "Vehicle"]
        obj.random_frames = False
        obj.seed = 42
        obj.num_sample_frames = 8
        return obj

    def test_parse_exact(self, clf):
        assert clf._parse_action("ACTION: Running") == "Running"

    def test_parse_fallback_keyword(self, clf):
        assert clf._parse_action("child is cruising along wall") == "Cruising"

    def test_parse_none(self, clf):
        assert clf._parse_action("unknown action") is None

    def test_majority_vote(self, clf):
        clf._sample_frames = MagicMock(return_value=[dummy_image()] * 4)
        clf.classify_frame = MagicMock(side_effect=[
            ("Crawling", "ACTION: Crawling"),
            ("Crawling", "ACTION: Crawling"),
            ("Crawling", "ACTION: Crawling"),
            ("Walking", "ACTION: Walking"),
        ])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred == "Crawling"
        assert conf == pytest.approx(3 / 4)

    def test_empty_video(self, clf):
        clf._sample_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_clip("fake.mp4")
        assert pred is None



# ============================================================
#  window_classification — shared_utils
# ============================================================

class TestFrameLabelsToClips:
    """Tests for shared_utils.frame_labels_to_clip_labels."""

    def test_two_windows(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["Walking"] * 30 + ["Crawling"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert len(clips) == 2
        assert clips[0]["label_full"] == "Walking"
        assert clips[1]["label_full"] == "Crawling"

    def test_short_window_skipped(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["Walking"] * 5  # < 50% of 30 frames
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert len(clips) == 0

    def test_unknown_maps_to_no_locomotion(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["unknown_action"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        # label_full stores majority as-is; label_binary maps non-active → No_Locomotion
        assert clips[0]["label_binary"] == "No_Locomotion"

    def test_binary_active(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["Running"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_binary"] == "Locomotion"

    def test_binary_inactive(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["No_Locomotion"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_binary"] == "No_Locomotion"

    def test_rmm_task_active(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["Jumping"] * 30
        clips = frame_labels_to_clip_labels(labels, task="rmm")
        assert clips[0]["label_full"] == "Jumping"
        assert clips[0]["label_binary"] == "RMM"

    def test_rmm_inactive(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["No_RMM"] * 30
        clips = frame_labels_to_clip_labels(labels, task="rmm")
        assert clips[0]["label_binary"] == "No_RMM"

    def test_clip_timing_correct(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels, LABEL_FPS
        labels = ["Walking"] * 30
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["start_sec"] == pytest.approx(0.0)
        assert clips[0]["end_sec"] == pytest.approx(30 / LABEL_FPS)

    def test_majority_determines_label(self):
        from window_classification.shared_utils import frame_labels_to_clip_labels
        labels = ["Walking"] * 20 + ["Running"] * 10
        clips = frame_labels_to_clip_labels(labels, task="loco")
        assert clips[0]["label_full"] == "Walking"


class TestTop2Accuracy:
    """Tests for shared_utils.compute_top2_from_votes."""

    def test_top1_correct(self):
        from window_classification.shared_utils import compute_top2_from_votes
        assert compute_top2_from_votes([["Walking", "Walking"]], ["Walking"]) == 1.0

    def test_top2_second_place(self):
        from window_classification.shared_utils import compute_top2_from_votes
        preds = [["Running", "Running", "Walking"]]
        assert compute_top2_from_votes(preds, ["Walking"]) == 1.0

    def test_top2_miss(self):
        from window_classification.shared_utils import compute_top2_from_votes
        preds = [["Crawling", "Crawling", "Running"]]
        assert compute_top2_from_votes(preds, ["Walking"]) == 0.0

    def test_empty_preds_skip(self):
        from window_classification.shared_utils import compute_top2_from_votes
        assert compute_top2_from_votes([[]], ["Walking"]) == 0.0

    def test_multiple_clips(self):
        from window_classification.shared_utils import compute_top2_from_votes
        preds = [["Walking", "Walking"], ["Running", "Running"]]
        y_true = ["Walking", "Crawling"]
        # first correct, second wrong → 0.5
        assert compute_top2_from_votes(preds, y_true) == pytest.approx(0.5)


class TestLoadFrameLabels:
    """Tests for shared_utils.load_frame_labels."""

    def test_valid_loco_labels(self, tmp_path):
        from window_classification.shared_utils import load_frame_labels
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,locomotion\n0,Walking\n1,Crawling\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["Walking", "Crawling"]

    def test_unknown_maps_to_no_locomotion(self, tmp_path):
        from window_classification.shared_utils import load_frame_labels
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,locomotion\n0,FlyingThroughAir\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["No_Locomotion"]

    def test_rmm_labels(self, tmp_path):
        from window_classification.shared_utils import load_frame_labels
        csv = tmp_path / "labels.csv"
        csv.write_text("frame,repetitive_motor\n0,Jumping\n1,Rocking\n")
        labels = load_frame_labels(str(csv), task="rmm")
        assert labels == ["Jumping", "Rocking"]

    def test_bom_csv_handled(self, tmp_path):
        """CSV with BOM (utf-8-sig) should load cleanly."""
        from window_classification.shared_utils import load_frame_labels
        csv = tmp_path / "labels.csv"
        csv.write_bytes(b"\xef\xbb\xbfframe,locomotion\n0,Running\n")
        labels = load_frame_labels(str(csv), task="loco")
        assert labels == ["Running"]


# ============================================================
# window_classification — Ovis classifier 
# ============================================================

class TestOvisWindowParser:
    """Parse methods of OvisClassifier — no GPU."""

    @pytest.fixture()
    def clf(self):
        from window_classification.window_classifier_ovis import OvisClassifier
        from window_classification.shared_utils import TASK_CONFIG
        obj = object.__new__(OvisClassifier)
        cfg = TASK_CONFIG["loco"]
        obj.task = "loco"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.max_partition = 9
        obj.random_frames = False
        obj.seed = 42
        return obj

    @pytest.fixture()
    def clf_rmm(self):
        from window_classification.window_classifier_ovis import OvisClassifier
        from window_classification.shared_utils import TASK_CONFIG
        obj = object.__new__(OvisClassifier)
        cfg = TASK_CONFIG["rmm"]
        obj.task = "rmm"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.max_partition = 9
        obj.random_frames = False
        obj.seed = 42
        return obj

    def test_multiclass_action_tag(self, clf):
        assert clf._parse_multiclass("ACTION: Walking") == "Walking"

    def test_multiclass_no_locomotion(self, clf):
        assert clf._parse_multiclass("ACTION: No_Locomotion") == "No_Locomotion"

    def test_multiclass_space_variant(self, clf):
        # "No Locomotion" should match "No_Locomotion"
        result = clf._parse_multiclass("ACTION: No Locomotion")
        assert result == "No_Locomotion"

    def test_multiclass_none_on_garbage(self, clf):
        assert clf._parse_multiclass("I am unsure") is None

    def test_multiclass_empty_string(self, clf):
        assert clf._parse_multiclass("") is None

    def test_binary_yes(self, clf):
        assert clf._parse_binary("ANSWER: YES") is True

    def test_binary_no(self, clf):
        assert clf._parse_binary("ANSWER: NO") is False

    def test_binary_bare_yes(self, clf):
        assert clf._parse_binary("YES") is True

    def test_binary_bare_no(self, clf):
        assert clf._parse_binary("NO") is False

    def test_binary_garbage(self, clf):
        assert clf._parse_binary("maybe") is None

    def test_binary_empty(self, clf):
        assert clf._parse_binary("") is None

    def test_finegrained_action_tag(self, clf):
        assert clf._parse_finegrained("ACTION: Cruising") == "Cruising"

    def test_finegrained_flap_alias_rmm(self, clf_rmm):
        assert clf_rmm._parse_finegrained("ACTION: flapping") == "Hands_flapping"

    def test_finegrained_none_on_garbage(self, clf):
        assert clf._parse_finegrained("hmm") is None

    def test_classify_multiclass_empty_frames_returns_no_label(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "No_Locomotion"
        assert conf == 0.0

    def test_classify_binary_empty_frames_returns_no_label(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"
        assert votes == []

    def test_classify_binary_yes_vote(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()])
        clf._make_grid = MagicMock(return_value=dummy_image())
        clf._call = MagicMock(return_value="ANSWER: YES")
        label, conf, _ = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "Locomotion"

    def test_classify_binary_no_vote(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()])
        clf._make_grid = MagicMock(return_value=dummy_image())
        clf._call = MagicMock(return_value="ANSWER: NO")
        label, conf, _ = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"


# ============================================================
# window_classification — Qwen classifier 
# ============================================================

class TestQwenWindowParser:
    """Parse methods of QwenClassifier — no GPU."""

    @pytest.fixture()
    def clf(self):
        from window_classification.window_classifier_qwen import QwenClassifier
        from window_classification.shared_utils import TASK_CONFIG
        obj = object.__new__(QwenClassifier)
        cfg = TASK_CONFIG["loco"]
        obj.task = "loco"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.random_frames = False
        obj.seed = 42
        return obj

    @pytest.fixture()
    def clf_rmm(self):
        from window_classification.window_classifier_qwen import QwenClassifier
        from window_classification.shared_utils import TASK_CONFIG
        obj = object.__new__(QwenClassifier)
        cfg = TASK_CONFIG["rmm"]
        obj.task = "rmm"
        obj.cfg = cfg
        obj.active_classes = cfg["active_classes"]
        obj.all_classes = cfg["all_classes"]
        obj.no_label = cfg["no_action_label"]
        obj.binary_pos = cfg["binary_positive"]
        obj.num_frames = 6
        obj.random_frames = False
        obj.seed = 42
        return obj

    def test_multiclass_action_tag(self, clf):
        assert clf._parse_multiclass("ACTION: Running") == "Running"

    def test_multiclass_no_loco_label(self, clf):
        assert clf._parse_multiclass("ACTION: No_Locomotion") == "No_Locomotion"

    def test_multiclass_space_variant(self, clf):
        assert clf._parse_multiclass("ACTION: No Locomotion") == "No_Locomotion"

    def test_multiclass_none_garbage(self, clf):
        assert clf._parse_multiclass("I don't know") is None

    def test_binary_yes(self, clf):
        assert clf._parse_binary("ANSWER: YES") is True

    def test_binary_no(self, clf):
        assert clf._parse_binary("ANSWER: NO") is False

    def test_binary_garbage(self, clf):
        assert clf._parse_binary("probably") is None

    def test_finegrained_walking(self, clf):
        assert clf._parse_finegrained("ACTION: Walking") == "Walking"

    def test_finegrained_flap_rmm(self, clf_rmm):
        assert clf_rmm._parse_finegrained("child is flapping hands") == "Hands_flapping"

    def test_classify_multiclass_empty_frames(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "No_Locomotion"

    def test_classify_multiclass_majority(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ACTION: Walking",
            "ACTION: Walking",
            "ACTION: Running",
        ])
        pred, fpreds, conf = clf.classify_multiclass("v.mp4", 0.0, 2.0, 0)
        assert pred == "Walking"
        assert conf == pytest.approx(2 / 3)

    def test_classify_binary_empty_frames(self, clf):
        clf._get_frames = MagicMock(return_value=[])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"

    def test_classify_binary_majority_yes(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ANSWER: YES",
            "ANSWER: YES",
            "ANSWER: NO",
        ])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "Locomotion"
        assert conf == pytest.approx(2 / 3)

    def test_classify_binary_majority_no(self, clf):
        clf._get_frames = MagicMock(return_value=[dummy_image()] * 3)
        clf._call = MagicMock(side_effect=[
            "ANSWER: NO",
            "ANSWER: NO",
            "ANSWER: YES",
        ])
        label, conf, votes = clf.classify_binary("v.mp4", 0.0, 2.0, 0)
        assert label == "No_Locomotion"