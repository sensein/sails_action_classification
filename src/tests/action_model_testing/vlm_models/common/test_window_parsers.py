"""
Tests for src/sailsprep/action_model_testing/vlm_models/common/window_parsers.py
"""
from __future__ import annotations

from common.window_parsers import parse_binary, parse_finegrained, parse_multiclass

ALL_CLASSES = ["No_Locomotion", "Walking", "Cruising", "Crawling", "Running", "Vehicle"]
ACTIVE_CLASSES = ["Walking", "Cruising", "Crawling", "Running", "Vehicle"]


class TestParseMulticlass:
    def test_action_tag_exact(self):
        assert parse_multiclass("ACTION: Walking", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") == "Walking"

    def test_action_tag_space_variant(self):
        assert parse_multiclass("ACTION: No Locomotion", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") == "No_Locomotion"

    def test_fallback_keyword_in_body(self):
        assert parse_multiclass("the child is running fast", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") == "Running"

    def test_no_label_variant_matched(self):
        assert parse_multiclass("no locomotion detected", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") == "No_Locomotion"

    def test_garbage_returns_none(self):
        assert parse_multiclass("I am unsure", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") is None

    def test_empty_returns_none(self):
        assert parse_multiclass("", ALL_CLASSES, ACTIVE_CLASSES, "No_Locomotion") is None


class TestParseBinary:
    def test_answer_yes(self):
        assert parse_binary("ANSWER: YES") is True

    def test_answer_no(self):
        assert parse_binary("ANSWER: NO") is False

    def test_bare_yes(self):
        assert parse_binary("YES") is True

    def test_bare_no(self):
        assert parse_binary("NO.") is False

    def test_garbage_returns_none(self):
        assert parse_binary("maybe") is None

    def test_empty_returns_none(self):
        assert parse_binary("") is None


class TestParseFinegrained:
    def test_action_tag_exact(self):
        assert parse_finegrained("ACTION: Cruising", ACTIVE_CLASSES, task="loco") == "Cruising"

    def test_flap_alias_rmm(self):
        assert parse_finegrained("ACTION: flapping", ["Jumping", "Hands_flapping", "Rocking", "Spinning"], task="rmm") == "Hands_flapping"

    def test_flap_alias_body_rmm(self):
        assert parse_finegrained("child is flapping hands", ["Jumping", "Hands_flapping"], task="rmm") == "Hands_flapping"

    def test_garbage_returns_none(self):
        assert parse_finegrained("hmm", ACTIVE_CLASSES, task="loco") is None

    def test_empty_returns_none(self):
        assert parse_finegrained("", ACTIVE_CLASSES, task="loco") is None
