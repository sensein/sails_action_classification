"""
Tests for src/sailsprep/action_model_testing/Videomaev2/utils/windowing.py
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "Videomaev2" / "utils" / "windowing.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("videomae2_windowing", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


class TestGetWindowLabel:
    def test_majority_label(self, mod):
        frame_to_label = {0: "Walking", 1: "Walking", 2: "Running"}
        assert mod.get_window_label(frame_to_label, 0, 3) == "Walking"

    def test_empty_window_returns_na(self, mod):
        assert mod.get_window_label({}, 5, 5) == "N/A"

    def test_missing_frames_default_to_na(self, mod):
        assert mod.get_window_label({}, 0, 2) == "N/A"

    def test_nan_like_strings_treated_as_na(self, mod):
        frame_to_label = {0: "nan", 1: "None", 2: ""}
        assert mod.get_window_label(frame_to_label, 0, 3) == "N/A"

    def test_custom_na_label(self, mod):
        assert mod.get_window_label({}, 0, 1, na_label="UNKNOWN") == "UNKNOWN"
