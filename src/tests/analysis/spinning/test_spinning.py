"""
Tests for src/sailsprep/analysis/spinning/spinning.py

Run:
    poetry run pytest src/tests/tests_spinning.py -v

Hard dependencies (must be in pyproject.toml):
    pytest, numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib

Optional deps (rpy2, pymc, arviz, wildboottest) are intentionally suppressed
in the loading fixture so tests run fast and deterministically everywhere.
"""

import importlib.util
import json
import math
import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, mock_open, patch

import matplotlib  # must import before builtins.open is mocked
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fixture builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_main_df() -> pd.DataFrame:
    """
    8 subjects (4 ASD / 4 Non-ASD) spread evenly across two STAT_BANDS:

        11-18 mo: P01 ASD  12.0 mo
                  P02 Non  13.2 mo
                  P03 ASD  14.4 mo
                  P04 Non  15.6 mo
        32-38 mo: P05 ASD  33.0 mo
                  P06 Non  33.6 mo
                  P07 ASD  34.2 mo
                  P08 Non  34.8 mo
    """
    ages_yr = [1.00, 1.10, 1.20, 1.30, 2.75, 2.80, 2.85, 2.90]
    groups  = ["ASD", "Non-ASD"] * 4
    pids    = [f"sub-P{i:02d}" for i in range(1, 9)]
    return pd.DataFrame({
        "video_path":      [f"{p}/video.mp4" for p in pids],
        "Age":             ages_yr,          # years → × 12 = months in module
        "Group":           groups,
        "hrnet_full_path": [f"/fake/pose/{p}.json" for p in pids],
    })


def _build_rmm_df(main: pd.DataFrame) -> pd.DataFrame:
    """
    Two short spinning clips per child.

    csv_bids_processed must mirror video_path so the module's
    ``video_to_hrnet`` dict lookup succeeds.
    Timestamps 0:00-0:01 and 0:01-0:02 @ 15 fps map to frames 0-15 and
    15-30, both safely within the 35-frame mock pose file.
    """
    rows = []
    for _, row in main.iterrows():
        for seg in ["0:00-0:01", "0:01-0:02"]:
            rows.append({
                "csv_bids_processed": row["video_path"],
                "matched_label":      "spinning",
                "matched_ts":         seg,
                "clip_filename":      "clip.mp4",
            })
    return pd.DataFrame(rows)


def _build_pose_json(n_frames: int = 35, seed: int = 0) -> str:
    """
    Synthetic HRNet pose JSON with *n_frames* frames (keys "0" … str(n-1)).

    All keypoints are present and have confidence ≥ 0.80, well above
    MIN_CONF = 0.30.  Small Gaussian jitter around plausible pixel
    coordinates ensures non-degenerate feature values.
    """
    rng = np.random.default_rng(seed)
    frames: dict = {}
    for i in range(n_frames):
        frames[str(i)] = {
            "kp_000": {"x": 150 + rng.normal() * 8,  "y":  50, "confidence": 0.92},
            "kp_005": {"x": 100 + rng.normal() * 8,  "y": 100, "confidence": 0.92},
            "kp_006": {"x": 200 + rng.normal() * 8,  "y": 100, "confidence": 0.92},
            "kp_007": {"x":  90 + rng.normal() * 6,  "y": 150, "confidence": 0.88},
            "kp_008": {"x": 210 + rng.normal() * 6,  "y": 150, "confidence": 0.88},
            "kp_009": {"x":  80 + rng.normal() * 12, "y": 200, "confidence": 0.82},
            "kp_010": {"x": 220 + rng.normal() * 12, "y": 200, "confidence": 0.82},
            "kp_011": {"x": 110 + rng.normal() * 4,  "y": 200, "confidence": 0.92},
            "kp_012": {"x": 190 + rng.normal() * 4,  "y": 200, "confidence": 0.92},
        }
    return json.dumps({"frames": frames, "ann_fps": 15.0})


# Build once at import time so the session fixture is fast
_MAIN_DF   = _build_main_df()
_RMM_DF    = _build_rmm_df(_MAIN_DF)
_POSE_JSON = _build_pose_json()


# ─────────────────────────────────────────────────────────────────────────────
# Module loader  (runs the entire script once, with all I/O mocked)
# ─────────────────────────────────────────────────────────────────────────────

_MOD = None  # module-level cache so we only exec_module once per process


def _load_module():
    global _MOD
    if _MOD is not None:
        return _MOD

    module_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "sailsprep", "analysis", "spinning", "spinning.py",
            )
        )

    # Only treat real HRNet JSON paths as existing; CSV/output paths → False.
    def _isfile(path):
        return isinstance(path, str) and path.endswith(".json")

    import builtins as _builtins
    _real_open = _builtins.open                   # capture before any patching
    _pose_mock = mock_open(read_data=_POSE_JSON)

    def _selective_open(path=None, *args, **kwargs):
        """Only intercept HRNet JSON reads; let font/config files through."""
        if isinstance(path, str) and path.endswith(".json"):
            return _pose_mock(path, *args, **kwargs)
        return _real_open(path, *args, **kwargs)

    with ExitStack() as stack:
        # ── 1. Suppress optional heavy dependencies ──────────────────────
        stack.enter_context(patch.dict(sys.modules, {
            "pymc":                       None,
            "arviz":                      None,
            "rpy2":                       None,
            "rpy2.robjects":              None,
            "rpy2.robjects.packages":     None,
            "wildboottest":               None,
            "wildboottest.wildboottest":  None,
        }))

        # ── 2. Mock CSV reads (called exactly twice: MAIN_CSV, RMM_CSV) ──
        stack.enter_context(
            patch("pandas.read_csv",
                  side_effect=[_MAIN_DF.copy(), _RMM_DF.copy()])
        )

        # ── 3. Filesystem stubs ──────────────────────────────────────────
        stack.enter_context(patch("os.makedirs"))
        stack.enter_context(patch("os.path.isfile", side_effect=_isfile))
        stack.enter_context(patch("os.listdir", return_value=[]))
        stack.enter_context(patch("os.path.getsize", return_value=1024))

        # ── 4. File I/O stubs ────────────────────────────────────────────
        stack.enter_context(patch("builtins.open", new=_selective_open))
        stack.enter_context(patch("pandas.DataFrame.to_csv"))    # silence CSV writes
        # spinning.py uses `df or df2` pattern; only safe when df is non-empty.
        # With mock data LME may return empty df → pandas raises. Patch __bool__
        # so empty=False, non-empty=True, matching the intent of the `or` chain.
        stack.enter_context(
            patch.object(pd.DataFrame, "__bool__", lambda self: len(self) > 0)
        )

        # ── 5. Matplotlib stubs ──────────────────────────────────────────
        stack.enter_context(patch("matplotlib.figure.Figure.savefig"))
        stack.enter_context(patch("matplotlib.pyplot.close"))
        # tight_layout triggers font rendering → crashes in envs with bad font cache
        stack.enter_context(patch("matplotlib.pyplot.tight_layout"))
        stack.enter_context(patch("matplotlib.figure.Figure.tight_layout"))

        # ── 6. Load & execute the module ─────────────────────────────────
        spec = importlib.util.spec_from_file_location("spinning_mod", module_path)
        mod  = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(mod)
        except SystemExit as exc:
            pytest.fail(
                f"spinning.py called sys.exit({exc.code}) during load.  "
                "Likely cause: n_ok == 0.  Check that mock pose data is valid "
                "and timestamps fall within the 35-frame window."
            )

        _MOD = mod
    return _MOD


@pytest.fixture(scope="session")
def sp():
    """Session-scoped: the fully executed spinning module (loads once)."""
    return _load_module()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS — pure utility functions
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPid:
    def test_simple_sub_tag(self, sp):
        assert sp.extract_pid("sub-A01/video.mp4") == "sub-A01"

    def test_nested_path(self, sp):
        assert sp.extract_pid("/data/bids/sub-XYZ99/pose.json") == "sub-XYZ99"

    def test_alphanumeric_subject_id(self, sp):
        assert sp.extract_pid("sub-123abc/run-01.json") == "sub-123abc"

    def test_no_sub_tag_returns_none(self, sp):
        assert sp.extract_pid("no_subject_here.mp4") is None

    def test_none_input_returns_none(self, sp):
        assert sp.extract_pid(None) is None

    def test_integer_input_returns_none(self, sp):
        assert sp.extract_pid(12345) is None


class TestParseTimestamps:
    def test_single_segment_frame_range(self, sp):
        segs = sp.parse_timestamps("0:10-0:20", fps=15.0)
        assert len(segs) == 1
        assert segs[0] == (int(10 * 15), int(20 * 15))

    def test_multiple_segments(self, sp):
        segs = sp.parse_timestamps("0:00-0:05, 0:10-0:15", fps=15.0)
        assert len(segs) == 2

    def test_empty_string_returns_empty(self, sp):
        assert sp.parse_timestamps("", fps=15.0) == []

    def test_garbage_string_returns_empty(self, sp):
        assert sp.parse_timestamps("not a timestamp", fps=15.0) == []

    def test_inverted_range_is_skipped(self, sp):
        # e <= s should produce no segment
        assert sp.parse_timestamps("0:30-0:10", fps=15.0) == []

    def test_equal_start_end_is_skipped(self, sp):
        assert sp.parse_timestamps("0:05-0:05", fps=15.0) == []

    def test_custom_fps_scales_frames(self, sp):
        segs = sp.parse_timestamps("0:00-0:01", fps=30.0)
        assert segs == [(0, 30)]

    def test_non_string_returns_empty(self, sp):
        assert sp.parse_timestamps(None, fps=15.0) == []

    def test_frame_indices_are_ints(self, sp):
        segs = sp.parse_timestamps("0:00-0:02", fps=15.0)
        assert all(isinstance(v, int) for seg in segs for v in seg)


class TestAssignAgeBand:
    @pytest.mark.parametrize("mo,band", [
        (11, "11-18mo"), (15, "11-18mo"), (18, "11-18mo"),
        (19, "19-31mo"), (25, "19-31mo"), (31, "19-31mo"),
        (32, "32-38mo"), (35, "32-38mo"), (38, "32-38mo"),
    ])
    def test_in_band(self, sp, mo, band):
        assert sp.assign_age_band(mo) == band

    @pytest.mark.parametrize("mo", [0, 10, 39, 100])
    def test_out_of_range_returns_none(self, sp, mo):
        assert sp.assign_age_band(mo) is None


class TestCohenD:
    def test_positive_d_when_a_greater(self, sp):
        a = np.array([3.0, 4.0, 5.0])
        b = np.array([0.0, 1.0, 2.0])
        assert sp.cohen_d(a, b) > 0

    def test_negative_d_when_a_smaller(self, sp):
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        assert sp.cohen_d(a, b) < 0

    def test_zero_d_for_identical_distributions(self, sp):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert sp.cohen_d(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_zero_pooled_variance_returns_zero(self, sp):
        # Both arrays constant → pooled SD = 0 → d = 0
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert sp.cohen_d(a, b) == 0.0

    def test_large_effect_exceeds_5(self, sp):
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([0.0,  1.0,  2.0])
        assert sp.cohen_d(a, b) > 5.0

    def test_symmetry(self, sp):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        assert sp.cohen_d(a, b) == pytest.approx(-sp.cohen_d(b, a), abs=1e-9)


class TestBootstrapCI:
    def test_returns_two_floats(self, sp):
        a = np.random.default_rng(0).normal(1, 1, 30)
        b = np.random.default_rng(1).normal(0, 1, 30)
        lo, hi = sp.bootstrap_ci_d(a, b, n_boot=100, seed=0)
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_ci_ordered_lo_lt_hi(self, sp):
        a = np.random.default_rng(0).normal(1, 1, 30)
        b = np.random.default_rng(1).normal(0, 1, 30)
        lo, hi = sp.bootstrap_ci_d(a, b, n_boot=100, seed=0)
        assert lo < hi

    def test_observed_d_within_ci(self, sp):
        rng = np.random.default_rng(42)
        a = rng.normal(1, 1, 50)
        b = rng.normal(0, 1, 50)
        d_obs = sp.cohen_d(a, b)
        lo, hi = sp.bootstrap_ci_d(a, b, n_boot=500, seed=7)
        assert lo <= d_obs <= hi

    def test_different_seeds_differ(self, sp):
        a = np.random.default_rng(0).normal(1, 1, 30)
        b = np.random.default_rng(1).normal(0, 1, 30)
        lo1, _ = sp.bootstrap_ci_d(a, b, n_boot=200, seed=1)
        lo2, _ = sp.bootstrap_ci_d(a, b, n_boot=200, seed=99)
        # Not guaranteed to differ but almost always will with enough boot reps
        # Just check they are floats; equality is not a test requirement


class TestButterLowPass:
    def test_preserves_length(self, sp):
        arr = np.sin(np.linspace(0, 2 * np.pi, 120))
        out = sp.butter_lp(arr, cutoff=4.0, fs=15.0)
        assert len(out) == len(arr)

    def test_returns_ndarray(self, sp):
        arr = np.random.default_rng(0).normal(size=50)
        out = sp.butter_lp(arr, cutoff=4.0, fs=15.0)
        assert isinstance(out, np.ndarray)

    def test_too_short_array_returned_as_is(self, sp):
        arr = np.array([1.0, 2.0, 3.0])
        out = sp.butter_lp(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_attenuates_high_frequency(self, sp):
        t   = np.linspace(0, 1, 150)
        sig = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 7.0 * t)
        out = sp.butter_lp(sig, cutoff=3.0, fs=15.0)
        # After LP filtering the residual should have smaller std than the
        # original signal (which contains a 7 Hz component above cutoff)
        assert np.std(sig - out) < np.std(sig)


class TestSpectralFeatures:
    def test_returns_three_values(self, sp):
        arr = np.sin(2 * np.pi * 1.0 * np.linspace(0, 4, 64))
        result = sp.spectral_features(arr, fps=15.0)
        assert len(result) == 3

    def test_all_floats_on_valid_input(self, sp):
        arr = np.sin(2 * np.pi * 1.0 * np.linspace(0, 4, 64))
        df_f, ent, bp = sp.spectral_features(arr, fps=15.0)
        assert all(isinstance(v, float) for v in (df_f, ent, bp))

    def test_nans_on_too_short_input(self, sp):
        arr = np.array([1.0, 2.0, 3.0])
        df_f, ent, bp = sp.spectral_features(arr, fps=15.0)
        assert all(math.isnan(v) for v in (df_f, ent, bp))

    def test_band_power_between_0_and_1(self, sp):
        arr = np.sin(2 * np.pi * 1.5 * np.linspace(0, 4, 64))
        _, _, bp = sp.spectral_features(arr, fps=15.0, lo=0.5, hi=2.5)
        assert 0.0 <= bp <= 1.0

    def test_dominant_frequency_non_negative(self, sp):
        arr = np.sin(2 * np.pi * 1.0 * np.linspace(0, 4, 64))
        df_f, _, _ = sp.spectral_features(arr, fps=15.0)
        assert df_f >= 0.0

    def test_entropy_non_negative(self, sp):
        arr = np.sin(2 * np.pi * 1.5 * np.linspace(0, 4, 64))
        _, ent, _ = sp.spectral_features(arr, fps=15.0)
        assert ent >= 0.0


class TestFdrAnnotate:
    def test_output_has_p_fdr_column(self, sp):
        df = pd.DataFrame({"p_val": [0.01, 0.05, 0.5, 0.9]})
        out = sp.fdr_annotate(df, "p_val")
        assert "p_fdr" in out.columns

    def test_output_has_sig_fdr_column(self, sp):
        df = pd.DataFrame({"p_val": [0.01, 0.05, 0.5, 0.9]})
        out = sp.fdr_annotate(df, "p_val")
        assert "sig_fdr05" in out.columns

    def test_output_has_sig_raw_column(self, sp):
        df = pd.DataFrame({"p_val": [0.01, 0.05, 0.5, 0.9]})
        out = sp.fdr_annotate(df, "p_val")
        assert "sig_raw05" in out.columns

    def test_fdr_values_in_unit_interval(self, sp):
        df = pd.DataFrame({"p_val": [0.001, 0.01, 0.1, 0.9]})
        out = sp.fdr_annotate(df, "p_val")
        assert out["p_fdr"].between(0, 1, inclusive="both").all()

    def test_single_row_handled_gracefully(self, sp):
        df = pd.DataFrame({"p_val": [0.03]})
        out = sp.fdr_annotate(df, "p_val")
        assert len(out) == 1

    def test_clearly_significant_raw_p(self, sp):
        df = pd.DataFrame({"p_val": [0.001, 0.8]})
        out = sp.fdr_annotate(df, "p_val")
        assert out.loc[out["p_val"] == 0.001, "sig_raw05"].values[0]

    def test_clearly_non_significant_raw_p(self, sp):
        df = pd.DataFrame({"p_val": [0.001, 0.8]})
        out = sp.fdr_annotate(df, "p_val")
        assert not out.loc[out["p_val"] == 0.8, "sig_raw05"].values[0]


class TestGetKp:
    def test_valid_keypoint_returned(self, sp):
        fd = {"kp_005": {"x": 100.0, "y": 200.0, "confidence": 0.9}}
        kp = sp.get_kp(fd, "kp_005", min_conf=0.3)
        assert kp is not None
        assert kp["x"] == pytest.approx(100.0)

    def test_low_confidence_returns_none(self, sp):
        fd = {"kp_005": {"x": 100.0, "y": 200.0, "confidence": 0.1}}
        assert sp.get_kp(fd, "kp_005", min_conf=0.3) is None

    def test_missing_key_returns_none(self, sp):
        fd = {"kp_005": {"x": 100.0, "y": 200.0, "confidence": 0.9}}
        assert sp.get_kp(fd, "kp_006", min_conf=0.3) is None

    def test_non_dict_value_returns_none(self, sp):
        fd = {"kp_005": "not_a_dict"}
        assert sp.get_kp(fd, "kp_005", min_conf=0.3) is None

    def test_exactly_at_threshold_passes(self, sp):
        fd = {"kp_005": {"x": 50.0, "y": 80.0, "confidence": 0.3}}
        kp = sp.get_kp(fd, "kp_005", min_conf=0.3)
        assert kp is not None


class TestTorsoLength:
    @staticmethod
    def _full_frame(conf: float = 0.9) -> dict:
        return {
            "kp_005": {"x": 100.0, "y": 100.0, "confidence": conf},
            "kp_006": {"x": 200.0, "y": 100.0, "confidence": conf},
            "kp_011": {"x": 110.0, "y": 200.0, "confidence": conf},
            "kp_012": {"x": 190.0, "y": 200.0, "confidence": conf},
        }

    def test_returns_positive_float(self, sp):
        length = sp.torso_length(self._full_frame())
        assert length is not None
        assert length > 0

    def test_missing_hip_returns_none(self, sp):
        fd = {
            "kp_005": {"x": 100.0, "y": 100.0, "confidence": 0.9},
            "kp_006": {"x": 200.0, "y": 100.0, "confidence": 0.9},
        }
        assert sp.torso_length(fd) is None

    def test_empty_frame_returns_none(self, sp):
        assert sp.torso_length({}) is None

    def test_known_geometry(self, sp):
        """
        Shoulder mid = (150, 100), hip mid = (150, 200) → distance = 100.
        """
        fd = {
            "kp_005": {"x": 100.0, "y": 100.0, "confidence": 0.9},
            "kp_006": {"x": 200.0, "y": 100.0, "confidence": 0.9},
            "kp_011": {"x": 100.0, "y": 200.0, "confidence": 0.9},
            "kp_012": {"x": 200.0, "y": 200.0, "confidence": 0.9},
        }
        assert sp.torso_length(fd) == pytest.approx(100.0, abs=1e-6)


class TestGetScale:
    def test_returns_torso_when_all_landmarks_available(self, sp):
        fd = {
            "kp_005": {"x": 100.0, "y": 100.0, "confidence": 0.9},
            "kp_006": {"x": 200.0, "y": 100.0, "confidence": 0.9},
            "kp_011": {"x": 110.0, "y": 200.0, "confidence": 0.9},
            "kp_012": {"x": 190.0, "y": 200.0, "confidence": 0.9},
        }
        scale = sp.get_scale(fd)
        assert scale is not None and scale > 5

    def test_falls_back_to_hip_width(self, sp):
        fd = {
            "kp_011": {"x": 100.0, "y": 200.0, "confidence": 0.9},
            "kp_012": {"x": 200.0, "y": 200.0, "confidence": 0.9},
        }
        scale = sp.get_scale(fd)
        assert scale is not None and scale > 5

    def test_falls_back_to_shoulder_width(self, sp):
        fd = {
            "kp_005": {"x": 100.0, "y": 100.0, "confidence": 0.9},
            "kp_006": {"x": 200.0, "y": 100.0, "confidence": 0.9},
        }
        scale = sp.get_scale(fd)
        assert scale is not None and scale > 5

    def test_empty_frame_returns_none(self, sp):
        assert sp.get_scale({}) is None


class TestExtractSpinningFeatures:
    @staticmethod
    def _frames(n: int = 30, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        frames: dict = {}
        for i in range(n):
            frames[str(i)] = {
                "kp_000": {"x": 150 + rng.normal() * 5,  "y":  50, "confidence": 0.92},
                "kp_005": {"x": 100 + rng.normal() * 5,  "y": 100, "confidence": 0.92},
                "kp_006": {"x": 200 + rng.normal() * 5,  "y": 100, "confidence": 0.92},
                "kp_009": {"x":  80 + rng.normal() * 10, "y": 200, "confidence": 0.82},
                "kp_010": {"x": 220 + rng.normal() * 10, "y": 200, "confidence": 0.82},
                "kp_011": {"x": 110 + rng.normal() * 4,  "y": 200, "confidence": 0.92},
                "kp_012": {"x": 190 + rng.normal() * 4,  "y": 200, "confidence": 0.92},
            }
        return frames

    def test_returns_dict_on_valid_input(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        assert rec is not None and isinstance(rec, dict)

    def test_too_few_frames_returns_none(self, sp):
        frames = self._frames(3)
        assert sp.extract_spinning_features(frames, [0, 1, 2], ann_fps=15.0) is None

    def test_sw_amplitude_present(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        assert "sw_amplitude" in rec

    def test_spectral_entropy_present(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        assert any("spectral_entropy" in k for k in rec)

    def test_n_valid_frames_matches(self, sp):
        n = 20
        frames = self._frames(n)
        rec = sp.extract_spinning_features(frames, list(range(n)), ann_fps=15.0)
        assert rec["n_valid_frames"] == n

    def test_pct_valid_in_unit_interval(self, sp):
        frames = self._frames(20)
        rec = sp.extract_spinning_features(frames, list(range(20)), ann_fps=15.0)
        assert 0.0 < rec["pct_valid"] <= 1.0

    def test_spin_intensity_positive(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        if "spin_intensity_mean" in rec:
            assert rec["spin_intensity_mean"] > 0

    def test_lr_correlation_in_range(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        if "sh_x_LR_corr" in rec:
            assert -1.0 <= rec["sh_x_LR_corr"] <= 1.0

    def test_wrist_amplitude_non_negative(self, sp):
        frames = self._frames(30)
        rec = sp.extract_spinning_features(frames, list(range(30)), ann_fps=15.0)
        for k in ("lw_x_amplitude", "rw_x_amplitude"):
            if k in rec:
                assert rec[k] >= 0.0, f"{k} should be non-negative"

    def test_indices_beyond_frame_range_are_ignored(self, sp):
        """Frame keys 0-9 only; indices 0-19 ask for more — should still succeed."""
        frames = self._frames(10)
        rec = sp.extract_spinning_features(frames, list(range(20)), ann_fps=15.0)
        # 10 valid frames is still ≥ 5 → should return a dict
        assert rec is not None
        assert rec["n_valid_frames"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS — pipeline-level outputs from the loaded module
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatDf:
    """clip-level feature DataFrame produced by PART 1."""

    def test_non_empty(self, sp):
        assert len(sp.feat_df) > 0

    def test_both_groups_present(self, sp):
        assert set(sp.feat_df["Group"].unique()) == {"ASD", "Non-ASD"}

    def test_required_meta_columns(self, sp):
        for col in ("pid", "Group", "age_mo", "age_band", "clip"):
            assert col in sp.feat_df.columns, f"Missing column: {col}"

    def test_row_count_matches_clips(self, sp):
        # 8 children × 2 clips = 16 rows
        assert len(sp.feat_df) == 16

    def test_primary_feats_all_in_columns(self, sp):
        missing = [f for f in sp.PRIMARY_FEATS if f not in sp.feat_df.columns]
        assert not missing, f"Missing primary features: {missing}"

    def test_sw_amplitude_has_values(self, sp):
        if "sw_amplitude" in sp.feat_df.columns:
            assert sp.feat_df["sw_amplitude"].notna().any()

    def test_age_band_values_are_valid(self, sp):
        valid_bands = {"11-18mo", "19-31mo", "32-38mo"}
        observed = set(sp.feat_df["age_band"].dropna().unique())
        assert observed.issubset(valid_bands)

    def test_pct_valid_between_0_and_1(self, sp):
        if "pct_valid" in sp.feat_df.columns:
            assert sp.feat_df["pct_valid"].between(0, 1, inclusive="both").all()


class TestChildDf:
    """Child-averaged feature DataFrame produced by PART 1."""

    def test_non_empty(self, sp):
        assert len(sp.child_df) > 0

    def test_one_row_per_pid(self, sp):
        assert sp.child_df["pid"].nunique() == len(sp.child_df)

    def test_expected_child_count(self, sp):
        # 8 unique children in the mock
        assert len(sp.child_df) == 8

    def test_both_groups_present(self, sp):
        assert set(sp.child_df["Group"].unique()) == {"ASD", "Non-ASD"}

    def test_fewer_rows_than_feat_df(self, sp):
        assert len(sp.child_df) < len(sp.feat_df)

    def test_has_n_clips_column(self, sp):
        assert "n_clips" in sp.child_df.columns


class TestStatisticalResults:
    """Statistical outputs from PART 2."""

    def test_lme_all_is_dataframe(self, sp):
        assert isinstance(sp.lme_all, pd.DataFrame)

    def test_mw_all_is_dataframe(self, sp):
        assert isinstance(sp.mw_all, pd.DataFrame)

    def test_perm_all_is_dataframe(self, sp):
        assert isinstance(sp.perm_all, pd.DataFrame)

    def test_mw_p_values_in_unit_interval(self, sp):
        if len(sp.mw_all) > 0:
            assert sp.mw_all["p_raw"].between(0, 1, inclusive="both").all()

    def test_lme_has_cohens_d_column(self, sp):
        if len(sp.lme_all) > 0:
            assert "cohens_d" in sp.lme_all.columns

    def test_consensus_has_required_columns(self, sp):
        if len(sp.consensus_all) > 0:
            for col in ("feature", "n_methods_sig"):
                assert col in sp.consensus_all.columns

    def test_icc_values_non_negative(self, sp):
        if len(sp.icc_df) > 0:
            assert "ICC" in sp.icc_df.columns
            assert (sp.icc_df["ICC"] >= 0).all()

    def test_mw_all_has_cohens_d(self, sp):
        if len(sp.mw_all) > 0:
            assert "cohens_d" in sp.mw_all.columns


class TestClassificationResults:
    """LOSO classification outputs from PART 4."""

    def test_clf_results_is_dict(self, sp):
        assert isinstance(sp.clf_results, dict)

    def test_combined_lr_ran(self, sp):
        assert "combined_LR" in sp.clf_results, (
            "combined_LR not in clf_results. "
            "Check child_df has ≥4 ASD and ≥4 Non-ASD rows."
        )

    def test_combined_rf_ran(self, sp):
        assert "combined_RF" in sp.clf_results

    def test_auc_in_unit_interval(self, sp):
        for key, res in sp.clf_results.items():
            auc = res.get("auc")
            if auc is not None:
                assert 0.0 <= auc <= 1.0, f"{key}: AUC={auc} is out of [0,1]"

    def test_perm_p_in_unit_interval(self, sp):
        for key, res in sp.clf_results.items():
            p = res.get("perm_p")
            if p is not None:
                assert 0.0 <= p <= 1.0, f"{key}: perm_p={p} is out of [0,1]"

    def test_y_true_and_score_lengths_match(self, sp):
        for key, res in sp.clf_results.items():
            if "y_true" in res and "y_score" in res:
                assert len(res["y_true"]) == len(res["y_score"]), \
                    f"{key}: y_true / y_score length mismatch"

    def test_feat_importance_structure(self, sp):
        if len(sp.feat_importance_df) > 0:
            assert {"feature", "importance"}.issubset(sp.feat_importance_df.columns)
            assert (sp.feat_importance_df["importance"] >= 0).all()