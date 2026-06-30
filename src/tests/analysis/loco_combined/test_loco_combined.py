"""
Tests for loco_combined.py — pure utility functions.

Uses AST extraction to load only function definitions + constants
without executing the top-level script (which requires real data files).
os.makedirs is patched during exec() to avoid touching the read-only /orcd filesystem.
"""
from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
    
# ---------------------------------------------------------------------------
# Locate source file
# ---------------------------------------------------------------------------
def _find_src_dir(start: Path) -> Path:
    d = start
    while d.name != "src":
        if d.parent == d:
            raise RuntimeError(f"Could not locate 'src' directory above {start}")
        d = d.parent
    return d

_SRC = (
    _find_src_dir(Path(__file__).resolve().parent)
    / "sailsprep" / "analysis" / "loco_combined" / "loco_combined.py"
)
# ---------------------------------------------------------------------------
# AST-based loader: imports + constants + all function defs, no script body
# ---------------------------------------------------------------------------
def _load_namespace() -> dict[str, Any]:
    """
    Parse loco_combined.py with AST.
    Returns a namespace containing only imports, constants, and function defs.
    os.makedirs is mocked to prevent touching the read-only /orcd filesystem.
    """
    if not _SRC.exists():
        pytest.skip(f"Source not found: {_SRC}")

    with open(_SRC, encoding="utf-8") as fh:
        source = fh.read()

    tree = ast.parse(source)

    # Find boundary: first call to hr(...)
    boundary = float("inf")
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "hr"
        ):
            boundary = node.lineno
            break

    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            selected.append(node)   # always include all function defs
        elif node.lineno < boundary:
            selected.append(node)   # everything before first hr() call

    new_mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(new_mod)
    code = compile(new_mod, str(_SRC), "exec")

    import matplotlib
    matplotlib.use("Agg")

    ns: dict[str, Any] = {}
    # Patch os.makedirs so the module-level makedirs(OUTPUT_DIR) doesn't fail
    # on the read-only /orcd filesystem.
    with mock.patch("os.makedirs"):
        exec(code, ns)  # noqa: S102
    return ns


@pytest.fixture(scope="module")
def lc() -> dict[str, Any]:
    """Module-scoped fixture: the loco_combined function namespace."""
    return _load_namespace()


# ===========================================================================
# extract_pid / extract_session
# ===========================================================================

class TestExtractPid:
    def test_valid_path(self, lc):
        assert lc["extract_pid"]("/data/sub-ABC123/ses-01/video.mp4") == "sub-ABC123"

    def test_valid_underscore(self, lc):
        assert lc["extract_pid"]("/data/sub-XYZ/file.json") == "sub-XYZ"

    def test_no_match(self, lc):
        assert lc["extract_pid"]("/data/participant/video.mp4") is None

    def test_non_string_none(self, lc):
        assert lc["extract_pid"](None) is None

    def test_non_string_int(self, lc):
        assert lc["extract_pid"](42) is None

    def test_multiple_subs_first_match(self, lc):
        # re.search returns first match
        result = lc["extract_pid"]("/data/sub-AAA/sub-BBB/file")
        assert result == "sub-AAA"


class TestExtractSession:
    def test_valid(self, lc):
        assert lc["extract_session"]("/data/sub-X/ses-03/video.mp4") == 3

    def test_two_digit(self, lc):
        assert lc["extract_session"]("/data/ses-12/x") == 12

    def test_no_match(self, lc):
        assert lc["extract_session"]("/data/session_1/x") is None

    def test_non_string(self, lc):
        assert lc["extract_session"](None) is None


# ===========================================================================
# compute_angle
# ===========================================================================

class TestComputeAngle:
    def test_right_angle(self, lc):
        # L-shape: p1=(1,0) p2=(0,0) p3=(0,1) → 90°
        angle = lc["compute_angle"]([1, 0], [0, 0], [0, 1])
        assert abs(angle - 90.0) < 1e-6

    def test_straight_line(self, lc):
        # Collinear: p1=(-1,0) p2=(0,0) p3=(1,0) → ~180°
        # The +1e-8 fudge in the denominator shifts the result slightly,
        # so we allow a tolerance of 0.01 degrees.
        angle = lc["compute_angle"]([-1, 0], [0, 0], [1, 0])
        assert abs(angle - 180.0) < 0.01

    def test_45_degrees(self, lc):
        angle = lc["compute_angle"]([1, 0], [0, 0], [1, 1])
        assert abs(angle - 45.0) < 1e-5

    def test_returns_float(self, lc):
        result = lc["compute_angle"]([1, 0], [0, 0], [0, 1])
        assert isinstance(result, float)

    def test_symmetric(self, lc):
        a = lc["compute_angle"]([1, 0], [0, 0], [0, 1])
        b = lc["compute_angle"]([0, 1], [0, 0], [1, 0])
        assert abs(a - b) < 1e-9


# ===========================================================================
# butter_lp
# ===========================================================================

class TestButterLp:
    def test_output_shape(self, lc):
        data = np.random.default_rng(0).random(50)
        out = lc["butter_lp"](data, cutoff=4.0, fs=15.0)
        assert len(out) == len(data)

    def test_short_array_passthrough(self, lc):
        # len < 12 → returned as-is
        data = np.array([1.0, 2.0, 3.0])
        out = lc["butter_lp"](data)
        np.testing.assert_array_equal(out, data)

    def test_attenuates_high_freq(self, lc):
        # Use arange/fs to generate samples at exactly 15 fps — linspace would
        # produce the wrong effective sample rate and defeat the filter test.
        t = np.arange(120) / 15.0          # 8 s at 15 fps = 120 samples
        low  = np.sin(2 * np.pi * 1.0 * t) # 1 Hz — well below 4 Hz cutoff
        high = np.sin(2 * np.pi * 6.0 * t) # 6 Hz — above cutoff (fs=15, cutoff=4)
        filtered_low  = lc["butter_lp"](low,  cutoff=4.0, fs=15.0)
        filtered_high = lc["butter_lp"](high, cutoff=4.0, fs=15.0)
        # Low-freq component should pass through largely intact
        assert np.var(filtered_low) > 0.3
        # High-freq component should be heavily suppressed
        assert np.var(filtered_high) < 0.05

    def test_returns_ndarray(self, lc):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = lc["butter_lp"](data)
        assert isinstance(out, np.ndarray)


# ===========================================================================
# dominant_freq
# ===========================================================================

class TestDominantFreq:
    def test_known_frequency(self, lc):
        fps = 15.0
        target = 2.0
        t = np.arange(0, 128) / fps
        arr = np.sin(2 * np.pi * target * t)
        freq = lc["dominant_freq"](arr, fps=fps)
        assert abs(freq - target) < 0.5

    def test_short_returns_nan(self, lc):
        # len < 16 → nan
        assert math.isnan(lc["dominant_freq"](np.array([1.0, 2.0])))

    def test_returns_float(self, lc):
        arr = np.sin(np.linspace(0, 4 * np.pi, 64))
        result = lc["dominant_freq"](arr, fps=15.0)
        assert isinstance(result, float)

    def test_positive_freq(self, lc):
        arr = np.sin(np.linspace(0, 8 * np.pi, 128))
        freq = lc["dominant_freq"](arr, fps=15.0)
        assert freq >= 0


# ===========================================================================
# spectral_entropy
# ===========================================================================

class TestSpectralEntropy:
    def test_short_returns_nan(self, lc):
        assert math.isnan(lc["spectral_entropy"](np.array([1.0, 2.0])))

    def test_returns_float(self, lc):
        arr = np.random.default_rng(1).random(64)
        result = lc["spectral_entropy"](arr, fps=15.0)
        assert isinstance(result, float)

    def test_noise_higher_than_sinusoid(self, lc):
        rng   = np.random.default_rng(42)
        noise = rng.random(128)
        pure  = np.sin(np.linspace(0, 8 * np.pi, 128))
        se_noise = lc["spectral_entropy"](noise, fps=15.0)
        se_pure  = lc["spectral_entropy"](pure,  fps=15.0)
        assert se_noise > se_pure

    def test_nonnegative(self, lc):
        arr = np.random.default_rng(7).random(64)
        assert lc["spectral_entropy"](arr, fps=15.0) >= 0


# ===========================================================================
# jerk_cost
# ===========================================================================

# jerk_cost uses np.trapz which was removed in NumPy 2.0 (renamed np.trapezoid).
# Skip the tests that actually call it if running on NumPy >= 2.0.
_TRAPZ_OK = hasattr(np, 'trapz')
_skip_trapz = pytest.mark.skipif(not _TRAPZ_OK, reason="np.trapz removed in NumPy 2.0")


class TestJerkCost:
    def test_short_returns_nan(self, lc):
        # len < 6 → returns nan before ever reaching np.trapz
        assert math.isnan(lc["jerk_cost"](np.array([1.0, 2.0])))

    def test_constant_returns_nan(self, lc):
        # amp < 1e-8 → returns nan before ever reaching np.trapz
        arr = np.ones(30)
        assert math.isnan(lc["jerk_cost"](arr))

    @_skip_trapz
    def test_smooth_lower_than_noisy(self, lc):
        fps  = 15.0
        t    = np.linspace(0, 2, 60)
        rng  = np.random.default_rng(0)
        smooth = np.sin(2 * np.pi * t)
        noisy  = smooth + rng.normal(0, 1.0, len(t))
        jc_smooth = lc["jerk_cost"](smooth, fps=fps)
        jc_noisy  = lc["jerk_cost"](noisy,  fps=fps)
        assert jc_smooth < jc_noisy

    @_skip_trapz
    def test_returns_float(self, lc):
        arr = np.sin(np.linspace(0, 4 * np.pi, 30))
        assert isinstance(lc["jerk_cost"](arr), float)


# ===========================================================================
# ac_strength
# ===========================================================================

class TestAcStrength:
    def test_short_returns_nan(self, lc):
        # len < 4 → nan
        assert math.isnan(lc["ac_strength"](np.array([1.0])))

    def test_periodic_high(self, lc):
        arr = np.sin(np.linspace(0, 6 * np.pi, 64))
        ac  = lc["ac_strength"](arr)
        assert ac > 0.5

    def test_random_low(self, lc):
        rng = np.random.default_rng(99)
        arr = rng.random(200)
        ac  = lc["ac_strength"](arr)
        assert abs(ac) < 0.3

    def test_constant_returns_nan(self, lc):
        # denom (dot(arr,arr)) ≈ 0 after mean-subtraction → nan
        arr = np.ones(10)
        assert math.isnan(lc["ac_strength"](arr))

    def test_range(self, lc):
        arr = np.sin(np.linspace(0, 4 * np.pi, 50))
        ac  = lc["ac_strength"](arr)
        assert -1.0 <= ac <= 1.0


# ===========================================================================
# cohen_d
# ===========================================================================

class TestCohenD:
    def test_identical_groups_zero(self, lc):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert lc["cohen_d"](a, a) == 0.0

    def test_known_value(self, lc):
        # Two groups with mean diff ≈ 1, SD ≈ 1 → d ≈ 1
        rng = np.random.default_rng(0)
        a   = rng.normal(1.0, 1.0, 1000)
        b   = rng.normal(0.0, 1.0, 1000)
        d   = lc["cohen_d"](a, b)
        assert abs(d - 1.0) < 0.1

    def test_antisymmetric(self, lc):
        a = np.array([3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0])
        assert abs(lc["cohen_d"](a, b) + lc["cohen_d"](b, a)) < 1e-10

    def test_returns_float(self, lc):
        assert isinstance(lc["cohen_d"]([1, 2, 3], [4, 5, 6]), float)

    def test_zero_variance_returns_zero(self, lc):
        # pooled SD == 0 → returns 0.0 by guard
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert lc["cohen_d"](a, b) == 0.0


# ===========================================================================
# bootstrap_ci_d
# ===========================================================================

class TestBootstrapCiD:
    def test_returns_tuple_lo_lt_hi(self, lc):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = lc["bootstrap_ci_d"](a, b, n_boot=200, seed=0)
        assert lo < hi

    def test_ci_contains_true_d(self, lc):
        rng = np.random.default_rng(1)
        a   = rng.normal(1.0, 1.0, 50)
        b   = rng.normal(0.0, 1.0, 50)
        true_d = lc["cohen_d"](a, b)
        lo, hi = lc["bootstrap_ci_d"](a, b, n_boot=500, seed=42)
        assert lo <= true_d <= hi

    def test_reproducible_with_seed(self, lc):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.0, 1.0, 2.0, 3.0])
        r1 = lc["bootstrap_ci_d"](a, b, n_boot=100, seed=7)
        r2 = lc["bootstrap_ci_d"](a, b, n_boot=100, seed=7)
        assert r1 == r2


# ===========================================================================
# fdr_annotate
# ===========================================================================

class TestFdrAnnotate:
    def _make_df(self, p_values):
        return pd.DataFrame({
            "feature": [f"f{i}" for i in range(len(p_values))],
            "p_raw":   p_values,
        })

    def test_output_columns(self, lc):
        df  = self._make_df([0.01, 0.05, 0.5, 0.9])
        out = lc["fdr_annotate"](df, "p_raw")
        assert "p_fdr"     in out.columns
        assert "sig_fdr05" in out.columns
        assert "sig_raw05" in out.columns

    def test_sig_raw_threshold(self, lc):
        df  = self._make_df([0.01, 0.10])
        out = lc["fdr_annotate"](df, "p_raw")
        assert out.loc[out["p_raw"] == 0.01, "sig_raw05"].values[0]
        assert not out.loc[out["p_raw"] == 0.10, "sig_raw05"].values[0]

    def test_single_row_p_fdr_equals_p_raw(self, lc):
        # single row: fdr_bh branch skipped, p_fdr == p_raw
        df  = self._make_df([0.03])
        out = lc["fdr_annotate"](df, "p_raw")
        assert "p_fdr" in out.columns
        assert out["p_fdr"].values[0] == pytest.approx(0.03)

    def test_does_not_mutate_input(self, lc):
        df  = self._make_df([0.01, 0.5])
        _   = lc["fdr_annotate"](df, "p_raw")
        assert "p_fdr" not in df.columns


# ===========================================================================
# stream_filter
# ===========================================================================

class TestStreamFilter:
    def _make_df(self):
        return pd.DataFrame({
            "pid":    ["a", "b", "c", "d"],
            "age_mo": [12.0, 20.0, 35.0, 45.0],
            "value":  [1.0, 2.0, 3.0, 4.0],
        })

    def test_full_returns_all(self, lc):
        df  = self._make_df()
        out = lc["stream_filter"](df, "full")
        assert len(out) == len(df)

    def test_age_band_11_18(self, lc):
        df  = self._make_df()
        out = lc["stream_filter"](df, "11-18mo")
        # AGE_STREAMS['11-18mo'] = (11, 18) → age_mo between 11 and 18 inclusive
        assert all(out["age_mo"].between(11, 18))
        assert len(out) == 1

    def test_age_band_32_38(self, lc):
        df  = self._make_df()
        out = lc["stream_filter"](df, "32-38mo")
        assert all(out["age_mo"].between(32, 38))
        assert len(out) == 1

    def test_returns_copy(self, lc):
        df  = self._make_df()
        out = lc["stream_filter"](df, "full")
        out["value"] = 99
        assert df["value"].iloc[0] != 99


# ===========================================================================
# get_contiguous_segments
# ===========================================================================

class TestGetContiguousSegments:
    def _make_ldf(self, labels):
        return pd.DataFrame({
            "Frame":      list(range(len(labels))),
            "Locomotion": labels,
        })

    def test_single_segment(self, lc):
        ldf  = self._make_ldf(["Walking"] * 5)
        segs = lc["get_contiguous_segments"](ldf, "Locomotion", {"Walking"})
        assert segs == [(0, 4)]

    def test_two_segments(self, lc):
        labels = ["Walking"] * 3 + ["Other"] * 2 + ["Walking"] * 3
        ldf  = self._make_ldf(labels)
        segs = lc["get_contiguous_segments"](ldf, "Locomotion", {"Walking"})
        assert len(segs) == 2
        assert segs[0] == (0, 2)
        assert segs[1] == (5, 7)

    def test_no_valid_labels(self, lc):
        ldf  = self._make_ldf(["Other"] * 5)
        segs = lc["get_contiguous_segments"](ldf, "Locomotion", {"Walking"})
        assert segs == []

    def test_multiple_valid_labels(self, lc):
        labels = ["Walking", "Crawling", "Other", "Walking"]
        ldf  = self._make_ldf(labels)
        segs = lc["get_contiguous_segments"](ldf, "Locomotion", {"Walking", "Crawling"})
        # Walking(0)+Crawling(1) are consecutive valid labels → merged into (0,1)
        # Other(2) breaks the segment, then Walking(3) → (3,3)
        assert len(segs) == 2
        assert segs[0] == (0, 1)
        assert segs[1] == (3, 3)


# ===========================================================================
# run_pseudobulk_mw  (integration test with a fake child-level DataFrame)
# ===========================================================================

class TestRunPseudobulkMw:
    def _make_child_df(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "pid":    [f"s{i}" for i in range(20)],
            "Group":  ["ASD"] * 10 + ["Non-ASD"] * 10,
            "feat_a": np.concatenate([rng.normal(2.0, 1.0, 10), rng.normal(0.0, 1.0, 10)]),
            "feat_b": rng.random(20),
        })

    def test_returns_dataframe(self, lc):
        df  = self._make_child_df()
        out = lc["run_pseudobulk_mw"](df, ["feat_a", "feat_b"])
        assert isinstance(out, pd.DataFrame)

    def test_output_columns(self, lc):
        df  = self._make_child_df()
        out = lc["run_pseudobulk_mw"](df, ["feat_a"])
        for col in ("feature", "p_raw", "cohens_d", "p_fdr", "sig_raw05"):
            assert col in out.columns

    def test_well_separated_groups_low_p(self, lc):
        rng = np.random.default_rng(1)
        df  = pd.DataFrame({
            "pid":    [f"s{i}" for i in range(40)],
            "Group":  ["ASD"] * 20 + ["Non-ASD"] * 20,
            "feat_x": np.concatenate([rng.normal(5.0, 0.5, 20), rng.normal(0.0, 0.5, 20)]),
        })
        out = lc["run_pseudobulk_mw"](df, ["feat_x"])
        assert out.loc[out["feature"] == "feat_x", "p_raw"].values[0] < 0.01

    def test_empty_feat_cols_returns_empty_df(self, lc):
        df  = self._make_child_df()
        out = lc["run_pseudobulk_mw"](df, [])
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_insufficient_n_skipped(self, lc):
        # only 1 ASD and 1 Non-ASD → both < 3 → skipped → empty result
        df = pd.DataFrame({
            "pid":    ["a", "b"],
            "Group":  ["ASD", "Non-ASD"],
            "feat_z": [1.0, 2.0],
        })
        out = lc["run_pseudobulk_mw"](df, ["feat_z"])
        assert len(out) == 0


# ===========================================================================
# make_consensus
# ===========================================================================

class TestMakeConsensus:
    def _dummy_result(self, feat, p):
        return pd.DataFrame({
            "feature":   [feat],
            "p_raw":     [p],
            "cohens_d":  [0.5],
            "d_ci_lo":   [0.1],
            "d_ci_hi":   [0.9],
            "sig_raw05": [p < 0.05],
            "sig_fdr05": [p < 0.05],
        })

    def test_n_methods_sig_counting(self, lc):
        results = {
            "A": self._dummy_result("feat1", 0.01),
            "B": self._dummy_result("feat1", 0.04),
            "C": self._dummy_result("feat1", 0.20),
        }
        out = lc["make_consensus"](results, ["feat1"])
        assert out.loc[out["feature"] == "feat1", "n_methods_sig"].values[0] == 2

    def test_output_has_p_columns(self, lc):
        results = {"MethodX": self._dummy_result("f1", 0.05)}
        out = lc["make_consensus"](results, ["f1"])
        assert "p_MethodX" in out.columns

    def test_missing_feature_gets_nan(self, lc):
        results = {"A": self._dummy_result("feat1", 0.01)}
        out = lc["make_consensus"](results, ["feat1", "feat2"])
        assert pd.isna(out.loc[out["feature"] == "feat2", "p_A"].values[0])

    def test_sorted_by_n_methods_sig_descending(self, lc):
        results = {
            "A": pd.DataFrame({
                "feature": ["f1", "f2"], "p_raw": [0.01, 0.5],
                "cohens_d": [0.5, 0.1], "d_ci_lo": [0.1, 0.0],
                "d_ci_hi": [0.9, 0.2], "sig_raw05": [True, False],
                "sig_fdr05": [True, False],
            })
        }
        out = lc["make_consensus"](results, ["f1", "f2"])
        # f1 is sig in A (n_methods_sig=1), f2 is not (n_methods_sig=0)
        assert out.iloc[0]["feature"] == "f1"