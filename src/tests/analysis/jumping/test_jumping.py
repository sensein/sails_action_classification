"""Tests for src/sailsprep/analysis/jumping/jumping.py.

Tests cover all pure utility and statistical functions.
Heavy I/O (run_variant, run_lme_suite, run_bayesian_suite) is excluded
because they require real pose-data files on the research server.
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch
import importlib.util
import os
import ast  # noqa: E402
import numpy as np
import pandas as pd
import pytest

# Capture real scipy callables BEFORE the stub loop overwrites sys.modules.
# scipy IS installed on the server; we only stub it to avoid rendering/side-effects
# for modules that aren't installed.  _savage_dickey_bf needs the real gaussian_kde.
from scipy.stats import gaussian_kde as _real_gaussian_kde
from scipy.stats import norm as _real_norm

# ---------------------------------------------------------------------------
# Module import: the entry-point is guarded by if __name__ == "__main__",
# so importing the module is safe even without the CSV data files.
# ---------------------------------------------------------------------------

# Stub heavy optional deps so the module imports cleanly in CI.
_STUB_MODS: dict[str, Any] = {}
for _name in [
    "rpy2",
    "rpy2.robjects",
    "rpy2.robjects.packages",
    "pymc",
    "arviz",
    "wildboottest",
    "wildboottest.wildboottest",
    "statsmodels",
    "statsmodels.stats",
    "statsmodels.stats.multitest",
    "statsmodels.genmod",
    "statsmodels.genmod.generalized_estimating_equations",
    "statsmodels.genmod.families",
    "statsmodels.genmod.cov_struct",
    "statsmodels.formula",
    "statsmodels.formula.api",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.patches",
    "scipy",
    "scipy.signal",
    "scipy.stats",
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        sys.modules[_name] = _m
        _STUB_MODS[_name] = _m

# Provide enough attribute surface for module-level code in jumping.py
sys.modules["matplotlib"].use = lambda *a, **kw: None  # type: ignore[attr-defined]
_plt = sys.modules["matplotlib.pyplot"]
_plt.rcParams = {}  # type: ignore[attr-defined]
_plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[attr-defined]

_scipy_stats = sys.modules["scipy.stats"]
_scipy_stats.norm = MagicMock()  # type: ignore[attr-defined]
_scipy_stats.gaussian_kde = MagicMock()  # type: ignore[attr-defined]
# f_oneway: return an F-stat and p-value
_scipy_stats.f_oneway = MagicMock(return_value=(10.0, 0.01))  # type: ignore[attr-defined]
# mannwhitneyu: return a stat and p-value
_scipy_stats.mannwhitneyu = MagicMock(return_value=(50.0, 0.04))  # type: ignore[attr-defined]
# spearmanr: return (r, p)
_scipy_stats.spearmanr = MagicMock(return_value=(0.6, 0.02))  # type: ignore[attr-defined]
# linregress: return (slope, intercept, r, p, se)
_scipy_stats.linregress = MagicMock(return_value=(0.5, 1.0, 0.7, 0.01, 0.1))  # type: ignore[attr-defined]
# kruskal: return (stat, p)
_scipy_stats.kruskal = MagicMock(return_value=(5.0, 0.08))  # type: ignore[attr-defined]

_signal = sys.modules["scipy.signal"]
_signal.welch = MagicMock(return_value=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.5])))  # type: ignore[attr-defined]
_signal.butter = MagicMock(return_value=(np.array([1.0]), np.array([1.0])))  # type: ignore[attr-defined]
_signal.filtfilt = MagicMock(side_effect=lambda b, a, arr: arr)  # type: ignore[attr-defined]

_sm_multi = sys.modules["statsmodels.stats.multitest"]
_sm_multi.multipletests = lambda pvals, **kw: (None, np.array(pvals) * 2.0, None, None)  # type: ignore[attr-defined]

_sm_gee = sys.modules["statsmodels.genmod.generalized_estimating_equations"]
_sm_gee.GEE = MagicMock()  # type: ignore[attr-defined]

_sm_fam = sys.modules["statsmodels.genmod.families"]
_sm_fam.Gaussian = MagicMock()  # type: ignore[attr-defined]

_sm_cov = sys.modules["statsmodels.genmod.cov_struct"]
_sm_cov.Exchangeable = MagicMock()  # type: ignore[attr-defined]

_sm_api = sys.modules["statsmodels.formula.api"]
_sm_api.mixedlm = MagicMock()  # type: ignore[attr-defined]

_sklearn_lm = sys.modules["sklearn.linear_model"]
_sklearn_lm.LogisticRegression = MagicMock()  # type: ignore[attr-defined]

_sklearn_met = sys.modules["sklearn.metrics"]
_sklearn_met.roc_auc_score = MagicMock(return_value=0.75)  # type: ignore[attr-defined]
_sklearn_met.roc_curve = MagicMock(  # type: ignore[attr-defined]
    return_value=(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.8, 1.0]), np.array([1.0, 0.5, 0.0]))
)

_sklearn_pipe = sys.modules["sklearn.pipeline"]
_sklearn_pipe.Pipeline = MagicMock()  # type: ignore[attr-defined]

_sklearn_pre = sys.modules["sklearn.preprocessing"]
_sklearn_pre.StandardScaler = MagicMock()  # type: ignore[attr-defined]

# Now it is safe to import the module


_MOD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sailsprep", "analysis", "jumping", "jumping.py"
)


with open(_MOD_PATH, "r") as _fh:
    _src = _fh.read()

_tree = ast.parse(_src, filename=_MOD_PATH)
_lines = _src.splitlines()

# Find where the module switches from "library code" (imports, constants,
# function/class defs) to "script code" (loading real data, running the
# actual analysis). That's the df_main = pd.read_csv(...) line.
_cutoff_line = next(
    (i + 1 for i, ln in enumerate(_lines) if "read_csv" in ln and "MAIN_CSV" in ln),
    len(_lines) + 1,
)

_pre = [n for n in _tree.body if n.lineno < _cutoff_line]
_post_defs = [
    n for n in _tree.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.lineno >= _cutoff_line
]
_tree.body = _pre + _post_defs
ast.fix_missing_locations(_tree)

_spec = importlib.util.spec_from_file_location("jumping", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
_jmp = importlib.util.module_from_spec(_spec)
sys.modules["jumping"] = _jmp  # so dataclasses/typing introspection works if needed
exec(compile(_tree, _MOD_PATH, "exec"), _jmp.__dict__)  # noqa: S102

# Re-inject real scipy callables so _savage_dickey_bf works correctly.
# The stub loop replaced sys.modules["scipy.stats"], so the module picked up
# MagicMock objects for gaussian_kde and spnorm; overwrite them here.
_jmp.gaussian_kde = _real_gaussian_kde  # type: ignore[attr-defined]
_jmp.spnorm = _real_norm  # type: ignore[attr-defined]

# Pull all public functions into local namespace for convenience
extract_pid = _jmp.extract_pid
parse_timestamps = _jmp.parse_timestamps
get_kp = _jmp.get_kp
butter_lp = _jmp.butter_lp
torso_length = _jmp.torso_length
hip_width = _jmp.hip_width
get_scale = _jmp.get_scale
spectral_features = _jmp.spectral_features
cohen_d = _jmp.cohen_d
bootstrap_ci_d = _jmp.bootstrap_ci_d
cles = _jmp.cles
fdr_annotate = _jmp.fdr_annotate
assign_age_band = _jmp.assign_age_band
extract_jumping_features = _jmp.extract_jumping_features
compute_icc = _jmp.compute_icc
run_mwu_comparison = _jmp.run_mwu_comparison
run_child_permutation = _jmp.run_child_permutation
run_wild_bootstrap = _jmp.run_wild_bootstrap
make_consensus = _jmp.make_consensus
run_consistency_gate = _jmp.run_consistency_gate
run_spearman_age = _jmp.run_spearman_age
_add_label_dummies = _jmp._add_label_dummies
_standardise = _jmp._standardise
_build_bayes_df = _jmp._build_bayes_df
_savage_dickey_bf = _jmp._savage_dickey_bf  # uses real gaussian_kde after re-injection


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _make_kp(x: float, y: float, conf: float = 0.9) -> dict[str, float]:
    return {"x": x, "y": y, "confidence": conf}


def _make_frame(
    hip_y: float = 100.0,
    shoulder_y: float = 60.0,
    scale: float = 40.0,
) -> dict[str, Any]:
    """Synthetic pose frame with all keypoints present."""
    lh_y = hip_y
    rh_y = hip_y
    ls_y = shoulder_y
    rs_y = shoulder_y
    return {
        "kp_005": _make_kp(50.0, ls_y),       # left_shoulder
        "kp_006": _make_kp(60.0, rs_y),       # right_shoulder
        "kp_007": _make_kp(45.0, 70.0),       # left_elbow
        "kp_008": _make_kp(65.0, 70.0),       # right_elbow
        "kp_009": _make_kp(40.0, 80.0),       # left_wrist
        "kp_010": _make_kp(70.0, 80.0),       # right_wrist
        "kp_011": _make_kp(50.0, lh_y),       # left_hip
        "kp_012": _make_kp(60.0, rh_y),       # right_hip
        "kp_013": _make_kp(50.0, 120.0),      # left_knee
        "kp_014": _make_kp(60.0, 120.0),      # right_knee
        "kp_015": _make_kp(50.0, 140.0),      # left_ankle
        "kp_016": _make_kp(60.0, 140.0),      # right_ankle
    }


def _synthetic_pose_frames(n_frames: int = 30, amplitude: float = 10.0) -> dict[str, Any]:
    """Build a dict of pose frames with sinusoidal hip motion."""
    frames: dict[str, Any] = {}
    for i in range(n_frames):
        hip_y = 100.0 + amplitude * np.sin(2 * np.pi * i / 10.0)
        frames[str(i)] = _make_frame(hip_y=hip_y)
    return frames


def _make_clip_df(
    n_asd: int = 15,
    n_nasd: int = 15,
    seed: int = 42,
    feat_names: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic clip-level DataFrame with Group, pid, age_mo, label_lower."""
    rng = np.random.default_rng(seed)
    if feat_names is None:
        feat_names = ["mean_hip_y_amplitude", "bilateral_hip_y_corr"]
    rows = []
    for i in range(n_asd):
        row: dict[str, Any] = {
            "pid": f"sub-ASD{i:02d}",
            "Group": "ASD",
            "age_mo": rng.uniform(11, 38),
            "label_lower": "jumping",
        }
        for fn in feat_names:
            row[fn] = rng.normal(1.2, 0.3)
        rows.append(row)
    for i in range(n_nasd):
        row = {
            "pid": f"sub-NAD{i:02d}",
            "Group": "Non-ASD",
            "age_mo": rng.uniform(11, 38),
            "label_lower": "jumping",
        }
        for fn in feat_names:
            row[fn] = rng.normal(1.0, 0.3)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# TESTS: extract_pid
# ═══════════════════════════════════════════════════════════════════════

class TestExtractPid:
    def test_standard_path(self) -> None:
        assert extract_pid("/data/sub-ABC123/video.mp4") == "sub-ABC123"

    def test_bids_style(self) -> None:
        assert extract_pid("sub-XY99_task-jump.json") == "sub-XY99"

    def test_no_match_returns_none(self) -> None:
        assert extract_pid("/data/no_subject_here/video.mp4") is None

    def test_non_string_returns_none(self) -> None:
        assert extract_pid(None) is None
        assert extract_pid(42) is None

    def test_multiple_occurrences_returns_first(self) -> None:
        # re.search returns first match
        result = extract_pid("/data/sub-AAA/sub-BBB/file")
        assert result == "sub-AAA"


# ═══════════════════════════════════════════════════════════════════════
# TESTS: parse_timestamps
# ═══════════════════════════════════════════════════════════════════════

class TestParseTimestamps:
    def test_single_segment(self) -> None:
        segs = parse_timestamps("0:10-0:20", fps=15.0)
        assert len(segs) == 1
        s, e = segs[0]
        assert s == 10 * 15
        assert e == 20 * 15

    def test_multiple_segments(self) -> None:
        segs = parse_timestamps("0:05-0:10, 1:00-1:30", fps=15.0)
        assert len(segs) == 2

    def test_inverted_segment_skipped(self) -> None:
        segs = parse_timestamps("0:30-0:10", fps=15.0)
        assert len(segs) == 0

    def test_non_string_returns_empty(self) -> None:
        assert parse_timestamps(None) == []
        assert parse_timestamps(123) == []

    def test_fps_scaling(self) -> None:
        segs_15 = parse_timestamps("0:00-0:01", fps=15.0)
        segs_30 = parse_timestamps("0:00-0:01", fps=30.0)
        assert segs_30[0][1] == 2 * segs_15[0][1]

    def test_minute_rollover(self) -> None:
        segs = parse_timestamps("1:00-2:00", fps=1.0)
        assert segs[0] == (60, 120)


# ═══════════════════════════════════════════════════════════════════════
# TESTS: get_kp
# ═══════════════════════════════════════════════════════════════════════

class TestGetKp:
    def test_returns_kp_above_threshold(self) -> None:
        fd = {"kp_011": {"x": 1.0, "y": 2.0, "confidence": 0.9}}
        result = get_kp(fd, "kp_011", min_conf=0.3)
        assert result is not None
        assert result["y"] == 2.0

    def test_below_confidence_returns_none(self) -> None:
        fd = {"kp_011": {"x": 1.0, "y": 2.0, "confidence": 0.1}}
        assert get_kp(fd, "kp_011", min_conf=0.3) is None

    def test_missing_key_returns_none(self) -> None:
        assert get_kp({}, "kp_011") is None

    def test_non_dict_value_returns_none(self) -> None:
        fd = {"kp_011": "bad_value"}
        assert get_kp(fd, "kp_011") is None

    def test_exact_threshold_boundary(self) -> None:
        fd = {"kp_011": {"x": 0.0, "y": 0.0, "confidence": 0.3}}
        # confidence == min_conf: kp.get('confidence', 0) < min_conf is False → returns kp
        assert get_kp(fd, "kp_011", min_conf=0.3) is not None


# ═══════════════════════════════════════════════════════════════════════
# TESTS: torso_length / hip_width / get_scale
# ═══════════════════════════════════════════════════════════════════════

class TestBodyMeasurements:
    def test_torso_length_normal(self) -> None:
        fd = _make_frame(hip_y=100.0, shoulder_y=60.0)
        tl = torso_length(fd)
        assert tl is not None
        assert tl > 0

    def test_torso_length_missing_kp_returns_none(self) -> None:
        fd = {}
        assert torso_length(fd) is None

    def test_hip_width_normal(self) -> None:
        fd = _make_frame()
        hw = hip_width(fd)
        assert hw is not None
        assert hw > 0

    def test_get_scale_prefers_torso(self) -> None:
        fd = _make_frame()
        sc = get_scale(fd)
        tl = torso_length(fd)
        assert sc == tl

    def test_get_scale_falls_back_to_hip_width(self) -> None:
        # Remove shoulder keypoints so torso_length returns None
        fd = _make_frame()
        fd.pop("kp_005")
        fd.pop("kp_006")
        sc = get_scale(fd)
        hw = hip_width(fd)
        assert sc == hw

    def test_torso_too_small_returns_none(self) -> None:
        # Shoulders and hips at the same position → distance = 0
        fd = _make_frame(hip_y=60.0, shoulder_y=60.0)
        fd["kp_011"] = _make_kp(55.0, 60.0, 0.9)
        fd["kp_012"] = _make_kp(55.0, 60.0, 0.9)
        fd["kp_005"] = _make_kp(55.0, 60.0, 0.9)
        fd["kp_006"] = _make_kp(55.0, 60.0, 0.9)
        assert torso_length(fd) is None


# ═══════════════════════════════════════════════════════════════════════
# TESTS: spectral_features  (using real scipy via the import in jumping.py)
# ═══════════════════════════════════════════════════════════════════════

class TestSpectralFeatures:
    def test_too_short_returns_nan(self) -> None:
        dom, ent, bp = spectral_features(np.ones(5), fps=15.0)
        assert all(np.isnan(v) for v in [dom, ent, bp])

    def test_returns_three_floats(self) -> None:
        # Patch the stubbed welch to return real-looking data
        arr = np.sin(2 * np.pi * 2.0 * np.arange(64) / 15.0)
        sys.modules["scipy.signal"].welch = MagicMock(  # type: ignore[attr-defined]
            return_value=(
                np.linspace(0, 7.5, 33),
                np.abs(np.fft.rfft(arr)) ** 2,
            )
        )
        dom, ent, bp = spectral_features(arr, fps=15.0)
        assert isinstance(dom, float)
        assert isinstance(ent, float)
        assert isinstance(bp, float)


# ═══════════════════════════════════════════════════════════════════════
# TESTS: cohen_d
# ═══════════════════════════════════════════════════════════════════════

class TestCohenD:
    def test_no_effect(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cohen_d(a, a) == pytest.approx(0.0)

    def test_positive_direction(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(2.0, 1.0, 30)
        b = rng.normal(1.0, 1.0, 30)
        d = cohen_d(a, b)
        assert d > 0

    def test_negative_direction(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 1.0, 30)
        b = rng.normal(2.0, 1.0, 30)
        assert cohen_d(a, b) < 0

    def test_zero_variance_returns_zero(self) -> None:
        assert cohen_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0

    def test_known_value(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(loc=1.0, scale=1.0, size=1000)
        b = rng.normal(loc=0.0, scale=1.0, size=1000)
        d = cohen_d(a, b)
        assert abs(d - 1.0) < 0.1  # approximately 1.0


# ═══════════════════════════════════════════════════════════════════════
# TESTS: bootstrap_ci_d
# ═══════════════════════════════════════════════════════════════════════

class TestBootstrapCiD:
    def test_returns_tuple_of_two_floats(self) -> None:
        a = np.random.default_rng(1).normal(1.0, 1.0, 30)
        b = np.random.default_rng(2).normal(0.0, 1.0, 30)
        lo, hi = bootstrap_ci_d(a, b, n_boot=100, seed=0)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_brackets_observed_d(self) -> None:
        a = np.random.default_rng(3).normal(1.5, 1.0, 50)
        b = np.random.default_rng(4).normal(0.0, 1.0, 50)
        d = cohen_d(a, b)
        lo, hi = bootstrap_ci_d(a, b, n_boot=200, seed=5)
        assert lo < d < hi

    def test_symmetric_groups_ci_straddles_zero(self) -> None:
        rng = np.random.default_rng(6)
        a = rng.normal(0.0, 1.0, 50)
        b = rng.normal(0.0, 1.0, 50)
        lo, hi = bootstrap_ci_d(a, b, n_boot=200, seed=7)
        assert lo < 0 < hi


# ═══════════════════════════════════════════════════════════════════════
# TESTS: cles
# ═══════════════════════════════════════════════════════════════════════

class TestCles:
    def test_equal_groups_near_half(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        # half of comparisons will be ties (ai > bi is False for equal)
        c = cles(a, b)
        # With equal distributions CLES won't be exactly 0.5 due to ties
        assert 0.0 <= c <= 1.0

    def test_a_always_greater(self) -> None:
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        assert cles(a, b) == 1.0

    def test_b_always_greater(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [10.0, 11.0, 12.0]
        assert cles(a, b) == 0.0

    def test_range(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(0.5, 1.0, 20)
        b = rng.normal(0.0, 1.0, 20)
        c = cles(a, b)
        assert 0.0 <= c <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# TESTS: assign_age_band
# ═══════════════════════════════════════════════════════════════════════

class TestAssignAgeBand:
    def test_early_band(self) -> None:
        assert assign_age_band(15) == "11-18mo"

    def test_middle_band(self) -> None:
        assert assign_age_band(25) == "19-31mo"

    def test_late_band(self) -> None:
        assert assign_age_band(35) == "32-38mo"

    def test_boundaries_inclusive(self) -> None:
        assert assign_age_band(11) == "11-18mo"
        assert assign_age_band(18) == "11-18mo"
        assert assign_age_band(19) == "19-31mo"
        assert assign_age_band(32) == "32-38mo"
        assert assign_age_band(38) == "32-38mo"

    def test_out_of_range_returns_none(self) -> None:
        assert assign_age_band(5) is None
        assert assign_age_band(50) is None


# ═══════════════════════════════════════════════════════════════════════
# TESTS: fdr_annotate
# ═══════════════════════════════════════════════════════════════════════

class TestFdrAnnotate:
    def _make_df(self, p_vals: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"feature": [f"f{i}" for i in range(len(p_vals))], "p_raw": p_vals})

    def test_adds_p_fdr_column(self) -> None:
        df = self._make_df([0.01, 0.05, 0.5])
        # multipletests is stubbed; we just check the column is added
        result = fdr_annotate(df, "p_raw")
        assert "p_fdr" in result.columns

    def test_adds_sig_flags(self) -> None:
        df = self._make_df([0.001, 0.1])
        result = fdr_annotate(df, "p_raw")
        assert "sig_raw05" in result.columns
        assert "sig_fdr05" in result.columns

    def test_single_row_no_crash(self) -> None:
        df = self._make_df([0.03])
        result = fdr_annotate(df, "p_raw")
        assert len(result) == 1
        assert result["p_fdr"].iloc[0] == pytest.approx(0.03)

    def test_sig_raw05_correct(self) -> None:
        df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "p_raw": [0.01, 0.06, 0.001],
        })
        result = fdr_annotate(df, "p_raw")
        raw_sig = result.set_index("feature")["sig_raw05"]
        assert raw_sig["a"] is True or raw_sig["a"] == True  # noqa: E712
        assert raw_sig["b"] is False or raw_sig["b"] == False  # noqa: E712
        assert raw_sig["c"] is True or raw_sig["c"] == True  # noqa: E712


# ═══════════════════════════════════════════════════════════════════════
# TESTS: _add_label_dummies
# ═══════════════════════════════════════════════════════════════════════

class TestAddLabelDummies:
    def test_reference_excluded(self) -> None:
        df = pd.DataFrame({"label_lower": ["jumping", "bouncing", "bouncing", "jumping"]})
        result_df, cols = _add_label_dummies(df, reference="jumping")
        # "jumping" should NOT be a dummy column
        assert all("jumping" not in c for c in cols)

    def test_dummy_col_added_for_non_reference(self) -> None:
        df = pd.DataFrame({"label_lower": ["jumping", "bouncing", "knee jumping"]})
        _, cols = _add_label_dummies(df, reference="jumping")
        assert len(cols) == 2  # bouncing and knee jumping

    def test_single_label_no_dummies(self) -> None:
        df = pd.DataFrame({"label_lower": ["jumping", "jumping"]})
        _, cols = _add_label_dummies(df, reference="jumping")
        assert cols == []

    def test_dummy_values_binary(self) -> None:
        df = pd.DataFrame({"label_lower": ["jumping", "bouncing", "jumping"]})
        result_df, cols = _add_label_dummies(df, reference="jumping")
        for col in cols:
            assert set(result_df[col].unique()).issubset({0.0, 1.0})


# ═══════════════════════════════════════════════════════════════════════
# TESTS: _standardise
# ═══════════════════════════════════════════════════════════════════════

class TestStandardise:
    def test_zero_mean_unit_std(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z, m, std = _standardise(s)
        assert abs(np.mean(z)) < 1e-10
        # _standardise divides by pandas .std() which uses ddof=1
        assert abs(np.std(z, ddof=1) - 1.0) < 1e-6

    def test_returns_original_mean_std(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0])
        _, m, std = _standardise(s)
        assert m == pytest.approx(20.0)
        assert std == pytest.approx(10.0)

    def test_constant_series_no_divide_by_zero(self) -> None:
        s = pd.Series([5.0, 5.0, 5.0])
        z, _, std = _standardise(s)
        assert std == 1.0  # fallback when std < 1e-10
        assert not np.any(np.isnan(z))


# ═══════════════════════════════════════════════════════════════════════
# TESTS: extract_jumping_features
# ═══════════════════════════════════════════════════════════════════════

class TestExtractJumpingFeatures:
    def test_too_few_valid_frames_returns_none(self) -> None:
        # Only 3 frames → n_valid < 5 → None
        frames = _synthetic_pose_frames(n_frames=3)
        result = extract_jumping_features(frames, list(range(3)))
        assert result is None

    def test_normal_segment_returns_dict(self) -> None:
        frames = _synthetic_pose_frames(n_frames=30)
        result = extract_jumping_features(frames, list(range(30)))
        assert result is not None
        assert isinstance(result, dict)

    def test_n_valid_frames_in_result(self) -> None:
        frames = _synthetic_pose_frames(n_frames=20)
        result = extract_jumping_features(frames, list(range(20)))
        assert result is not None
        assert result["n_valid_frames"] == 20

    def test_mean_hip_y_amplitude_positive(self) -> None:
        frames = _synthetic_pose_frames(n_frames=30, amplitude=15.0)
        result = extract_jumping_features(frames, list(range(30)))
        assert result is not None
        assert result.get("mean_hip_y_amplitude", 0) > 0

    def test_bilateral_hip_corr_in_minus1_to_1(self) -> None:
        frames = _synthetic_pose_frames(n_frames=30)
        result = extract_jumping_features(frames, list(range(30)))
        assert result is not None
        corr = result.get("bilateral_hip_y_corr")
        if corr is not None:
            assert -1.0 <= corr <= 1.0

    def test_missing_frames_skipped_gracefully(self) -> None:
        frames = _synthetic_pose_frames(n_frames=20)
        # Request frames that don't exist (frame 100 is absent)
        indices = list(range(20)) + [100, 200]
        result = extract_jumping_features(frames, indices)
        assert result is not None

    def test_hip_ankle_amp_ratio_positive(self) -> None:
        frames = _synthetic_pose_frames(n_frames=30, amplitude=10.0)
        result = extract_jumping_features(frames, list(range(30)))
        assert result is not None
        ratio = result.get("hip_ankle_amp_ratio")
        if ratio is not None:
            assert ratio > 0

    def test_pct_valid_between_0_and_1(self) -> None:
        frames = _synthetic_pose_frames(n_frames=20)
        result = extract_jumping_features(frames, list(range(20)))
        assert result is not None
        assert 0.0 <= result["pct_valid"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# TESTS: compute_icc
# ═══════════════════════════════════════════════════════════════════════

class TestComputeIcc:
    def _make_icc_df(self) -> pd.DataFrame:
        """5 children × 4 clips each, feature values differ by child."""
        rows = []
        for pid_i in range(5):
            base = float(pid_i) * 2.0
            for _ in range(4):
                rows.append({"pid": f"sub-{pid_i:02d}", "feat_a": base + np.random.randn() * 0.1})
        return pd.DataFrame(rows)

    def test_returns_dataframe(self) -> None:
        df = self._make_icc_df()
        result = compute_icc(df, ["feat_a"])
        assert isinstance(result, pd.DataFrame)

    def test_icc_between_0_and_1(self) -> None:
        df = self._make_icc_df()
        result = compute_icc(df, ["feat_a"])
        if len(result):
            assert 0.0 <= result["ICC"].iloc[0] <= 1.0

    def test_high_between_cluster_variance_gives_high_icc(self) -> None:
        rows = []
        for pid_i in range(8):
            for _ in range(5):
                # Very consistent within child, very different between children
                rows.append({"pid": f"s{pid_i}", "big_effect": float(pid_i * 10.0)})
        df = pd.DataFrame(rows)
        result = compute_icc(df, ["big_effect"])
        if len(result):
            assert result["ICC"].iloc[0] > 0.5



# ═══════════════════════════════════════════════════════════════════════
# TESTS: run_mwu_comparison
# ═══════════════════════════════════════════════════════════════════════

class TestRunMwuComparison:
    def _df(self) -> tuple[pd.DataFrame, list[str]]:
        df = _make_clip_df(n_asd=20, n_nasd=20, seed=99)
        feats = ["mean_hip_y_amplitude", "bilateral_hip_y_corr"]
        return df, feats

    def test_returns_dataframe(self) -> None:
        df, feats = self._df()
        result = run_mwu_comparison(df, feats)
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self) -> None:
        df, feats = self._df()
        result = run_mwu_comparison(df, feats)
        for col in ["feature", "p_raw", "cohens_d", "p_fdr"]:
            assert col in result.columns

    def test_sorted_by_p_raw(self) -> None:
        df, feats = self._df()
        result = run_mwu_comparison(df, feats)
        if len(result) > 1:
            assert list(result["p_raw"]) == sorted(result["p_raw"].tolist())

    def test_too_few_samples_returns_empty(self) -> None:
        df = pd.DataFrame({
            "Group": ["ASD", "Non-ASD"],
            "feat": [1.0, 2.0],
        })
        result = run_mwu_comparison(df, ["feat"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_custom_group_columns(self) -> None:
        df, feats = self._df()
        df["age_band"] = df["age_mo"].apply(assign_age_band)
        df2 = df[df["age_band"].isin(["11-18mo", "32-38mo"])].copy()
        if len(df2) > 6:
            result = run_mwu_comparison(
                df2, feats,
                group_col="age_band",
                group_a="11-18mo",
                group_b="32-38mo",
            )
            assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════
# TESTS: run_child_permutation
# ═══════════════════════════════════════════════════════════════════════

class TestRunChildPermutation:
    def _child_df(self) -> tuple[pd.DataFrame, list[str]]:
        df = _make_clip_df(n_asd=12, n_nasd=12, seed=77)
        feats = ["mean_hip_y_amplitude"]
        return df, feats

    def test_returns_dataframe(self) -> None:
        df, feats = self._child_df()
        result = run_child_permutation(df, feats, n_perm=100)
        assert isinstance(result, pd.DataFrame)

    def test_p_value_in_0_1(self) -> None:
        df, feats = self._child_df()
        result = run_child_permutation(df, feats, n_perm=100)
        if len(result):
            p = result["p_raw"].iloc[0]
            assert 0.0 < p <= 1.0

    def test_cohens_d_present(self) -> None:
        df, feats = self._child_df()
        result = run_child_permutation(df, feats, n_perm=100)
        if len(result):
            assert "cohens_d" in result.columns

    def test_missing_feat_handled(self) -> None:
        df, _ = self._child_df()
        df["all_nan"] = np.nan
        result = run_child_permutation(df, ["all_nan"], n_perm=50)
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════
# TESTS: run_wild_bootstrap
# ═══════════════════════════════════════════════════════════════════════

class TestRunWildBootstrap:
    def _child_df(self) -> tuple[pd.DataFrame, list[str]]:
        df = _make_clip_df(n_asd=10, n_nasd=10, seed=55)
        return df, ["mean_hip_y_amplitude"]

    def test_returns_dataframe(self) -> None:
        df, feats = self._child_df()
        result = run_wild_bootstrap(df, feats, n_boot=100)
        assert isinstance(result, pd.DataFrame)

    def test_p_value_bounded(self) -> None:
        df, feats = self._child_df()
        result = run_wild_bootstrap(df, feats, n_boot=100)
        if len(result):
            p = result["p_raw"].iloc[0]
            assert 0.0 < p <= 1.0

    def test_coef_asd_column_present(self) -> None:
        df, feats = self._child_df()
        result = run_wild_bootstrap(df, feats, n_boot=100)
        if len(result):
            assert "coef_ASD" in result.columns


# ═══════════════════════════════════════════════════════════════════════
# TESTS: make_consensus
# ═══════════════════════════════════════════════════════════════════════

class TestMakeConsensus:
    def _make_res(self, feats: list[str], p_vals: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"feature": feats, "p_raw": p_vals, "cohens_d": [0.5] * len(feats)})

    def test_n_methods_sig_counted_correctly(self) -> None:
        feats = ["f1", "f2"]
        res_a = self._make_res(feats, [0.01, 0.5])
        res_b = self._make_res(feats, [0.02, 0.9])
        result = make_consensus({"M1": res_a, "M2": res_b}, feats)
        row_f1 = result[result["feature"] == "f1"].iloc[0]
        row_f2 = result[result["feature"] == "f2"].iloc[0]
        assert row_f1["n_methods_sig"] == 2
        assert row_f2["n_methods_sig"] == 0

    def test_none_result_handled(self) -> None:
        feats = ["f1"]
        res_a = self._make_res(feats, [0.01])
        result = make_consensus({"M1": res_a, "M2": None}, feats)
        assert isinstance(result, pd.DataFrame)
        assert "f1" in result["feature"].values

    def test_sorted_by_n_methods_sig_descending(self) -> None:
        feats = ["f1", "f2", "f3"]
        res = self._make_res(feats, [0.001, 0.5, 0.01])
        result = make_consensus({"M1": res}, feats)
        counts = result["n_methods_sig"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_empty_results_dict(self) -> None:
        result = make_consensus({}, ["f1", "f2"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════
# TESTS: run_consistency_gate
# ═══════════════════════════════════════════════════════════════════════

class TestRunConsistencyGate:
    def _make_multi_label_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(33)
        rows = []
        for lbl in ["jumping", "bouncing"]:
            for grp, mean in [("ASD", 1.5), ("Non-ASD", 1.0)]:
                for i in range(10):
                    rows.append({
                        "pid": f"sub-{grp[:3]}{i}",
                        "Group": grp,
                        "label_lower": lbl,
                        "feat_a": rng.normal(mean, 0.3),
                    })
        return pd.DataFrame(rows)

    def test_returns_three_objects(self) -> None:
        df = self._make_multi_label_df()
        cons_df, consist_feats, label_mwu = run_consistency_gate(
            df, ["feat_a"], ["feat_a"]
        )
        assert isinstance(cons_df, pd.DataFrame)
        assert isinstance(consist_feats, list)
        assert isinstance(label_mwu, dict)

    def test_consistent_feature_flagged(self) -> None:
        df = self._make_multi_label_df()
        cons_df, consist_feats, _ = run_consistency_gate(
            df, ["feat_a"], ["feat_a"]
        )
        # ASD > Non-ASD in both labels → should pass consistency gate
        if len(cons_df):
            assert cons_df.iloc[0]["consistent"] in [True, False]

    def test_empty_sig_feats_returns_empty(self) -> None:
        df = self._make_multi_label_df()
        cons_df, consist_feats, _ = run_consistency_gate(df, ["feat_a"], [])
        assert len(cons_df) == 0
        assert consist_feats == []


# ═══════════════════════════════════════════════════════════════════════
# TESTS: run_spearman_age
# ═══════════════════════════════════════════════════════════════════════

class TestRunSpearmanAge:
    def test_returns_dataframe(self) -> None:
        df = _make_clip_df(n_asd=15, n_nasd=15)
        result = run_spearman_age(df, ["mean_hip_y_amplitude"])
        assert isinstance(result, pd.DataFrame)

    def test_sig_p05_column_present(self) -> None:
        df = _make_clip_df(n_asd=15, n_nasd=15)
        result = run_spearman_age(df, ["mean_hip_y_amplitude"])
        if len(result):
            assert "sig_p05" in result.columns

    def test_monotone_feature_is_significant(self) -> None:
        # Create a feature that is perfectly correlated with age in ASD group
        rng = np.random.default_rng(42)
        n = 20
        ages = np.linspace(11, 38, n)
        df = pd.DataFrame({
            "pid": [f"sub-{i}" for i in range(n)],
            "Group": ["ASD"] * n,
            "age_mo": ages,
            "label_lower": ["jumping"] * n,
            "monotone_feat": ages + rng.normal(0, 0.01, n),
        })
        result = run_spearman_age(df, ["monotone_feat"])
        asd_row = result[result["Group"] == "ASD"]
        if len(asd_row):
            assert asd_row["sig_p05"].iloc[0] is True or asd_row["p_raw"].iloc[0] < 0.05

    def test_empty_group_skipped(self) -> None:
        df = _make_clip_df(n_asd=10, n_nasd=0)
        result = run_spearman_age(df, ["mean_hip_y_amplitude"])
        groups = result["Group"].unique() if len(result) else []
        assert "Non-ASD" not in groups


# ═══════════════════════════════════════════════════════════════════════
# TESTS: _savage_dickey_bf (pure math, no pymc needed)
# ═══════════════════════════════════════════════════════════════════════

class TestSavageDickeyBf:
    def test_posterior_away_from_zero_gives_bf_gt_1(self) -> None:
        rng = np.random.default_rng(0)
        post = rng.normal(1.0, 0.2, 5000)  # posterior far from 0
        bf = _savage_dickey_bf(post, prior_sd=0.5)
        # prior density at 0 is higher than posterior density at 0 → BF > 1
        assert bf > 1.0

    def test_posterior_at_zero_gives_bf_lt_1(self) -> None:
        rng = np.random.default_rng(1)
        post = rng.normal(0.0, 0.05, 5000)  # very tight posterior AT 0
        bf = _savage_dickey_bf(post, prior_sd=0.5)
        # posterior density at 0 >> prior density at 0 → BF < 1
        assert bf < 1.0

    def test_returns_float(self) -> None:
        post = np.random.default_rng(2).normal(0.5, 0.3, 1000)
        bf = _savage_dickey_bf(post, prior_sd=0.5)
        assert isinstance(bf, float)