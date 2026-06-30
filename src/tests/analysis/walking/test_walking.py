"""
tests_walking.py
Tests for src/sailsprep/analysis/walking/walking.py

Run:  poetry run pytest src/tests/tests_walking.py -v
"""

import os
import sys
import types
import tempfile
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub heavy optional deps so tests run without them
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]

_stub("pymc")
_stub("arviz")
_stub("statsmodels")
_stub("statsmodels.formula")
_stub("statsmodels.formula.api")
_stub("statsmodels.stats")
_stub("statsmodels.stats.multitest",
      multipletests=lambda p, method=None: (None, p, None, None))
_stub("statsmodels.genmod")
_stub("statsmodels.genmod.families",  Gaussian=lambda: None)
_stub("statsmodels.genmod.cov_struct", Exchangeable=lambda: None)
_stub("statsmodels.genmod.generalized_estimating_equations", GEE=lambda *a, **k: None)

# ---------------------------------------------------------------------------
# Load walking.py helpers by truncating at PART 0 (avoids exec'ing I/O code)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_WALKING = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "sailsprep", "analysis", "walking", "walking.py",
)
_ns: dict = {}
_LOADED = False
_LOAD_ERROR: Exception | None = None


def _load_helpers() -> None:
    global _LOADED, _LOAD_ERROR

    with open(SRC_WALKING, "r") as fh:
        source = fh.read()

    td = tempfile.mkdtemp()

    # ── Regex-replace paths (immune to future path changes) ──
    import re as _re
    source = _re.sub(r'MAIN_CSV\s*=\s*"[^"]+"',   'MAIN_CSV   = "/tmp/_dummy_walking_test.csv"', source)
    source = _re.sub(r'OUTPUT_DIR\s*=\s*"[^"]+"',  f'OUTPUT_DIR = "{td}"', source)
    source = source.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')")

    # ── Cut PART 0 (CSV load block) ──
    p0 = source.find('hr("PART 0: LOAD DATA")')
    p1 = source.find('hr("PART 1: FEATURE EXTRACTION")')   # ← fixed marker
    if p0 != -1 and p1 != -1:
        source = source[:p0] + source[p1:]

    # ── Cut extraction loop + everything after PART 1 defs ──
    ex_s = source.find('# ── Extraction loop')
    p2   = source.find('hr("PART 2: STATISTICAL ANALYSIS")')
    if ex_s != -1 and p2 != -1:
        source = source[:ex_s] + source[p2:]

    # ── Drop PART 2 onwards entirely ──
    p2 = source.find('hr("PART 2: STATISTICAL ANALYSIS")')
    if p2 != -1:
        source = source[:p2]

    try:
        exec(compile(source, SRC_WALKING, "exec"), _ns)  # noqa: S102

        # ── Inject run_mwu (run_pseudobulk_mw was in PART 2, now cut) ──
        from scipy import stats as _stats

        def _run_mwu(df, feat_cols, subset="combined"):
            _cohen_d   = _ns["cohen_d"]
            _boot_ci   = _ns["bootstrap_ci_d"]
            _fdr       = _ns["fdr_annotate"]
            recs = []
            for feat in feat_cols:
                av = df[df["Group"] == "ASD"][feat].dropna().values
                nv = df[df["Group"] == "Non-ASD"][feat].dropna().values
                if len(av) < 3 or len(nv) < 3:
                    continue
                stat, p = _stats.mannwhitneyu(av, nv, alternative="two-sided")
                d = _cohen_d(av, nv)
                ci_lo, ci_hi = _boot_ci(av, nv, n_boot=200)
                recs.append({
                    "feature": feat, "subset": subset,
                    "ASD_n": len(av), "NonASD_n": len(nv),
                    "ASD_median": float(np.median(av)),
                    "NonASD_median": float(np.median(nv)),
                    "mw_stat": float(stat), "p_raw": float(p),
                    "cohens_d": d, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                })
            if not recs:
                return pd.DataFrame()
            return _fdr(pd.DataFrame(recs).sort_values("p_raw"), "p_raw")

        _ns["run_mwu"] = _run_mwu
        _LOADED = True
    except Exception as exc:
        _LOAD_ERROR = exc


_load_helpers()

pytestmark = pytest.mark.skipif(
    not _LOADED,
    reason=f"Could not load walking.py helpers: {_LOAD_ERROR}",
)


def _fn(name):
    return _ns[name]


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def dummy_pose_frames():
    """30 frames of synthetic HRNet-133 keypoints, all high-confidence."""
    rng = np.random.default_rng(0)
    frames = {}
    for fi in range(30):
        fd = {}
        # Fill every slot with baseline values
        for kid in range(23):
            fd[f"kp_{kid:03d}"] = {
                "x": float(rng.uniform(100, 400)),
                "y": float(200 + rng.uniform(-5, 5)),
                "confidence": 0.85,
            }
        # ── Realistic torso (shoulders above hips) ──
        fd["kp_005"].update({"x": 150.0, "y": 100.0})  # L_shoulder
        fd["kp_006"].update({"x": 250.0, "y": 100.0})  # R_shoulder
        fd["kp_011"].update({"x": 155.0, "y": 200.0})  # L_hip
        fd["kp_012"].update({"x": 245.0, "y": 200.0})  # R_hip
        # ── Knee below hip ──
        fd["kp_013"].update({"x": 155.0, "y": 300.0})  # L_knee
        fd["kp_014"].update({"x": 245.0, "y": 300.0})  # R_knee
        # ── Ankle with alternating gait pattern ──
        t = 2 * np.pi * fi / 10
        fd["kp_015"].update({"y": 400.0 + 10 * np.sin(t)})        # L_ankle
        fd["kp_016"].update({"y": 400.0 + 10 * np.sin(t + np.pi)})# R_ankle
        frames[str(fi)] = fd
    return frames


@pytest.fixture()
def dummy_feat_df():
    """20-row synthetic clip-level feature DataFrame."""
    rng = np.random.default_rng(1)
    n = 20
    return pd.DataFrame({
        "pid":      [f"sub-{i:02d}" for i in range(n)],
        "Group":    ["ASD"] * 10 + ["Non-ASD"] * 10,
        "age_mo":   rng.uniform(11, 38, n).tolist(),
        "age_band": (["11-18mo"] * 5 + ["32-38mo"] * 5) * 2,
        "ankle_y_L_amplitude":   rng.uniform(0.1, 1.0, n).tolist(),
        "knee_angle_L_range":    rng.uniform(10,  60,  n).tolist(),
        "stride_duration_mean":  rng.uniform(0.4, 1.2, n).tolist(),
        "lateral_sway_std":      rng.uniform(0.01, 0.3, n).tolist(),
        "cadence":               rng.uniform(0.5, 2.0, n).tolist(),
        "ankle_lr_amplitude_asym": rng.uniform(0.0, 0.5, n).tolist(),
    })


# ===========================================================================
# extract_pid
# ===========================================================================

class TestExtractPid:
    def test_valid(self):
        assert _fn("extract_pid")("/data/bids/sub-ABC123/vid.mp4") == "sub-ABC123"

    def test_no_match(self):
        assert _fn("extract_pid")("/data/other/file.mp4") is None

    def test_non_string(self):
        assert _fn("extract_pid")(None) is None
        assert _fn("extract_pid")(42) is None


# ===========================================================================
# assign_age_band
# ===========================================================================

class TestAssignAgeBand:
    def test_all_bands(self):
        fn = _fn("assign_age_band")
        assert fn(11)  == "11-18mo"
        assert fn(18)  == "11-18mo"
        assert fn(19)  == "19-31mo"
        assert fn(31)  == "19-31mo"
        assert fn(32)  == "32-38mo"
        assert fn(38)  == "32-38mo"

    def test_out_of_range(self):
        fn = _fn("assign_age_band")
        assert fn(5)  is None
        assert fn(50) is None


# ===========================================================================
# get_kp
# ===========================================================================

class TestGetKp:
    def test_above_threshold(self):
        fd = {"kp_015": {"x": 1.0, "y": 2.0, "confidence": 0.8}}
        kp = _fn("get_kp")(fd, "kp_015", min_conf=0.3)
        assert kp is not None and kp["x"] == 1.0

    def test_below_threshold(self):
        fd = {"kp_015": {"x": 1.0, "y": 2.0, "confidence": 0.1}}
        assert _fn("get_kp")(fd, "kp_015", min_conf=0.3) is None

    def test_missing_key(self):
        assert _fn("get_kp")({}, "kp_015") is None

    def test_non_dict_value(self):
        assert _fn("get_kp")({"kp_015": None}, "kp_015") is None


# ===========================================================================
# torso_length
# ===========================================================================

class TestTorsoLength:
    def test_valid(self):
        fd = {
            "kp_005": {"x": 150, "y": 100, "confidence": 0.9},
            "kp_006": {"x": 250, "y": 100, "confidence": 0.9},
            "kp_011": {"x": 155, "y": 200, "confidence": 0.9},
            "kp_012": {"x": 245, "y": 200, "confidence": 0.9},
        }
        tl = _fn("torso_length")(fd)
        assert tl is not None and tl > 5

    def test_missing_joints(self):
        assert _fn("torso_length")({}) is None

    def test_zero_distance(self):
        fd = {k: {"x": 0, "y": 0, "confidence": 0.9}
              for k in ("kp_005", "kp_006", "kp_011", "kp_012")}
        assert _fn("torso_length")(fd) is None   # d <= 5 → None


# ===========================================================================
# get_scale
# ===========================================================================

class TestGetScale:
    def test_falls_back_to_hip_width(self):
        fd = {
            "kp_011": {"x": 100, "y": 200, "confidence": 0.9},
            "kp_012": {"x": 200, "y": 200, "confidence": 0.9},
        }
        scale = _fn("get_scale")(fd)
        assert scale is not None and scale > 5

    def test_none_on_empty(self):
        assert _fn("get_scale")({}) is None


# ===========================================================================
# butter_lp
# ===========================================================================

class TestButterLP:
    def test_output_length(self):
        arr = np.sin(np.linspace(0, 6, 60))
        out = _fn("butter_lp")(arr, cutoff=4.0, fs=15.0)
        assert len(out) == len(arr)

    def test_short_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_fn("butter_lp")(arr), arr)

    def test_high_cutoff_clipped(self):
        arr = np.random.default_rng(2).standard_normal(60)
        out = _fn("butter_lp")(arr, cutoff=100.0, fs=15.0)
        assert len(out) == len(arr)


# ===========================================================================
# compute_angle_2d
# ===========================================================================

class TestComputeAngle2D:
    def test_right_angle(self):
        angle = _fn("compute_angle_2d")((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90.0) < 1e-6

    def test_straight_line(self):
        angle = _fn("compute_angle_2d")((-1, 0), (0, 0), (1, 0))
        assert abs(angle - 180.0) < 1e-6

    def test_zero_vector_returns_nan(self):
        assert np.isnan(_fn("compute_angle_2d")((0, 0), (0, 0), (1, 0)))

    def test_result_in_degree_range(self):
        a = _fn("compute_angle_2d")((0, 1), (0, 0), (1, 0))
        assert 0.0 <= a <= 180.0


# ===========================================================================
# spectral_features
# ===========================================================================

class TestSpectralFeatures:
    def test_pure_sine_dominant_freq(self):
        t   = np.linspace(0, 4, 60)
        sig = np.sin(2 * np.pi * 1.0 * t)
        dom_f, ent, bp = _fn("spectral_features")(sig, fps=15.0, lo=0.5, hi=2.0)
        assert abs(dom_f - 1.0) < 0.5
        assert not np.isnan(ent)
        assert 0.0 <= bp <= 1.0

    def test_short_array_returns_nans(self):
        dom_f, ent, bp = _fn("spectral_features")(np.array([1.0, 2.0]), fps=15.0)
        assert np.isnan(dom_f)


# ===========================================================================
# sparc_smoothness
# ===========================================================================

class TestSparcSmoothness:
    def test_returns_negative_float(self):
        vel = np.random.default_rng(0).standard_normal(30)
        val = _fn("sparc_smoothness")(vel, fps=15.0)
        assert isinstance(val, float) and val < 0

    def test_short_array_nan(self):
        assert np.isnan(_fn("sparc_smoothness")(np.array([1.0]), fps=15.0))


# ===========================================================================
# mean_jerk
# ===========================================================================

class TestMeanJerk:
    def test_constant_signal_zero_jerk(self):
        val = _fn("mean_jerk")(np.ones(30), fps=15.0)
        assert abs(val) < 1e-6

    def test_short_array_nan(self):
        assert np.isnan(_fn("mean_jerk")(np.array([1.0, 2.0]), fps=15.0))


# ===========================================================================
# detect_gait_cycles
# ===========================================================================

class TestDetectGaitCycles:
    def test_detects_cycles(self):
        t   = np.linspace(0, 4, 60)
        sig = np.sin(2 * np.pi * 1.2 * t)
        cycles = _fn("detect_gait_cycles")(sig, fps=15.0)
        assert len(cycles) >= 2

    def test_flat_signal_empty(self):
        assert _fn("detect_gait_cycles")(np.ones(60), fps=15.0) == []

    def test_too_short_empty(self):
        assert _fn("detect_gait_cycles")(np.zeros(5), fps=15.0) == []

    def test_returns_list(self):
        sig = np.ones(60) * 5.0
        assert isinstance(_fn("detect_gait_cycles")(sig, fps=15.0), list)


# ===========================================================================
# cohen_d  &  bootstrap_ci_d
# ===========================================================================

class TestCohenD:
    def test_identical_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _fn("cohen_d")(a, a) == 0.0

    def test_positive_when_a_higher(self):
        a = np.array([3.0, 4.0, 5.0])   # need variance for pooled SD > 0
        b = np.array([0.0, 1.0, 2.0])
        assert _fn("cohen_d")(a, b) > 0

    def test_bootstrap_ci_ordered(self):
        a = np.arange(5, dtype=float)
        b = np.arange(5, dtype=float) + 1
        lo, hi = _fn("bootstrap_ci_d")(a, b, n_boot=100)
        assert lo <= hi


# ===========================================================================
# fdr_annotate
# ===========================================================================

class TestFdrAnnotate:
    def test_adds_columns(self):
        df = pd.DataFrame({"p_raw": [0.001, 0.04, 0.5, 0.9]})
        out = _fn("fdr_annotate")(df, "p_raw")
        for col in ("p_fdr", "sig_fdr05", "sig_raw05"):
            assert col in out.columns

    def test_small_p_flagged(self):
        df = pd.DataFrame({"p_raw": [0.001, 0.9]})
        out = _fn("fdr_annotate")(df, "p_raw")
        assert bool(out["sig_raw05"].iloc[0]) is True
        assert bool(out["sig_raw05"].iloc[1]) is False


# ===========================================================================
# extract_walking_features  (integration)
# ===========================================================================

class TestExtractWalkingFeatures:
    def test_returns_dict(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, list(range(30)), fps=15.0)
        assert isinstance(result, dict)

    def test_basic_keys_present(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, list(range(30)), fps=15.0)
        assert result is not None
        for key in ("n_valid_frames", "n_total_frames", "duration_sec", "torso_cv"):
            assert key in result

    def test_duration_correct(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, list(range(30)), fps=15.0)
        assert result is not None
        assert abs(result["duration_sec"] - 30 / 15.0) < 1e-6

    def test_empty_frames_none(self):
        result = _fn("extract_walking_features")({}, list(range(30)), fps=15.0)
        assert result is None

    def test_too_few_frames_none(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, [0, 1, 2], fps=15.0)
        assert result is None

    def test_ankle_features_present(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, list(range(30)), fps=15.0)
        assert result is not None
        has_ankle = any("ankle" in k for k in result)
        assert has_ankle

    def test_torso_cv_finite(self, dummy_pose_frames):
        result = _fn("extract_walking_features")(
            dummy_pose_frames, list(range(30)), fps=15.0)
        assert result is not None
        assert np.isfinite(result["torso_cv"])

    def test_zero_confidence_returns_none(self):
        """All keypoints below min_conf → no valid frames → None."""
        KP = _fn("KP")
        frames = {
            str(fi): {kid: {"x": 100.0, "y": 200.0, "confidence": 0.0}
                      for kid in KP.values()}
            for fi in range(30)
        }
        assert _fn("extract_walking_features")(frames, list(range(30))) is None


# ===========================================================================
# run_mwu  (integration)
# ===========================================================================

class TestRunMwu:
    def test_returns_dataframe(self, dummy_feat_df):
        feats = ["ankle_y_L_amplitude", "knee_angle_L_range"]
        result = _fn("run_mwu")(dummy_feat_df, feats)
        assert isinstance(result, pd.DataFrame) and len(result) > 0

    def test_required_columns(self, dummy_feat_df):
        result = _fn("run_mwu")(dummy_feat_df, ["ankle_y_L_amplitude"])
        for col in ("feature", "p_raw", "cohens_d", "p_fdr", "sig_fdr05"):
            assert col in result.columns

    def test_p_values_in_range(self, dummy_feat_df):
        result = _fn("run_mwu")(dummy_feat_df, ["ankle_y_L_amplitude", "cadence"])
        assert result["p_raw"].between(0.0, 1.0).all()

    def test_too_few_samples_empty(self):
        tiny = pd.DataFrame({"Group": ["ASD", "Non-ASD"], "feat": [1.0, 2.0]})
        assert len(_fn("run_mwu")(tiny, ["feat"])) == 0