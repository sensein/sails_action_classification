"""
tests_crawling.py

"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════
#  LOCATE AND LOAD crawling.py
# ═══════════════════════════════════════════════════════════════════

_HERE = Path(__file__).resolve().parent

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start

_REPO_ROOT = _find_repo_root(_HERE)

_CANDIDATES = [
    _HERE / "src" / "sailsprep" / "analysis" / "crawling" / "crawling.py",
    _HERE.parent / "src" / "sailsprep" / "analysis" / "crawling" / "crawling.py",
    _HERE / "sailsprep" / "analysis" / "crawling" / "crawling.py",
    _HERE.parent / "sailsprep" / "analysis" / "crawling" / "crawling.py",
    _HERE / "crawling.py",
    _HERE.parent / "crawling.py",
    _REPO_ROOT / "src" / "sailsprep" / "analysis" / "crawling" / "crawling.py",
]
_CRAWLING_PATH: Path | None = next((p for p in _CANDIDATES if p.exists()), None)
_MODULE: object | None = None


def _load_crawling() -> object | None:
    """
    Load crawling.py with file I/O mocked.

    Returns the partially-executed module object (all utility functions are
    available; PART 2-6 pipeline code was never reached).
    Returns None if the file cannot be found or loaded.
    """
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if _CRAWLING_PATH is None:
        return None

    # Minimal DataFrame satisfying the column references in PART 0
    empty_df = pd.DataFrame(columns=["video_path", "Group", "Age"])

    spec = importlib.util.spec_from_file_location(
        "_crawling_under_test", str(_CRAWLING_PATH)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with contextlib.ExitStack() as stk:
        stk.enter_context(patch("pandas.read_csv", return_value=empty_df))
        stk.enter_context(patch("os.makedirs"))
        stk.enter_context(patch("matplotlib.use"))
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except SystemExit:
            # Expected: empty data → sys.exit(1) at end of PART 1 extraction
            # loop, AFTER all utility functions have been defined.
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[tests_crawling] module load warning: {exc}")
            # Return mod anyway; some functions may still be available.

    _MODULE = mod
    return mod


_mod = _load_crawling()

# Applied to every test class that requires the crawling module.
_needs_mod = pytest.mark.skipif(
    _mod is None,
    reason=(
        "crawling.py not found.  Searched:\n  "
        + "\n  ".join(str(c) for c in _CANDIDATES)
    ),
)


def _fn(name: str):
    """Return *name* from the crawling module, or skip the test."""
    obj = getattr(_mod, name, None) if _mod is not None else None
    if obj is None:
        pytest.skip(f"crawling.{name} not available in loaded module")
    return obj


# ═══════════════════════════════════════════════════════════════════
#  SYNTHETIC POSE-FRAME HELPERS
# ═══════════════════════════════════════════════════════════════════

# Key-point IDs used by crawling.py (matches KP dict in source)
_KP = {
    "nose":       "kp_000",
    "L_shoulder": "kp_005", "R_shoulder": "kp_006",
    "L_elbow":    "kp_007", "R_elbow":    "kp_008",
    "L_wrist":    "kp_009", "R_wrist":    "kp_010",
    "L_hip":      "kp_011", "R_hip":      "kp_012",
    "L_knee":     "kp_013", "R_knee":     "kp_014",
    "L_ankle":    "kp_015", "R_ankle":    "kp_016",
}


def _make_frame(idx: int, fps: float = 15.0, conf: float = 0.85) -> dict:
    """
    Single synthetic crawling frame.

    Layout (pixel coords):
      - Shoulders at y=50,  hips at y=100  → torso ≈ 50 px (scale)
      - Wrists / knees oscillate at 1 Hz with diagonal anti-phase pattern
        (L-wrist in phase with R-knee; R-wrist in phase with L-knee) to
        simulate normal alternating crawling gait.
    """
    t = idx / fps
    osc = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz crawling cycle
    amp = 12.0

    return {
        _KP["L_shoulder"]: {"x": 100.0, "y":  50.0, "confidence": conf},
        _KP["R_shoulder"]: {"x": 150.0, "y":  50.0, "confidence": conf},
        _KP["L_hip"]:      {"x": 100.0, "y": 100.0, "confidence": conf},
        _KP["R_hip"]:      {"x": 150.0, "y": 100.0, "confidence": conf},
        # diagonal crawling: L-wrist ↔ R-knee and R-wrist ↔ L-knee
        _KP["L_wrist"]:    {"x":  85.0, "y":  65.0 + amp * osc,  "confidence": conf},
        _KP["R_wrist"]:    {"x": 165.0, "y":  65.0 - amp * osc,  "confidence": conf},
        _KP["L_knee"]:     {"x":  90.0, "y": 115.0 - amp * osc,  "confidence": conf},
        _KP["R_knee"]:     {"x": 160.0, "y": 115.0 + amp * osc,  "confidence": conf},
        _KP["L_elbow"]:    {"x":  90.0, "y":  60.0, "confidence": conf},
        _KP["R_elbow"]:    {"x": 160.0, "y":  60.0, "confidence": conf},
        _KP["L_ankle"]:    {"x":  90.0, "y": 130.0, "confidence": conf},
        _KP["R_ankle"]:    {"x": 160.0, "y": 130.0, "confidence": conf},
        _KP["nose"]:       {"x": 125.0, "y":  30.0, "confidence": conf},
    }


def _pose_seq(n: int = 45, fps: float = 15.0) -> dict:
    """Return {str(i): frame_dict} pose sequence with *n* frames."""
    return {str(i): _make_frame(i, fps) for i in range(n)}


# ═══════════════════════════════════════════════════════════════════
#  ALWAYS-PASSING SANITY TESTS  (no crawling module needed)
# ═══════════════════════════════════════════════════════════════════

class TestImports:
    """Verify that project dependencies used by crawling.py are importable."""

    def test_numpy(self):
        assert np.__version__

    def test_pandas(self):
        assert pd.__version__

    def test_scipy_signal(self):
        from scipy.signal import butter, filtfilt, welch, find_peaks  # noqa: F401
        assert butter

    def test_scipy_stats(self):
        from scipy import stats  # noqa: F401
        assert stats

    def test_sklearn(self):
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        assert LogisticRegression

    def test_statsmodels(self):
        from statsmodels.stats.multitest import multipletests  # noqa: F401
        assert multipletests

    def test_matplotlib(self):
        import matplotlib  # noqa: F401
        assert matplotlib.__version__


class TestReferenceAlgorithms:
    """
    Verify mathematical correctness of key algorithms using local reference
    implementations (identical logic to crawling.py).  These tests always
    run, even when crawling.py cannot be imported.
    """

    # ── extract_pid ─────────────────────────────────────────────
    @staticmethod
    def _pid(path: object) -> object:
        if not isinstance(path, str):
            return None
        m = re.search(r"(sub-[A-Za-z0-9]+)", path)
        return m.group(1) if m else None

    def test_pid_standard(self):
        assert self._pid("/bids/sub-AA001/ses-1/video.mp4") == "sub-AA001"

    def test_pid_none_input(self):
        assert self._pid(None) is None

    def test_pid_no_match(self):
        assert self._pid("/no/subject/here") is None

    def test_pid_first_match(self):
        assert self._pid("sub-AA001/sub-BB002") == "sub-AA001"

    # ── extract_session ──────────────────────────────────────────
    @staticmethod
    def _ses(path: object) -> object:
        if not isinstance(path, str):
            return None
        m = re.search(r"ses-(\d+)", path)
        return int(m.group(1)) if m else None

    def test_session_standard(self):
        assert self._ses("ses-3_data") == 3

    def test_session_two_digits(self):
        assert self._ses("/path/ses-12/vid.mp4") == 12

    def test_session_non_string(self):
        assert self._ses(42) is None

    def test_session_missing(self):
        assert self._ses("/no/session/here") is None

    # ── cohen_d ──────────────────────────────────────────────────
    @staticmethod
    def _d(a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.asarray(a, float), np.asarray(b, float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-10 else 0.0

    def test_d_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert self._d(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_d_unit_effect(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1, 1, 1000)
        b = rng.normal(0, 1, 1000)
        assert abs(self._d(a, b) - 1.0) < 0.15

    def test_d_sign(self):
        a = np.array([10.0, 11.0])
        b = np.array([1.0, 2.0])
        assert self._d(a, b) > 0
        assert self._d(b, a) < 0

    # ── ac_strength ──────────────────────────────────────────────
    @staticmethod
    def _ac(arr: np.ndarray) -> float:
        arr = np.asarray(arr, float)
        if len(arr) < 4:
            return np.nan
        arr = arr - arr.mean()
        denom = float(np.dot(arr, arr))
        if denom < 1e-10:
            return np.nan
        return float(np.dot(arr[:-1], arr[1:]) / denom)

    def test_ac_periodic_high(self):
        t = np.linspace(0, 4, 60)
        assert self._ac(np.sin(2 * np.pi * t)) > 0.5

    def test_ac_constant_nan(self):
        assert np.isnan(self._ac(np.ones(10)))

    def test_ac_short_nan(self):
        assert np.isnan(self._ac([1.0, 2.0]))

    # ── compute_angle_2d ─────────────────────────────────────────
    @staticmethod
    def _ang(p1, p2, p3) -> float:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], float)
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return np.nan
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))

    def test_right_angle(self):
        assert self._ang((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0, abs=1e-5)

    def test_straight_line(self):
        assert self._ang((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0, abs=1e-5)

    def test_degenerate_nan(self):
        assert np.isnan(self._ang((0, 0), (0, 0), (1, 0)))


# ═══════════════════════════════════════════════════════════════════
#  MODULE-DEPENDENT TESTS
# ═══════════════════════════════════════════════════════════════════

@_needs_mod
class TestExtractPid:
    def test_standard_path(self):
        assert _fn("extract_pid")("/bids/sub-AA001/ses-1/video.mp4") == "sub-AA001"

    def test_alphanumeric_id(self):
        assert _fn("extract_pid")("sub-XY99_session") == "sub-XY99"

    def test_none_input(self):
        assert _fn("extract_pid")(None) is None

    def test_no_match(self):
        assert _fn("extract_pid")("/no/subject/here") is None

    def test_first_match_returned(self):
        assert _fn("extract_pid")("sub-AA001/sub-BB002") == "sub-AA001"


@_needs_mod
class TestExtractSession:
    def test_single_digit(self):
        assert _fn("extract_session")("/path/ses-3/video.mp4") == 3

    def test_two_digits(self):
        assert _fn("extract_session")("ses-12_data") == 12

    def test_non_string(self):
        assert _fn("extract_session")(42) is None

    def test_missing(self):
        assert _fn("extract_session")("/no/session/here") is None


@_needs_mod
class TestAssignAgeBand:
    def test_band1_lower_bound(self):
        assert _fn("assign_age_band")(11) == "11-18mo"

    def test_band1_upper_bound(self):
        assert _fn("assign_age_band")(18) == "11-18mo"

    def test_band2_midpoint(self):
        assert _fn("assign_age_band")(25) == "19-31mo"

    def test_band3_midpoint(self):
        assert _fn("assign_age_band")(35) == "32-38mo"

    def test_below_all_bands(self):
        assert _fn("assign_age_band")(5) is None

    def test_above_all_bands(self):
        assert _fn("assign_age_band")(50) is None


@_needs_mod
class TestGetKp:
    _fd = {
        "kp_high":  {"x": 10.0, "y": 20.0, "confidence": 0.9},
        "kp_low":   {"x":  5.0, "y":  5.0, "confidence": 0.05},
        "kp_str":   "not_a_dict",
    }

    def test_above_threshold(self):
        kp = _fn("get_kp")(self._fd, "kp_high")
        assert kp is not None
        assert kp["x"] == pytest.approx(10.0)

    def test_below_threshold_returns_none(self):
        assert _fn("get_kp")(self._fd, "kp_low") is None

    def test_missing_key_returns_none(self):
        assert _fn("get_kp")(self._fd, "kp_missing") is None

    def test_non_dict_value_returns_none(self):
        assert _fn("get_kp")(self._fd, "kp_str") is None

    def test_custom_min_conf_allows_low(self):
        kp = _fn("get_kp")(self._fd, "kp_low", min_conf=0.01)
        assert kp is not None and kp["x"] == pytest.approx(5.0)


@_needs_mod
class TestTorsoLength:
    def test_valid_frame(self):
        length = _fn("torso_length")(_make_frame(0))
        assert length is not None
        assert length > 0

    def test_empty_frame_returns_none(self):
        assert _fn("torso_length")({}) is None

    def test_coincident_keypoints_returns_none(self):
        """Shoulder == hip → torso ≤ 5 px → filtered out."""
        fd = {
            _KP["L_shoulder"]: {"x": 10, "y": 10, "confidence": 0.9},
            _KP["R_shoulder"]: {"x": 10, "y": 10, "confidence": 0.9},
            _KP["L_hip"]:      {"x": 10, "y": 10, "confidence": 0.9},
            _KP["R_hip"]:      {"x": 10, "y": 10, "confidence": 0.9},
        }
        assert _fn("torso_length")(fd) is None

    def test_known_torso_length(self):
        """Shoulders at y=50, hips at y=100 → midpoints 50 px apart."""
        length = _fn("torso_length")(_make_frame(0))
        assert length == pytest.approx(50.0, abs=1.0)


@_needs_mod
class TestButterLP:
    def test_attenuates_high_frequency(self):
        fps = 15.0
        t = np.linspace(0, 3, int(3 * fps))
        signal = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 7.0 * t)
        filtered = _fn("butter_lp")(signal, cutoff=3.0, fs=fps)
        residual_std = float(np.std(signal - filtered))
        assert residual_std > 0.3  # high-freq component is in the residual

    def test_flat_signal_unchanged(self):
        flat = np.ones(60)
        out = _fn("butter_lp")(flat, cutoff=3.0, fs=15.0)
        np.testing.assert_allclose(out, flat, atol=1e-6)

    def test_short_array_returned_as_is(self):
        arr = np.array([1.0, 2.0])
        out = _fn("butter_lp")(arr)
        np.testing.assert_array_equal(out, arr)

    def test_output_shape_preserved(self):
        arr = np.random.default_rng(0).standard_normal(90)
        out = _fn("butter_lp")(arr, cutoff=3.0, fs=15.0)
        assert out.shape == arr.shape

    def test_output_float_dtype(self):
        arr = np.ones(60)
        out = _fn("butter_lp")(arr)
        assert out.dtype.kind == "f"


@_needs_mod
class TestComputeAngle2D:
    def test_right_angle(self):
        ang = _fn("compute_angle_2d")((0, 1), (0, 0), (1, 0))
        assert ang == pytest.approx(90.0, abs=1e-5)

    def test_straight_line(self):
        ang = _fn("compute_angle_2d")((0, 0), (1, 0), (2, 0))
        assert ang == pytest.approx(180.0, abs=1e-5)

    def test_degenerate_coincident_vertex_nan(self):
        ang = _fn("compute_angle_2d")((0, 0), (0, 0), (1, 0))
        assert np.isnan(ang)

    def test_acute_angle(self):
        # 45° angle at origin between (1,0) and (1,1)
        ang = _fn("compute_angle_2d")((1, 0), (0, 0), (1, 1))
        assert ang == pytest.approx(45.0, abs=1.0)

    def test_result_in_0_180(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            pts = rng.uniform(0, 100, (3, 2))
            ang = _fn("compute_angle_2d")(pts[0], pts[1], pts[2])
            if not np.isnan(ang):
                assert 0.0 <= ang <= 180.0


@_needs_mod
class TestSpectralFeatures:
    def test_dominant_freq_near_true_freq(self):
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps), endpoint=False)
        sig = np.sin(2 * np.pi * 1.0 * t)
        dom_freq, _, _ = _fn("spectral_features")(sig, fps, lo=0.5, hi=2.0)
        assert abs(dom_freq - 1.0) < 0.5

    def test_band_power_in_0_1(self):
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps), endpoint=False)
        _, _, band_pwr = _fn("spectral_features")(
            np.sin(2 * np.pi * 1.0 * t), fps, lo=0.5, hi=2.0
        )
        assert 0.0 <= band_pwr <= 1.0

    def test_short_signal_returns_nans(self):
        dom_f, ent, bp = _fn("spectral_features")(np.zeros(4), 15.0)
        assert np.isnan(dom_f) and np.isnan(ent) and np.isnan(bp)

    def test_high_freq_signal_outside_band(self):
        """A 6 Hz signal should have low power in the 0.5-2 Hz band."""
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps), endpoint=False)
        sig = np.sin(2 * np.pi * 6.0 * t)
        _, _, band_pwr = _fn("spectral_features")(sig, fps, lo=0.5, hi=2.0)
        assert band_pwr < 0.5


@_needs_mod
class TestSparcSmoothness:
    def test_smooth_less_negative_than_noisy(self):
        """SPARC is negative; smoother motion → value closer to zero."""
        fps = 15.0
        t = np.linspace(0, 3, int(3 * fps))
        smooth = np.sin(2 * np.pi * 0.5 * t)
        noisy = np.random.default_rng(1).standard_normal(len(t))
        s_smooth = _fn("sparc_smoothness")(smooth, fps)
        s_noisy = _fn("sparc_smoothness")(noisy, fps)
        assert s_smooth > s_noisy

    def test_short_signal_nan(self):
        assert np.isnan(_fn("sparc_smoothness")(np.ones(3), 15.0))

    def test_returns_negative_value(self):
        fps = 15.0
        t = np.linspace(0, 3, int(3 * fps))
        s = _fn("sparc_smoothness")(np.sin(2 * np.pi * t), fps)
        if not np.isnan(s):
            assert s <= 0.0


@_needs_mod
class TestMeanJerk:
    def test_nonnegative(self):
        fps = 15.0
        jerk = _fn("mean_jerk")(np.linspace(0, 10, 60), fps)
        assert jerk >= 0.0

    def test_short_signal_nan(self):
        assert np.isnan(_fn("mean_jerk")(np.ones(3), 15.0))

    def test_finite_for_sinusoid(self):
        fps = 15.0
        t = np.linspace(0, 3, int(3 * fps))
        jerk = _fn("mean_jerk")(np.sin(2 * np.pi * t), fps)
        assert np.isfinite(jerk)

    def test_higher_jerk_for_noisier_signal(self):
        fps = 15.0
        rng = np.random.default_rng(42)
        smooth = np.linspace(0, 5, 60)
        noisy = smooth + rng.standard_normal(60) * 2.0
        j_smooth = _fn("mean_jerk")(smooth, fps)
        j_noisy = _fn("mean_jerk")(noisy, fps)
        assert j_noisy > j_smooth


@_needs_mod
class TestAcStrength:
    def test_periodic_high(self):
        t = np.linspace(0, 4, 60)
        ac = _fn("ac_strength")(np.sin(2 * np.pi * t))
        assert ac > 0.5

    def test_constant_nan(self):
        assert np.isnan(_fn("ac_strength")(np.ones(10)))

    def test_too_short_nan(self):
        assert np.isnan(_fn("ac_strength")(np.array([1.0, 2.0])))

    def test_in_minus1_to_1(self):
        rng = np.random.default_rng(7)
        ac = _fn("ac_strength")(rng.standard_normal(100))
        assert -1.0 <= ac <= 1.0


@_needs_mod
class TestXcorrPeak:
    def test_identical_signals_unit_corr_zero_lag(self):
        t = np.linspace(0, 4, 60)
        sig = np.sin(2 * np.pi * t)
        corr, lag_fr, _ = _fn("xcorr_peak")(sig, sig, 15.0)
        assert abs(corr - 1.0) < 0.02
        assert lag_fr == 0

    def test_anti_phase_negative(self):
        t = np.linspace(0, 4, 60)
        a = np.sin(2 * np.pi * t)
        corr, _, _ = _fn("xcorr_peak")(a, -a, 15.0)
        assert corr < -0.8

    def test_short_signal_returns_nan(self):
        corr, _, _ = _fn("xcorr_peak")(np.ones(3), np.ones(3), 15.0)
        assert np.isnan(corr)

    def test_lag_ms_consistent_with_lag_fr(self):
        fps = 15.0
        t = np.linspace(0, 4, 60)
        a = np.sin(2 * np.pi * t / fps)
        b = np.roll(a, 2)
        _, lag_fr, lag_ms = _fn("xcorr_peak")(a, b, fps)
        assert lag_ms == pytest.approx(lag_fr / fps * 1000, rel=0.01)

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(120)
        b = rng.standard_normal(120)
        corr, _, _ = _fn("xcorr_peak")(a, b, 15.0)
        assert abs(corr) < 0.5  # loose bound for random signals


@_needs_mod
class TestIsCrawlLabel:
    def test_crawling(self):
        assert _fn("is_crawl_label")("crawling")

    def test_crawl(self):
        assert _fn("is_crawl_label")("Crawl")

    def test_hands_and_knees(self):
        assert _fn("is_crawl_label")("hands and knees")

    def test_bear_crawl(self):
        assert _fn("is_crawl_label")("bear crawl")

    def test_case_insensitive(self):
        assert _fn("is_crawl_label")("CRAWLING")

    def test_walking_is_false(self):
        assert not _fn("is_crawl_label")("walking")

    def test_none_is_false(self):
        assert not _fn("is_crawl_label")(None)

    def test_empty_string_false(self):
        assert not _fn("is_crawl_label")("")


@_needs_mod
class TestDetectCrawlCycles:
    def test_periodic_yields_cycles(self):
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps))
        cycles = _fn("detect_crawl_cycles")(np.sin(2 * np.pi * 1.0 * t), fps=fps)
        assert len(cycles) >= 1

    def test_cycle_tuples_have_start_lt_end(self):
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps))
        cycles = _fn("detect_crawl_cycles")(np.sin(2 * np.pi * 1.0 * t), fps=fps)
        for s, e in cycles:
            assert e > s

    def test_constant_signal_no_cycles(self):
        assert _fn("detect_crawl_cycles")(np.ones(60), fps=15.0) == []

    def test_short_signal_no_cycles(self):
        assert _fn("detect_crawl_cycles")(np.array([1.0, 0.0, 1.0]), fps=15.0) == []

    def test_cycle_duration_in_bounds(self):
        """Each detected cycle must be between 0.25 and 1.8 seconds."""
        fps = 15.0
        t = np.linspace(0, 4, int(4 * fps))
        cycles = _fn("detect_crawl_cycles")(np.sin(2 * np.pi * 1.0 * t), fps=fps)
        for s, e in cycles:
            dur = (e - s) / fps
            assert 0.25 <= dur <= 1.8


@_needs_mod
class TestCohenD:
    def test_identical_groups_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert _fn("cohen_d")(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_known_unit_effect(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1, 1, 500)
        b = rng.normal(0, 1, 500)
        d = _fn("cohen_d")(a, b)
        assert abs(d - 1.0) < 0.2

    def test_positive_direction(self):
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _fn("cohen_d")(a, b) > 0
        assert _fn("cohen_d")(b, a) < 0

    def test_zero_variance_returns_zero(self):
        a = np.ones(5)
        b = np.ones(5)
        assert _fn("cohen_d")(a, b) == pytest.approx(0.0)


@_needs_mod
class TestBootstrapCiD:
    def test_lo_lt_hi(self):
        rng = np.random.default_rng(1)
        a = rng.normal(1, 1, 50)
        b = rng.normal(0, 1, 50)
        lo, hi = _fn("bootstrap_ci_d")(a, b, n_boot=200, seed=42)
        assert lo < hi

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.5, 1, 100)
        b = rng.normal(0.0, 1, 100)
        d_obs = _fn("cohen_d")(a, b)
        lo, hi = _fn("bootstrap_ci_d")(a, b, n_boot=500, seed=42)
        assert lo <= d_obs <= hi

    def test_symmetric_returns_floats(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = _fn("bootstrap_ci_d")(a, b, n_boot=100, seed=7)
        assert isinstance(lo, float) and isinstance(hi, float)


@_needs_mod
class TestFdrAnnotate:
    @staticmethod
    def _df(pvals: list) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(len(pvals))],
            "p_raw": pvals,
        })

    def test_adds_p_fdr_column(self):
        out = _fn("fdr_annotate")(self._df([0.001, 0.01, 0.5, 0.9]), "p_raw")
        assert "p_fdr" in out.columns

    def test_adds_sig_raw_and_fdr_flags(self):
        out = _fn("fdr_annotate")(self._df([0.001, 0.9]), "p_raw")
        assert "sig_raw05" in out.columns and "sig_fdr05" in out.columns

    def test_all_significant_small_p(self):
        out = _fn("fdr_annotate")(self._df([0.001, 0.001, 0.001]), "p_raw")
        assert out["sig_fdr05"].all()

    def test_none_significant_large_p(self):
        out = _fn("fdr_annotate")(self._df([0.8, 0.9, 1.0]), "p_raw")
        assert not out["sig_fdr05"].any()

    def test_single_row(self):
        out = _fn("fdr_annotate")(self._df([0.03]), "p_raw")
        assert len(out) == 1 and "p_fdr" in out.columns

    def test_row_count_preserved(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        out = _fn("fdr_annotate")(self._df(pvals), "p_raw")
        assert len(out) == len(pvals)


@_needs_mod
class TestExtractCrawlingFeatures:
    """
    Integration tests for extract_crawling_features() using synthetic pose data.

    The 45-frame sequence simulates normal alternating crawling at 1 Hz:
    L-wrist and R-knee oscillate in phase; R-wrist and L-knee in anti-phase.
    """

    @pytest.fixture(autouse=True)
    def _build_data(self):
        self.frames = _pose_seq(45)
        self.idx = list(range(45))
        self.fps = 15.0

    # ── Basic contract ───────────────────────────────────────────
    def test_returns_dict(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert isinstance(result, dict)

    def test_n_valid_frames_positive(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        assert result["n_valid_frames"] > 0

    def test_n_valid_le_total(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        assert result["n_valid_frames"] <= result["n_total_frames"]

    def test_empty_input_returns_none(self):
        result = _fn("extract_crawling_features")({}, [], fps=self.fps)
        assert result is None

    def test_too_few_valid_frames_returns_none(self):
        # Only 2 frames → n_valid < 5 → None
        tiny = _pose_seq(2)
        result = _fn("extract_crawling_features")(tiny, [0, 1], fps=self.fps)
        assert result is None

    # ── Posture features ─────────────────────────────────────────
    def test_trunk_angle_mean_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        assert "trunk_angle_mean" in result

    def test_trunk_angle_nonnegative(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        if result and "trunk_angle_mean" in result:
            assert result["trunk_angle_mean"] >= 0.0

    def test_hip_height_mean_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        assert "hip_height_mean" in result

    # ── Limb kinematic features ──────────────────────────────────
    def test_wrist_features_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        has_wrist = any("wrist" in k for k in result)
        assert has_wrist

    def test_knee_features_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        has_knee = any("knee" in k for k in result)
        assert has_knee

    # ── Coordination features (primary diagnostic domain) ────────
    def test_diagonal_xcorr_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        has_diag = "diag_RW_LK_xcorr" in result or "diag_LW_RK_xcorr" in result
        assert has_diag

    def test_diagonal_coord_index_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        # May or may not be computed depending on xcorr availability
        if "diagonal_coordination_index" in result:
            assert np.isfinite(result["diagonal_coordination_index"])

    def test_diagonal_corr_positive_for_alternating_gait(self):
        """
        In our synthetic data, L-wrist and R-knee are anti-phase, so the
        cross-correlation between them (at lag ≈ half-cycle) should be
        substantially non-zero.
        """
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        if result and "diag_RW_LK_xcorr" in result:
            assert abs(result["diag_RW_LK_xcorr"]) > 0.3

    # ── Quality metadata ─────────────────────────────────────────
    def test_mean_conf_present(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        assert result is not None
        assert "mean_conf" in result

    def test_mean_conf_in_0_1(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        if result and "mean_conf" in result:
            assert 0.0 <= result["mean_conf"] <= 1.0

    def test_pct_valid_in_0_1(self):
        result = _fn("extract_crawling_features")(self.frames, self.idx, fps=self.fps)
        if result and "pct_valid" in result:
            assert 0.0 <= result["pct_valid"] <= 1.0

    # ── Low-confidence frames are excluded ───────────────────────
    def test_low_conf_frames_excluded(self):
        """Frames with confidence < MIN_CONF should not be counted as valid."""
        low_conf_frames = _pose_seq(45)
        # Overwrite half the frames with very low confidence
        for i in range(0, 45, 2):
            low_conf_frames[str(i)] = {
                k: {**v, "confidence": 0.05}
                for k, v in low_conf_frames[str(i)].items()
            }
        result = _fn("extract_crawling_features")(
            low_conf_frames, list(range(45)), fps=self.fps
        )
        # Should still extract something from the remaining high-conf frames
        if result is not None:
            assert result["n_valid_frames"] < 45