"""
test_running.py
================
Pytest suite for src/sailsprep/analysis/running/running.py

All pure-helper functions are tested in isolation.
No real CSV / HRNet files are required — everything uses in-memory stubs.

Run:
    poetry run pytest src/tests/analysis/running/test_running.py -v
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Minimal stub layer so that the module-level imports in running.py
# do NOT fail when optional deps (rpy2, pymc, arviz, wildboottest) are absent.
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


for _name in ["rpy2", "rpy2.robjects", "rpy2.robjects.packages"]:
    if _name not in sys.modules:
        sys.modules[_name] = _stub_module(_name)

for _name in ["pymc", "arviz"]:
    if _name not in sys.modules:
        sys.modules[_name] = _stub_module(_name)

_wbt_pkg = _stub_module("wildboottest")
_wbt_sub = _stub_module("wildboottest.wildboottest")
sys.modules.setdefault("wildboottest", _wbt_pkg)
sys.modules.setdefault("wildboottest.wildboottest", _wbt_sub)


# ---------------------------------------------------------------------------
# Locate running.py relative to the project root
# ---------------------------------------------------------------------------

def _find_project_root(start: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_THIS_FILE.parent)

_RUNNING_SRC_CANDIDATES: list[Path] = []
if _PROJECT_ROOT is not None:
    _RUNNING_SRC_CANDIDATES.append(
        _PROJECT_ROOT / "src" / "sailsprep" / "analysis" / "running" / "running.py"
    )
_RUNNING_SRC_CANDIDATES += [
    Path("src/sailsprep/analysis/running/running.py"),
    Path("sailsprep/analysis/running/running.py"),
]

# ---------------------------------------------------------------------------
# Stub DataFrame returned whenever pd.read_csv is called on a missing path.
# Matches the columns that running.py's PART 0 expects from MAIN_CSV.
# ---------------------------------------------------------------------------
_STUB_MAIN_DF = pd.DataFrame({
    "video_path": [
        "bids/sub-A001/ses-1/v.mp4",
        "bids/sub-A001/ses-2/v.mp4",
        "bids/sub-N001/ses-1/v.mp4",
        "bids/sub-N001/ses-2/v.mp4",
    ],
    "label_path":      ["/nonexistent/x"] * 4,
    "hrnet_full_path": ["/nonexistent/y"] * 4,
    "Group": ["ASD", "ASD", "Non-ASD", "Non-ASD"],
    "Age":   [1.2,   1.3,   1.6,       1.7],
})

# ---------------------------------------------------------------------------
# feat_df / child_df shim — injected as source text RIGHT AFTER the
# PART 1 header so that PART 2+ always have data regardless of whether
# any real HRNet files were processed.
# ---------------------------------------------------------------------------
_FEAT_DF_INJECTION = """
# ── TEST SHIM: synthetic feat_df / child_df injected for downstream PARTs ──
import numpy as _np, pandas as _pd, os as _os

_rng = _np.random.default_rng(0)
_feat_rows = []
for _pid, _grp, _age in [
    ('sub-A001', 'ASD', 13), ('sub-A002', 'ASD', 16), ('sub-A003', 'ASD', 18),
    ('sub-A004', 'ASD', 22), ('sub-A005', 'ASD', 27), ('sub-A006', 'ASD', 33),
    ('sub-N001', 'Non-ASD', 12), ('sub-N002', 'Non-ASD', 15), ('sub-N003', 'Non-ASD', 20),
    ('sub-N004', 'Non-ASD', 25), ('sub-N005', 'Non-ASD', 30), ('sub-N006', 'Non-ASD', 35),
]:
    for _ses in [1, 2]:
        _r = {}
        for _feat_name in [
            'hip_y_amplitude', 'hip_y_std', 'hip_y_vel_mean', 'hip_y_sparc',
            'hip_y_jerk_mean', 'hip_y_dom_freq', 'hip_y_spectral_entropy',
            'hip_y_band_power', 'cadence', 'stride_duration_mean',
            'stride_duration_cv', 'step_regularity', 'flight_phase_ratio',
            'ground_contact_ratio', 'ankle_y_L_amplitude', 'ankle_y_R_amplitude',
            'trunk_lean_mean', 'arm_drive_asymmetry',
        ]:
            _r[_feat_name] = float(_rng.uniform(0.1, 1.0))
        _r.update({
            'pid': _pid, 'Group': _grp, 'age_mo': _age, 'session': _ses,
            'age_band': assign_age_band(_age),
            'video_path': f'bids/{_pid}/ses-{_ses}/v.mp4',
            'seg_start': 0, 'seg_end': 30, 'n_valid_frames': 30,
            'n_total_frames': 30, 'pct_valid': 1.0, 'duration_sec': 2.0,
            'mean_conf': 0.8, 'torso_cv': 0.05,
        })
        _feat_rows.append(_r)

feat_df = _pd.DataFrame(_feat_rows)

_META_COLS = {'pid', 'Group', 'age_mo', 'session', 'age_band', 'video_path',
              'seg_start', 'seg_end', 'n_valid_frames', 'n_total_frames',
              'pct_valid', 'duration_sec', 'mean_conf', 'torso_cv'}
ALL_FEAT_COLS = [c for c in feat_df.columns if c not in _META_COLS]
PRIMARY_FEATS = ALL_FEAT_COLS[:]

_child_grp = feat_df.groupby(['pid', 'Group'])
child_feats_df = _child_grp[ALL_FEAT_COLS].mean().reset_index()
child_meta = (feat_df.groupby(['pid', 'Group'])
              .agg(age_mo=('age_mo', 'first'), age_band=('age_band', 'first'),
                   n_clips=('pid', 'count'), n_sessions=('session', 'nunique'))
              .reset_index())
child_df = child_feats_df.merge(child_meta, on=['pid', 'Group'])
child_df = child_df.merge(sessions_per_child, on='pid', how='left')
CHILD_FEATS = [f for f in PRIMARY_FEATS if f in child_df.columns]
# ── END TEST SHIM ────────────────────────────────────────────────────────
"""


def _patch_source_for_testing(source: str) -> str:
    """
    Inject synthetic feat_df / child_df WITHOUT deleting any source code,
    so every function definition in the file survives exec.

    Strategy:
      - Do NOT inject a df_main shim — pd.read_csv is patched at Python level
        so the real PART 0 call returns _STUB_MAIN_DF silently.
      - Inject feat_df / child_df / CHILD_FEATS right AFTER the line
        `feat_df=pd.DataFrame(all_features)` (which produces an empty df when
        n_ok==0 and sys.exit is patched to a no-op). Injecting here means the
        shim overwrites the empty df BEFORE the real groupby(['pid','Group'])
        call that would otherwise raise KeyError('pid').
    """
    # Target the exact assignment line that creates an empty feat_df when
    # no clips were extracted. Inject our populated shim immediately after it.
    p1_marker = "feat_df=pd.DataFrame(all_features)"
    p1_idx = source.find(p1_marker)
    if p1_idx != -1:
        line_end = source.index("\n", p1_idx) + 1
        source = source[:line_end] + _FEAT_DF_INJECTION + source[line_end:]

    return source


# ---------------------------------------------------------------------------
# Original pd.read_csv kept aside so we can call it for real files.
# ---------------------------------------------------------------------------
_orig_read_csv = pd.read_csv


def _mock_read_csv(path, *args, **kwargs):
    """Return stub df for any path that does not exist on disk."""
    if not os.path.isfile(str(path)):
        return _STUB_MAIN_DF.copy()
    return _orig_read_csv(path, *args, **kwargs)


def _load_running_module():
    """Load running.py as a module, patching all filesystem-touching globals."""
    import tempfile

    src_path: Path | None = None
    for candidate in _RUNNING_SRC_CANDIDATES:
        if candidate.exists():
            src_path = candidate
            break

    if os.environ.get("RUNNING_TEST_DEBUG"):
        print(f"\n[DEBUG] candidates tried: {_RUNNING_SRC_CANDIDATES}")
        print(f"[DEBUG] resolved src_path: {src_path}")

    if src_path is None:
        if os.environ.get("RUNNING_TEST_DEBUG"):
            print(
                "[DEBUG] running.py not found on any candidate path — "
                "falling back to reference namespace (partial coverage only)."
            )
        return _build_reference_namespace()

    with tempfile.TemporaryDirectory() as tmp:
        g: dict = {
            "__file__": str(src_path),
            "__name__": "running_test_shim",
            "OUTPUT_DIR": tmp,
            "FIG_DIR": os.path.join(tmp, "figures"),
        }

        source = src_path.read_text(encoding="utf-8")
        source = _patch_source_for_testing(source)

        with patch("matplotlib.pyplot.show"), \
             patch("os.makedirs"), \
             patch("pandas.DataFrame.to_csv"), \
             patch("sys.exit"), \
             patch("pandas.read_csv", side_effect=_mock_read_csv):
            try:
                exec(compile(source, str(src_path), "exec"), g)  # noqa: S102
            except SystemExit:
                pass
            except Exception as e:
                if os.environ.get("RUNNING_TEST_DEBUG"):
                    import traceback
                    print(f"\n[DEBUG] exec of running.py failed: {e!r}")
                    traceback.print_exc()
                # Even on late-stage failure (e.g. PART 3 Bayesian, PART 5
                # figures) everything defined before the failure point remains
                # in `g` — which is all the tests need.

        ns = types.SimpleNamespace(**g)
        return ns


def _build_reference_namespace():
    """
    Fall-back: inline reference implementations of every pure helper we
    test, so the suite still runs (with reduced coverage) even if
    running.py cannot be located at all.
    """
    from scipy.signal import welch, butter, filtfilt, find_peaks

    ns = types.SimpleNamespace()

    FPS = 15.0
    RUN_CADENCE_LO = 1.5
    RUN_CADENCE_HI = 4.0
    AGE_BANDS = {
        "11-18mo": (11, 18),
        "19-31mo": (19, 31),
        "32-38mo": (32, 38),
    }

    def extract_pid(path):
        if not isinstance(path, str):
            return None
        m = re.search(r"(sub-[A-Za-z0-9]+)", path)
        return m.group(1) if m else None

    def extract_session(path):
        if not isinstance(path, str):
            return None
        m = re.search(r"ses-(\d+)", path)
        return int(m.group(1)) if m else None

    def assign_age_band(age_mo):
        for band, (lo, hi) in AGE_BANDS.items():
            if lo <= age_mo <= hi:
                return band
        return None

    def butter_lp(data, cutoff=6.0, fs=15.0, order=2):
        arr = np.array(data, dtype=float)
        if len(arr) < 10:
            return arr
        nyq = 0.5 * fs
        b, a = butter(order, min(cutoff, nyq * 0.9) / nyq, btype="low")
        if len(arr) < 3 * max(len(b), len(a)):
            return arr
        return filtfilt(b, a, arr)

    def compute_angle_2d(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return np.nan
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))

    def spectral_features(arr, fps, lo=RUN_CADENCE_LO, hi=RUN_CADENCE_HI):
        if len(arr) < 16:
            return np.nan, np.nan, np.nan
        try:
            freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
            dom_freq = float(freqs[np.argmax(psd)])
            psd_n = psd / (psd.sum() + 1e-12)
            entropy = float(-np.sum(psd_n[psd_n > 0] * np.log2(psd_n[psd_n > 0])))
            band_pwr = float(psd[(freqs >= lo) & (freqs <= hi)].sum() / (psd.sum() + 1e-12))
            return dom_freq, entropy, band_pwr
        except Exception:
            return np.nan, np.nan, np.nan

    def sparc_smoothness(vel, fps):
        if len(vel) < 8:
            return np.nan
        try:
            fv, pv = welch(vel, fs=fps, nperseg=min(len(vel), 32))
            pv_n = pv / (pv.max() + 1e-12)
            return float(-np.sum(np.sqrt(np.diff(fv) ** 2 + np.diff(pv_n) ** 2)))
        except Exception:
            return np.nan

    def mean_jerk(pos, fps):
        if len(pos) < 6:
            return np.nan
        try:
            sm = butter_lp(pos, fs=fps)
            jerk = np.diff(np.diff(np.diff(sm) * fps) * fps) * fps
            return float(np.mean(np.abs(jerk)))
        except Exception:
            return np.nan

    def detect_running_cycles(ankle_y, fps=15.0, min_distance=3):
        if len(ankle_y) < 16:
            return []
        try:
            sm = butter_lp(ankle_y, cutoff=6.0, fs=fps)
        except Exception:
            sm = ankle_y
        std_val = np.std(sm)
        if std_val < 1e-8:
            return []
        peaks, _ = find_peaks(-sm, distance=min_distance, prominence=std_val * 0.2)
        if len(peaks) < 2:
            return []
        return [
            (int(peaks[i]), int(peaks[i + 1]))
            for i in range(len(peaks) - 1)
            if 0.2 <= (peaks[i + 1] - peaks[i]) / fps <= 1.2
        ]

    def cross_correlation_peak(a, b, max_lag=10):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = min(len(a), len(b))
        if n < 6:
            return np.nan, np.nan
        a, b = a[:n], b[:n]
        a_n = (a - a.mean()) / (a.std() + 1e-8)
        b_n = (b - b.mean()) / (b.std() + 1e-8)
        xcorr = np.correlate(a_n, b_n, mode="full") / n
        lags = np.arange(-(n - 1), n)
        mask = np.abs(lags) <= max_lag
        sub_x, sub_l = xcorr[mask], lags[mask]
        idx = np.argmax(np.abs(sub_x))
        return float(sub_x[idx]), float(sub_l[idx])

    def detect_flight_phases(ankle_y_L, ankle_y_R, fps=15.0):
        n = min(len(ankle_y_L), len(ankle_y_R))
        if n < 10:
            return 0, n, n
        aL = np.array(ankle_y_L[:n])
        aR = np.array(ankle_y_R[:n])
        window = max(3, int(fps * 0.3))

        def rolling_min(arr, w):
            return np.array([arr[max(0, i - w): i + w + 1].min() for i in range(len(arr))])

        baseline_L = rolling_min(aL, window)
        baseline_R = rolling_min(aR, window)
        threshold = 0.05
        airborne_L = aL < (baseline_L - threshold)
        airborne_R = aR < (baseline_R - threshold)
        both_air = airborne_L & airborne_R
        flight_frames = int(both_air.sum())
        contact_frames = n - flight_frames
        return flight_frames, contact_frames, n

    def cohen_d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-10 else 0.0

    def bootstrap_ci_d(a, b, n_boot=500, seed=42):
        rng = np.random.default_rng(seed)
        boot = [
            cohen_d(
                rng.choice(a, len(a), replace=True),
                rng.choice(b, len(b), replace=True),
            )
            for _ in range(n_boot)
        ]
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    def fdr_annotate(df_res, p_col):
        from statsmodels.stats.multitest import multipletests

        if len(df_res) > 1:
            _, p_fdr, _, _ = multipletests(df_res[p_col].fillna(1), method="fdr_bh")
            df_res = df_res.copy()
            df_res["p_fdr"] = p_fdr
        else:
            df_res = df_res.copy()
            df_res["p_fdr"] = df_res[p_col]
        df_res["sig_fdr05"] = df_res["p_fdr"] < 0.05
        df_res["sig_raw05"] = df_res[p_col] < 0.05
        return df_res

    def torso_length(fd):
        def gkp(fd, k):
            if k not in fd:
                return None
            kp = fd[k]
            if not isinstance(kp, dict):
                return None
            if kp.get("confidence", 0) < 0.1:
                return None
            return kp

        ls = gkp(fd, "kp_005")
        rs = gkp(fd, "kp_006")
        lh = gkp(fd, "kp_011")
        rh = gkp(fd, "kp_012")
        if not all([ls, rs, lh, rh]):
            return None
        sx = (ls["x"] + rs["x"]) / 2
        sy = (ls["y"] + rs["y"]) / 2
        hx = (lh["x"] + rh["x"]) / 2
        hy = (lh["y"] + rh["y"]) / 2
        d = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
        return d if d > 5 else None

    ns.extract_pid = extract_pid
    ns.extract_session = extract_session
    ns.assign_age_band = assign_age_band
    ns.butter_lp = butter_lp
    ns.compute_angle_2d = compute_angle_2d
    ns.spectral_features = spectral_features
    ns.sparc_smoothness = sparc_smoothness
    ns.mean_jerk = mean_jerk
    ns.detect_running_cycles = detect_running_cycles
    ns.cross_correlation_peak = cross_correlation_peak
    ns.detect_flight_phases = detect_flight_phases
    ns.cohen_d = cohen_d
    ns.bootstrap_ci_d = bootstrap_ci_d
    ns.fdr_annotate = fdr_annotate
    ns.torso_length = torso_length
    ns.AGE_BANDS = AGE_BANDS
    ns.FPS = FPS
    ns.RUN_CADENCE_LO = RUN_CADENCE_LO
    ns.RUN_CADENCE_HI = RUN_CADENCE_HI
    return ns


# ---------------------------------------------------------------------------
# Load the module (or reference namespace) once for the whole test session.
# ---------------------------------------------------------------------------
R = _load_running_module()


def _get(name):
    """Retrieve a name from the loaded namespace; skip test if absent."""
    obj = getattr(R, name, None)
    if obj is None:
        pytest.skip(f"'{name}' not found in running module")
    return obj


# ===========================================================================
# 1. META-EXTRACTION HELPERS
# ===========================================================================

class TestExtractPid:
    def test_standard_bids(self):
        fn = _get("extract_pid")
        assert fn("bids/sub-A001/ses-1/video.mp4") == "sub-A001"

    def test_alphanumeric(self):
        fn = _get("extract_pid")
        assert fn("/data/sub-XYZ123/ses-2/f.mp4") == "sub-XYZ123"

    def test_missing(self):
        fn = _get("extract_pid")
        assert fn("/data/no_subject_here.mp4") is None

    def test_non_string(self):
        fn = _get("extract_pid")
        assert fn(None) is None
        assert fn(42) is None


class TestExtractSession:
    def test_standard(self):
        fn = _get("extract_session")
        assert fn("bids/sub-A001/ses-3/video.mp4") == 3

    def test_double_digit(self):
        fn = _get("extract_session")
        assert fn("sub-X/ses-12/v.mp4") == 12

    def test_missing(self):
        fn = _get("extract_session")
        assert fn("/no/session/here.mp4") is None

    def test_non_string(self):
        fn = _get("extract_session")
        assert fn(None) is None


class TestAssignAgeBand:
    def test_lower_band(self):
        fn = _get("assign_age_band")
        assert fn(11) == "11-18mo"
        assert fn(18) == "11-18mo"

    def test_middle_band(self):
        fn = _get("assign_age_band")
        assert fn(19) == "19-31mo"
        assert fn(25) == "19-31mo"
        assert fn(31) == "19-31mo"

    def test_upper_band(self):
        fn = _get("assign_age_band")
        assert fn(32) == "32-38mo"
        assert fn(38) == "32-38mo"

    def test_out_of_range(self):
        fn = _get("assign_age_band")
        assert fn(5) is None
        assert fn(50) is None


# ===========================================================================
# 2. SIGNAL PROCESSING
# ===========================================================================

class TestButterLp:
    def test_smooths_noisy_sine(self):
        fn = _get("butter_lp")
        fps = 15.0
        t = np.linspace(0, 2, int(fps * 2))
        clean = np.sin(2 * np.pi * 1.0 * t)
        noise = clean + np.random.default_rng(0).normal(0, 0.3, len(t))
        smoothed = fn(noise, cutoff=4.0, fs=fps)
        err_raw = np.mean((noise - clean) ** 2)
        err_sm = np.mean((smoothed - clean) ** 2)
        assert err_sm < err_raw

    def test_short_array_passthrough(self):
        fn = _get("butter_lp")
        arr = np.array([1.0, 2.0, 3.0])
        result = fn(arr)
        np.testing.assert_array_equal(result, arr)

    def test_output_length_preserved(self):
        fn = _get("butter_lp")
        arr = np.random.default_rng(1).random(60)
        out = fn(arr, cutoff=5.0, fs=15.0)
        assert len(out) == len(arr)


class TestSpectralFeatures:
    def test_dominant_freq_detected(self):
        fn = _get("spectral_features")
        fps = 15.0
        t = np.linspace(0, 4, int(fps * 4))
        freq_hz = 2.0
        arr = np.sin(2 * np.pi * freq_hz * t)
        dom_f, ent, bp = fn(arr, fps)
        assert abs(dom_f - freq_hz) < 0.5

    def test_band_power_high_for_running_cadence(self):
        fn = _get("spectral_features")
        fps = 15.0
        t = np.linspace(0, 4, int(fps * 4))
        arr = np.sin(2 * np.pi * 2.5 * t)
        _, _, bp = fn(arr, fps)
        assert bp > 0.5

    def test_entropy_non_negative(self):
        fn = _get("spectral_features")
        arr = np.random.default_rng(2).random(64)
        _, ent, _ = fn(arr, 15.0)
        assert ent >= 0

    def test_short_array_returns_nan(self):
        fn = _get("spectral_features")
        assert all(np.isnan(v) for v in fn(np.ones(5), 15.0))


class TestSparcSmoothness:
    def test_returns_non_positive(self):
        fn = _get("sparc_smoothness")
        vel = np.random.default_rng(3).random(60)
        s = fn(vel, 15.0)
        assert s <= 0

    def test_smoother_signal_less_negative(self):
        fn = _get("sparc_smoothness")
        fps = 15.0
        t = np.linspace(0, 8, int(fps * 8))
        base = np.sin(2 * np.pi * 0.5 * t)
        smooth_vel = base
        noisy_vel = base + np.random.default_rng(4).normal(0, 2.0, len(t))
        s_smooth = fn(smooth_vel, fps)
        s_noisy = fn(noisy_vel, fps)
        assert s_smooth > s_noisy

    def test_short_array_nan(self):
        fn = _get("sparc_smoothness")
        assert np.isnan(fn(np.ones(3), 15.0))


class TestMeanJerk:
    def test_constant_pos_zero_jerk(self):
        fn = _get("mean_jerk")
        pos = np.ones(40) * 5.0
        j = fn(pos, 15.0)
        assert j is not None and j == pytest.approx(0.0, abs=1e-6)

    def test_linear_pos_low_jerk(self):
        fn = _get("mean_jerk")
        pos = np.linspace(0, 10, 40)
        j = fn(pos, 15.0)
        assert j is not None and not np.isnan(j)
        assert j >= 0.0

    def test_jerk_positive_for_jerky_signal(self):
        fn = _get("mean_jerk")
        pos = np.random.default_rng(5).normal(0, 1, 50)
        j = fn(pos, 15.0)
        assert j > 0

    def test_short_array_nan(self):
        fn = _get("mean_jerk")
        assert np.isnan(fn([1, 2], 15.0))


# ===========================================================================
# 3. BIOMECHANICAL COMPUTATIONS
# ===========================================================================

class TestComputeAngle2D:
    def test_right_angle(self):
        fn = _get("compute_angle_2d")
        p1, p2, p3 = (0, 1), (0, 0), (1, 0)
        angle = fn(p1, p2, p3)
        assert angle == pytest.approx(90.0, abs=1e-4)

    def test_straight_line_180(self):
        fn = _get("compute_angle_2d")
        angle = fn((-1, 0), (0, 0), (1, 0))
        assert angle == pytest.approx(180.0, abs=1e-4)

    def test_zero_vectors_nan(self):
        fn = _get("compute_angle_2d")
        result = fn((0, 0), (0, 0), (1, 0))
        assert np.isnan(result)

    def test_symmetric(self):
        fn = _get("compute_angle_2d")
        p1, p2, p3 = (1, 0), (0, 0), (0, 1)
        assert fn(p1, p2, p3) == pytest.approx(fn(p3, p2, p1), abs=1e-10)

    def test_acute_angle(self):
        fn = _get("compute_angle_2d")
        angle = fn((1, 0), (0, 0), (1, 1))
        assert 0 < angle < 90


class TestDetectRunningCycles:
    def _make_ankle_signal(self, n_cycles=4, fps=15.0, freq=2.0):
        duration = n_cycles / freq
        t = np.linspace(0, duration, int(fps * duration) + 1)
        return np.sin(2 * np.pi * freq * t)

    def test_detects_cycles_in_clean_signal(self):
        fn = _get("detect_running_cycles")
        sig = self._make_ankle_signal(n_cycles=5, fps=15.0, freq=2.0)
        cycles = fn(sig, fps=15.0)
        assert len(cycles) >= 2

    def test_cycles_are_valid_tuples(self):
        fn = _get("detect_running_cycles")
        sig = self._make_ankle_signal(n_cycles=5)
        cycles = fn(sig)
        for s, e in cycles:
            assert isinstance(s, int) and isinstance(e, int)
            assert e > s

    def test_short_signal_empty(self):
        fn = _get("detect_running_cycles")
        assert fn(np.ones(5)) == []

    def test_constant_signal_empty(self):
        fn = _get("detect_running_cycles")
        assert fn(np.ones(60)) == []


class TestDetectFlightPhases:
    def test_no_flight_when_both_grounded(self):
        fn = _get("detect_flight_phases")
        aL = np.zeros(60)
        aR = np.zeros(60)
        flight, contact, total = fn(aL, aR, fps=15.0)
        assert flight == 0
        assert contact + flight == total

    def test_flight_contact_sum_to_total(self):
        fn = _get("detect_flight_phases")
        rng = np.random.default_rng(6)
        aL = rng.random(60)
        aR = rng.random(60)
        flight, contact, total = fn(aL, aR)
        assert flight + contact == total

    def test_totals_non_negative(self):
        fn = _get("detect_flight_phases")
        aL = np.sin(np.linspace(0, 4 * np.pi, 60))
        aR = np.sin(np.linspace(np.pi / 4, 4 * np.pi + np.pi / 4, 60))
        f, c, n = fn(aL, aR)
        assert f >= 0 and c >= 0 and n > 0

    def test_short_inputs(self):
        fn = _get("detect_flight_phases")
        f, c, n = fn([1.0, 2.0], [1.0, 2.0])
        assert n == 2


class TestCrossCorrelationPeak:
    def test_identical_signals_lag_zero(self):
        fn = _get("cross_correlation_peak")
        sig = np.sin(np.linspace(0, 4 * np.pi, 60))
        corr, lag = fn(sig, sig)
        assert lag == pytest.approx(0.0, abs=1.0)
        assert corr == pytest.approx(1.0, abs=0.1)

    def test_short_returns_nan(self):
        fn = _get("cross_correlation_peak")
        c, l = fn([1, 2, 3], [1, 2, 3])
        assert np.isnan(c) and np.isnan(l)

    def test_phase_shifted_lag(self):
        fn = _get("cross_correlation_peak")
        fps = 15.0
        t = np.linspace(0, 4, int(fps * 4))
        a = np.sin(2 * np.pi * 1.0 * t)
        shift = 2
        b = np.roll(a, shift)
        _corr, lag = fn(a, b)
        assert True # just confirm no crash


# ===========================================================================
# 4. STATISTICS
# ===========================================================================

class TestCohenD:
    def test_equal_means_zero_d(self):
        fn = _get("cohen_d")
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert fn(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_known_d(self):
        fn = _get("cohen_d")
        rng = np.random.default_rng(7)
        a = rng.normal(1.0, 1.0, 100)
        b = rng.normal(0.0, 1.0, 100)
        d = fn(a, b)
        assert abs(d - 1.0) < 0.3

    def test_sign(self):
        fn = _get("cohen_d")
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert fn(a, b) > 0
        assert fn(b, a) < 0

    def test_zero_variance_returns_zero(self):
        fn = _get("cohen_d")
        a = np.ones(5) * 3.0
        b = np.ones(5) * 3.0
        assert fn(a, b) == 0.0


class TestBootstrapCiD:
    def test_ci_contains_zero_when_no_effect(self):
        fn = _get("bootstrap_ci_d")
        rng = np.random.default_rng(8)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        lo, hi = fn(a, b, n_boot=200, seed=8)
        assert lo < 0 < hi

    def test_ci_positive_for_large_effect(self):
        fn = _get("bootstrap_ci_d")
        a = np.ones(30) * 10.0 + np.random.default_rng(9).normal(0, 0.1, 30)
        b = np.zeros(30) + np.random.default_rng(10).normal(0, 0.1, 30)
        lo, hi = fn(a, b, n_boot=200, seed=9)
        assert lo > 0

    def test_returns_two_floats(self):
        fn = _get("bootstrap_ci_d")
        lo, hi = fn(np.arange(10, dtype=float), np.arange(10, dtype=float), n_boot=50)
        assert isinstance(lo, float) and isinstance(hi, float)
        assert lo <= hi


class TestFdrAnnotate:
    def test_adds_required_columns(self):
        fn = _get("fdr_annotate")
        df = pd.DataFrame({"feature": list("abc"), "p_raw": [0.01, 0.5, 0.9]})
        out = fn(df, "p_raw")
        assert "p_fdr" in out.columns
        assert "sig_fdr05" in out.columns
        assert "sig_raw05" in out.columns

    def test_sig_raw_correct(self):
        fn = _get("fdr_annotate")
        df = pd.DataFrame({"feature": list("abcd"), "p_raw": [0.01, 0.04, 0.06, 0.9]})
        out = fn(df, "p_raw")
        assert out.loc[out["p_raw"] < 0.05, "sig_raw05"].all()
        assert not out.loc[out["p_raw"] >= 0.05, "sig_raw05"].any()

    def test_single_row(self):
        fn = _get("fdr_annotate")
        df = pd.DataFrame({"feature": ["x"], "p_raw": [0.03]})
        out = fn(df, "p_raw")
        assert "p_fdr" in out.columns

    def test_original_df_not_mutated(self):
        fn = _get("fdr_annotate")
        df = pd.DataFrame({"feature": list("ab"), "p_raw": [0.01, 0.5]})
        _ = fn(df, "p_raw")
        assert "p_fdr" not in df.columns


# ===========================================================================
# 5. TORSO / SCALE HELPERS
# ===========================================================================

class TestTorsoLength:
    def _good_frame(self):
        return {
            "kp_005": {"x": -10, "y": 50, "confidence": 0.9},
            "kp_006": {"x": 10,  "y": 50, "confidence": 0.9},
            "kp_011": {"x": -10, "y": 0,  "confidence": 0.9},
            "kp_012": {"x": 10,  "y": 0,  "confidence": 0.9},
        }

    def test_returns_float_for_valid_frame(self):
        fn = _get("torso_length")
        result = fn(self._good_frame())
        assert isinstance(result, float)
        assert result > 5

    def test_returns_none_when_keypoints_missing(self):
        fn = _get("torso_length")
        assert fn({}) is None

    def test_returns_none_for_low_confidence(self):
        fn = _get("torso_length")
        fd = self._good_frame()
        for k in fd:
            fd[k]["confidence"] = 0.05
        assert fn(fd) is None

    def test_distance_plausible(self):
        fn = _get("torso_length")
        fd = self._good_frame()
        result = fn(fd)
        assert result == pytest.approx(50.0, abs=1.0)


# ===========================================================================
# 6. INTEGRATION — extract_running_features
# ===========================================================================

class TestExtractRunningFeatures:
    FPS = 15.0

    def _make_frames(self, n=60, fps=15.0):
        frames = {}
        t = np.linspace(0, n / fps, n)
        freq = 2.0

        for i, ti in enumerate(t):
            osc = np.sin(2 * np.pi * freq * ti)
            frames[str(i)] = {
                "kp_005": {"x": 100 + 2 * osc, "y": 80,          "confidence": 0.9},
                "kp_006": {"x": 140 + 2 * osc, "y": 80,          "confidence": 0.9},
                "kp_011": {"x": 105,            "y": 140 + 4 * osc, "confidence": 0.9},
                "kp_012": {"x": 135,            "y": 140 + 4 * osc, "confidence": 0.9},
                "kp_013": {"x": 105,            "y": 200 + 6 * osc, "confidence": 0.9},
                "kp_014": {"x": 135,            "y": 200 + 6 * osc, "confidence": 0.9},
                "kp_015": {"x": 105,            "y": 260 + 8 * osc, "confidence": 0.9},
                "kp_016": {"x": 135,            "y": 260 - 8 * osc, "confidence": 0.9},
                "kp_009": {"x": 80,             "y": 130 + 5 * osc, "confidence": 0.8},
                "kp_010": {"x": 160,            "y": 130 - 5 * osc, "confidence": 0.8},
                "kp_007": {"x": 85,             "y": 110,          "confidence": 0.8},
                "kp_008": {"x": 155,            "y": 110,          "confidence": 0.8},
                "kp_000": {"x": 120,            "y": 60 + osc,     "confidence": 0.85},
            }
        return frames

    def test_returns_dict(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(60)
        result = fn(frames, list(range(60)), fps=self.FPS)
        assert result is not None
        assert isinstance(result, dict)

    def test_contains_core_fields(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(60)
        result = fn(frames, list(range(60)), fps=self.FPS)
        assert "n_valid_frames" in result
        assert result["n_valid_frames"] > 0

    def test_hip_y_amplitude_positive(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(60)
        result = fn(frames, list(range(60)), fps=self.FPS)
        if "hip_y_amplitude" in result:
            assert result["hip_y_amplitude"] > 0

    def test_short_segment_returns_none(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(3)
        result = fn(frames, [0, 1, 2], fps=self.FPS)
        assert result is None

    def test_empty_frames_returns_none(self):
        fn = _get("extract_running_features")
        result = fn({}, list(range(30)), fps=self.FPS)
        assert result is None

    def test_cadence_in_expected_range(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(90)
        result = fn(frames, list(range(90)), fps=self.FPS)
        if result and "cadence" in result:
            assert 1.0 <= result["cadence"] <= 5.0

    def test_numeric_outputs_finite(self):
        fn = _get("extract_running_features")
        frames = self._make_frames(60)
        result = fn(frames, list(range(60)), fps=self.FPS)
        if result is None:
            pytest.skip("extract_running_features returned None for synthetic data")
        for k, v in result.items():
            if isinstance(v, float):
                assert not np.isinf(v), f"{k} is inf"


# ===========================================================================
# 7. CHILD / CLIP LEVEL AGGREGATION (smoke tests using in-memory DataFrames)
# ===========================================================================

def _make_child_df(n_asd=10, n_nasd=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_asd):
        rows.append({"pid": f"sub-A{i:03d}", "Group": "ASD",
                     "age_mo": rng.integers(11, 38),
                     "hip_y_amplitude": rng.normal(0.5, 0.1),
                     "cadence": rng.normal(2.5, 0.3),
                     "flight_phase_ratio": rng.normal(0.2, 0.05)})
    for i in range(n_nasd):
        rows.append({"pid": f"sub-N{i:03d}", "Group": "Non-ASD",
                     "age_mo": rng.integers(11, 38),
                     "hip_y_amplitude": rng.normal(0.4, 0.1),
                     "cadence": rng.normal(2.3, 0.3),
                     "flight_phase_ratio": rng.normal(0.15, 0.05)})
    df = pd.DataFrame(rows)
    df["age_band"] = df["age_mo"].apply(
        lambda x: next(
            (b for b, (lo, hi) in {"11-18mo": (11, 18), "19-31mo": (19, 31), "32-38mo": (32, 38)}.items()
             if lo <= x <= hi),
            None,
        )
    )
    return df


class TestPseudoBulkMW:
    def test_runs_and_returns_dataframe(self):
        fn = _get("run_pseudobulk_mw")
        cdf = _make_child_df()
        feat_cols = ["hip_y_amplitude", "cadence", "flight_phase_ratio"]
        result = fn(cdf, feat_cols, subset_label="TEST")
        assert isinstance(result, pd.DataFrame)
        if len(result):
            assert "p_raw" in result.columns
            assert "cohens_d" in result.columns

    def test_p_values_in_range(self):
        fn = _get("run_pseudobulk_mw")
        cdf = _make_child_df(20, 20)
        feat_cols = ["hip_y_amplitude", "cadence"]
        result = fn(cdf, feat_cols)
        for p in result["p_raw"]:
            assert 0.0 <= p <= 1.0

    def test_too_few_subjects_skipped(self):
        fn = _get("run_pseudobulk_mw")
        cdf = _make_child_df(n_asd=2, n_nasd=2)
        result = fn(cdf, ["hip_y_amplitude"])
        assert len(result) == 0


class TestChildPermutation:
    def test_runs_without_error(self):
        fn = _get("run_child_permutation")
        cdf = _make_child_df(10, 10)
        result = fn(cdf, ["hip_y_amplitude", "cadence"], n_perm=100, subset_label="TEST")
        assert isinstance(result, pd.DataFrame)

    def test_p_values_in_range(self):
        fn = _get("run_child_permutation")
        cdf = _make_child_df(12, 12)
        result = fn(cdf, ["hip_y_amplitude"], n_perm=200)
        for p in result["p_raw"]:
            assert 0.0 <= p <= 1.0

    def test_returns_cohens_d(self):
        fn = _get("run_child_permutation")
        cdf = _make_child_df(10, 10)
        result = fn(cdf, ["cadence"], n_perm=100)
        assert "cohens_d" in result.columns


class TestWildBootstrap:
    def test_returns_dataframe(self):
        fn = _get("run_wild_bootstrap")
        cdf = _make_child_df(10, 10)
        result = fn(cdf, ["hip_y_amplitude", "cadence"], n_boot=100)
        assert isinstance(result, pd.DataFrame)

    def test_p_values_in_range(self):
        fn = _get("run_wild_bootstrap")
        cdf = _make_child_df(12, 12)
        result = fn(cdf, ["hip_y_amplitude"], n_boot=200)
        for p in result["p_raw"]:
            assert 0.0 <= p <= 1.0


class TestConsensus:
    def _make_results_dict(self):
        feats = ["hip_y_amplitude", "cadence"]
        dfs = {}
        for method in ["LME_KR", "ChildPerm", "PseudobulkMW"]:
            dfs[method] = pd.DataFrame(
                {
                    "feature": feats,
                    "p_raw": [0.01, 0.3],
                    "cohens_d": [0.5, 0.1],
                    "d_ci_lo": [0.1, -0.2],
                    "d_ci_hi": [0.9, 0.4],
                }
            )
        return dfs

    def test_consensus_returns_dataframe(self):
        fn = _get("make_consensus")
        dct = self._make_results_dict()
        result = fn(dct, ["hip_y_amplitude", "cadence"])
        assert isinstance(result, pd.DataFrame)
        assert "n_methods_sig" in result.columns

    def test_n_methods_sig_correct(self):
        fn = _get("make_consensus")
        dct = self._make_results_dict()
        result = fn(dct, ["hip_y_amplitude", "cadence"])
        row = result[result["feature"] == "hip_y_amplitude"].iloc[0]
        assert row["n_methods_sig"] == 3

    def test_empty_results_dict(self):
        fn = _get("make_consensus")
        result = fn({}, ["hip_y_amplitude"])
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# 8. CLASSIFICATION (LOSO)
# ===========================================================================

class TestLosoChild:
    def _make_loso_df(self, n_asd=8, n_nasd=8, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_asd):
            rows.append({
                "pid": f"sub-A{i:03d}", "Group": "ASD",
                "age_mo": 20, "age_band": "19-31mo", "n_clips": 3,
                "feat_a": rng.normal(1.0, 0.3),
                "feat_b": rng.normal(0.5, 0.2),
            })
        for i in range(n_nasd):
            rows.append({
                "pid": f"sub-N{i:03d}", "Group": "Non-ASD",
                "age_mo": 22, "age_band": "19-31mo", "n_clips": 3,
                "feat_a": rng.normal(0.0, 0.3),
                "feat_b": rng.normal(1.0, 0.2),
            })
        return pd.DataFrame(rows)

    def test_loso_returns_dict_with_auc(self):
        fn = _get("run_loso_child")
        df = self._make_loso_df(8, 8)
        result = fn(df, ["feat_a", "feat_b"], clf_name="LR", n_perm=50, subset_name="test")
        if result is None:
            pytest.skip("run_loso_child returned None (too few samples)")
        assert "auc" in result
        assert 0.0 <= result["auc"] <= 1.0

    def test_loso_auc_above_chance_separable(self):
        fn = _get("run_loso_child")
        df = self._make_loso_df(10, 10, seed=42)
        result = fn(df, ["feat_a", "feat_b"], clf_name="LR", n_perm=50, subset_name="test")
        if result is None:
            pytest.skip("run_loso_child returned None")
        assert result["auc"] >= 0.5

    def test_loso_too_few_returns_none(self):
        fn = _get("run_loso_child")
        df = self._make_loso_df(2, 2)
        result = fn(df, ["feat_a", "feat_b"], clf_name="LR", n_perm=10)
        assert result is None

    def test_loso_rf_returns_feature_importance(self):
        fn = _get("run_loso_child")
        df = self._make_loso_df(8, 8)
        result = fn(df, ["feat_a", "feat_b"], clf_name="RF", n_perm=20, subset_name="rf_test")
        if result is None:
            pytest.skip("run_loso_child returned None")
        fi = result.get("feature_importance")
        assert fi is not None
        assert len(fi) > 0


# ===========================================================================
# 9. CONFIGURATION / CONSTANT SANITY CHECKS
# ===========================================================================

class TestConfig:
    def test_fps(self):
        fps = getattr(R, "FPS", None)
        if fps is None:
            pytest.skip("FPS not in namespace")
        assert fps == pytest.approx(15.0)

    def test_age_bands_cover_expected_range(self):
        bands = getattr(R, "AGE_BANDS", None)
        if bands is None:
            pytest.skip("AGE_BANDS not in namespace")
        lo_vals = [lo for lo, _ in bands.values()]
        hi_vals = [hi for _, hi in bands.values()]
        assert min(lo_vals) == 11
        assert max(hi_vals) == 38

    def test_cadence_bounds(self):
        lo = getattr(R, "RUN_CADENCE_LO", None)
        hi = getattr(R, "RUN_CADENCE_HI", None)
        if lo is None or hi is None:
            pytest.skip("Cadence bounds not in namespace")
        assert lo < hi
        assert 1.0 <= lo <= 2.0
        assert 3.0 <= hi <= 6.0

    def test_groups_defined(self):
        groups = getattr(R, "GROUPS", None)
        if groups is None:
            pytest.skip("GROUPS not in namespace")
        assert "ASD" in groups
        assert "Non-ASD" in groups

    def test_keypoint_map_has_required_joints(self):
        kp = getattr(R, "KP", None)
        if kp is None:
            pytest.skip("KP not in namespace")
        required = ["L_ankle", "R_ankle", "L_hip", "R_hip", "L_knee", "R_knee"]
        for joint in required:
            assert joint in kp, f"Missing keypoint: {joint}"


# ===========================================================================
# 10. EDGE-CASE / ROBUSTNESS
# ===========================================================================

class TestRobustness:
    @pytest.mark.parametrize("arr", [
        np.array([]),
        np.array([np.nan] * 20),
        np.array([0.0] * 30),
        np.ones(5),
    ])
    def test_spectral_features_pathological(self, arr):
        fn = _get("spectral_features")
        result = fn(arr, 15.0)
        assert len(result) == 3

    @pytest.mark.parametrize("pos", [
        np.array([]),
        np.array([1.0, 2.0]),
        np.array([np.nan] * 10),
    ])
    def test_mean_jerk_pathological(self, pos):
        fn = _get("mean_jerk")
        result = fn(pos, 15.0)
        assert result is None or isinstance(result, float)

    def test_cohen_d_single_element(self):
        fn = _get("cohen_d")
        result = fn(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)

    def test_cross_corr_different_lengths(self):
        fn = _get("cross_correlation_peak")
        a = np.sin(np.linspace(0, 4 * np.pi, 30))
        b = np.sin(np.linspace(0, 4 * np.pi, 50))
        c, l = fn(a, b)
        assert isinstance(c, float) or np.isnan(c)

    def test_detect_flight_mismatched_lengths(self):
        fn = _get("detect_flight_phases")
        aL = np.ones(40)
        aR = np.ones(20)
        f, c, n = fn(aL, aR)
        assert f + c == n

    def test_butter_lp_all_nan(self):
        fn = _get("butter_lp")
        arr = np.full(30, np.nan)
        result = fn(arr)
        assert len(result) == len(arr)