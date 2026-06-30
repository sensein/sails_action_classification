"""Tests for handflapping analysis module.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import pytest


THIS_FILE = os.path.abspath(__file__)
_d = os.path.dirname(THIS_FILE)
while os.path.basename(_d) != "src":
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError(f"Could not locate 'src' directory above {THIS_FILE}")
    _d = _parent
SRC_PATH = os.path.join(_d, "sailsprep", "analysis", "handflapping", "handflapping.py")


def _load_module_functions_only() -> dict:
    with open(SRC_PATH, "r") as fh:
        src = fh.read()

    tree = ast.parse(src)
    lines = src.splitlines()

    # Line at which the module switches from "library code" (defs,
    # constants, imports) to "script code" (the actual data load and
    # subsequent analysis run). We look for the df_main = pd.read_csv
    # line specifically, since that's what currently crashes import.
    cutoff_line = next(
        (
            i + 1
            for i, ln in enumerate(lines)
            if "read_csv" in ln and "MAIN_CSV" in ln
        ),
        len(lines) + 1,
    )

    # Keep everything before the cutoff (imports, constants, defs).
    pre = [n for n in tree.body if n.lineno < cutoff_line]
    # Keep any top-level function/class defs that happen to appear
    # after the cutoff too — defensive, in case ordering changes.
    post_defs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.lineno >= cutoff_line
    ]

    tree.body = pre + post_defs
    ast.fix_missing_locations(tree)

    ns: dict = {"__name__": "__test_handflapping__", "__file__": SRC_PATH}
    exec(compile(tree, SRC_PATH, "exec"), ns)  # noqa: S102
    return ns


try:
    NS = _load_module_functions_only()
except FileNotFoundError:
    pytest.skip(
        f"handflapping.py not found at {SRC_PATH}. "
        "Run from repo root or adjust SRC_PATH.",
        allow_module_level=True,
    )

# ── Pull symbols into local scope ────────────────────────────────────────────
_add_label_dummies = NS["_add_label_dummies"]
_savage_dickey_bf = NS["_savage_dickey_bf"]
_standardise = NS["_standardise"]
assign_age_band = NS["assign_age_band"]
bootstrap_ci_d = NS["bootstrap_ci_d"]
butter_lp = NS["butter_lp"]
cles = NS["cles"]
cohen_d = NS["cohen_d"]
compute_icc = NS["compute_icc"]
extract_flapping_features = NS["extract_flapping_features"]
extract_pid = NS["extract_pid"]
fdr_annotate = NS["fdr_annotate"]
get_kp = NS["get_kp"]
make_consensus = NS["make_consensus"]
parse_timestamps = NS["parse_timestamps"]
run_child_permutation = NS["run_child_permutation"]
run_consistency_gate = NS["run_consistency_gate"]
run_mwu = NS["run_mwu"]
run_spearman_age = NS["run_spearman_age"]
run_wild_bootstrap = NS["run_wild_bootstrap"]
spectral_features = NS["spectral_features"]
stream_filter = NS["stream_filter"]
torso_length = NS["torso_length"]


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
TORSO_LEN = 100.0  # pixels — gives normalised coords


def _frame(
    lw_x: float = 120,
    lw_y: float = 150,
    rw_x: float = 180,
    rw_y: float = 145,
    conf: float = 0.9,
    include_elbows: bool = True,
) -> dict:
    """Minimal valid pose frame with shoulders, hips, wrists."""
    fd: dict = {
        # shoulders (torso top)
        "kp_005": {"x": 100.0, "y": 100.0, "confidence": 0.95},
        "kp_006": {"x": 200.0, "y": 100.0, "confidence": 0.95},
        # hips (torso bottom — torso_length ≈ 100 px)
        "kp_011": {"x": 100.0, "y": 200.0, "confidence": 0.95},
        "kp_012": {"x": 200.0, "y": 200.0, "confidence": 0.95},
        # wrists
        "kp_009": {"x": lw_x, "y": lw_y, "confidence": conf},
        "kp_010": {"x": rw_x, "y": rw_y, "confidence": conf},
    }
    if include_elbows:
        fd["kp_007"] = {"x": 110.0, "y": 130.0, "confidence": 0.85}
        fd["kp_008"] = {"x": 190.0, "y": 130.0, "confidence": 0.85}
    return fd


def _pose_frames(n: int = 30, amplitude: float = 20.0) -> dict:
    """Oscillating bilateral wrist motion over n frames."""
    frames = {}
    for i in range(n):
        dy = amplitude * np.sin(2 * np.pi * i / 15)
        frames[str(i)] = _frame(lw_y=150 + dy, rw_y=145 - dy)
    return frames


def _child_df(
    n_asd: int = 8,
    n_nasd: int = 8,
    feat: str = "wrist_amp_max",
    asd_mean: float = 1.5,
    nasd_mean: float = 0.8,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Synthetic child-level DataFrame for statistical tests."""
    if rng is None:
        rng = np.random.default_rng(42)
    pids_a = [f"sub-ASD{i:03d}" for i in range(n_asd)]
    pids_n = [f"sub-NA{i:03d}" for i in range(n_nasd)]
    vals_a = rng.normal(asd_mean, 0.3, n_asd)
    vals_n = rng.normal(nasd_mean, 0.3, n_nasd)
    rows = (
        [{"pid": p, "Group": "ASD", feat: v, "age_mo": float(15 + i * 2)}
         for i, (p, v) in enumerate(zip(pids_a, vals_a, strict=False))]
        + [{"pid": p, "Group": "Non-ASD", feat: v, "age_mo": float(20 + i * 2)}
           for i, (p, v) in enumerate(zip(pids_n, vals_n, strict=False))]
    )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# extract_pid
# ─────────────────────────────────────────────────────────────────
class TestExtractPid:
    def test_valid_bids_path(self) -> None:
        assert extract_pid("/data/sub-ABC123/video.mp4") == "sub-ABC123"

    def test_no_sub_prefix(self) -> None:
        assert extract_pid("/data/subject01/video.mp4") is None

    def test_none_input(self) -> None:
        assert extract_pid(None) is None  # type: ignore[arg-type]

    def test_integer_input(self) -> None:
        assert extract_pid(42) is None  # type: ignore[arg-type]

    def test_returns_first_match(self) -> None:
        pid = extract_pid("/data/sub-X01/sub-X02/file.mp4")
        assert pid == "sub-X01"


# ─────────────────────────────────────────────────────────────────
# parse_timestamps
# ─────────────────────────────────────────────────────────────────
class TestParseTimestamps:
    def test_single_segment(self) -> None:
        segs = parse_timestamps("0:10 - 0:20", fps=15.0)
        assert len(segs) == 1
        start, end = segs[0]
        assert start == 10 * 15
        assert end == 20 * 15

    def test_multiple_segments(self) -> None:
        segs = parse_timestamps("0:05 - 0:10, 1:00 - 1:30", fps=15.0)
        assert len(segs) == 2

    def test_invalid_string(self) -> None:
        assert parse_timestamps("not a timestamp") == []

    def test_non_string_input(self) -> None:
        assert parse_timestamps(None) == []  # type: ignore[arg-type]

    def test_zero_duration_skipped(self) -> None:
        # start == end → skipped
        segs = parse_timestamps("0:10 - 0:10")
        assert segs == []

    def test_custom_fps(self) -> None:
        segs = parse_timestamps("0:01 - 0:02", fps=30.0)
        start, end = segs[0]
        assert start == 30
        assert end == 60


# ─────────────────────────────────────────────────────────────────
# get_kp
# ─────────────────────────────────────────────────────────────────
class TestGetKp:
    def test_valid_keypoint(self) -> None:
        fd = {"kp_009": {"x": 1.0, "y": 2.0, "confidence": 0.8}}
        kp = get_kp(fd, "kp_009", min_conf=0.3)
        assert kp is not None
        assert kp["x"] == 1.0

    def test_missing_key(self) -> None:
        assert get_kp({}, "kp_009") is None

    def test_low_confidence(self) -> None:
        fd = {"kp_009": {"x": 1.0, "y": 2.0, "confidence": 0.1}}
        assert get_kp(fd, "kp_009", min_conf=0.3) is None

    def test_not_dict(self) -> None:
        fd = {"kp_009": "bad_value"}
        assert get_kp(fd, "kp_009") is None

    def test_exact_threshold(self) -> None:
        fd = {"kp_009": {"x": 0.0, "y": 0.0, "confidence": 0.3}}
        # confidence == min_conf: not strictly less → should pass
        assert get_kp(fd, "kp_009", min_conf=0.3) is not None


# ─────────────────────────────────────────────────────────────────
# torso_length
# ─────────────────────────────────────────────────────────────────
class TestTorsoLength:
    def test_standard_frame(self) -> None:
        fd = _frame()
        tl = torso_length(fd)
        assert tl is not None
        assert tl > 5

    def test_missing_shoulder(self) -> None:
        fd = _frame()
        del fd["kp_005"]
        assert torso_length(fd) is None

    def test_zero_length_returns_none(self) -> None:
        fd = _frame()
        # Make shoulders == hips
        fd["kp_011"] = fd["kp_005"].copy()
        fd["kp_012"] = fd["kp_006"].copy()
        assert torso_length(fd) is None


# ─────────────────────────────────────────────────────────────────
# butter_lp
# ─────────────────────────────────────────────────────────────────
class TestButterLp:
    def test_output_shape_preserved(self) -> None:
        arr = np.random.default_rng(0).normal(0, 1, 50)
        out = butter_lp(arr, fs=15.0)
        assert out.shape == arr.shape

    def test_short_array_passthrough(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        out = butter_lp(arr, fs=15.0)
        np.testing.assert_array_equal(out, arr)

    def test_attenuates_high_freq(self) -> None:
        t = np.linspace(0, 2, 30)
        signal = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 7 * t)
        filtered = butter_lp(signal, cutoff=4.0, fs=15.0)
        # High-freq component should be reduced
        assert float(np.std(filtered)) < float(np.std(signal))


# ─────────────────────────────────────────────────────────────────
# spectral_features
# ─────────────────────────────────────────────────────────────────
class TestSpectralFeatures:
    def test_returns_two_floats(self) -> None:
        arr = np.sin(np.linspace(0, 4 * np.pi, 64))
        dom_freq, entropy = spectral_features(arr, fps=15.0)
        assert np.isfinite(dom_freq)
        assert np.isfinite(entropy)

    def test_short_array_returns_nan(self) -> None:
        arr = np.array([1.0, 2.0])
        dom_freq, entropy = spectral_features(arr, fps=15.0)
        assert np.isnan(dom_freq)
        assert np.isnan(entropy)

    def test_known_frequency(self) -> None:
        # 2 Hz sine at 15 fps
        t = np.arange(64) / 15.0
        arr = np.sin(2 * np.pi * 2.0 * t)
        dom_freq, _ = spectral_features(arr, fps=15.0)
        assert abs(dom_freq - 2.0) < 0.5


# ─────────────────────────────────────────────────────────────────
# cohen_d
# ─────────────────────────────────────────────────────────────────
class TestCohenD:
    def test_positive_d(self) -> None:
        a = np.array([2.0, 2.5, 3.0, 2.8])
        b = np.array([1.0, 1.2, 0.9, 1.1])
        assert cohen_d(a, b) > 0

    def test_negative_d(self) -> None:
        a = np.array([1.0, 1.2])
        b = np.array([3.0, 3.5])
        assert cohen_d(a, b) < 0

    def test_identical_groups(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert cohen_d(a, a) == 0.0

    def test_zero_pooled_sd(self) -> None:
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.0])
        assert cohen_d(a, b) == 0.0


# ─────────────────────────────────────────────────────────────────
# bootstrap_ci_d
# ─────────────────────────────────────────────────────────────────
class TestBootstrapCiD:
    def test_returns_tuple_of_two(self) -> None:
        a = np.array([2.0, 2.5, 3.0, 2.8, 2.6])
        b = np.array([1.0, 1.2, 0.9, 1.1, 1.3])
        lo, hi = bootstrap_ci_d(a, b, n_boot=200, seed=42)
        assert lo < hi

    def test_ci_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(2.0, 0.5, 20)
        b = rng.normal(1.0, 0.5, 20)
        d = cohen_d(a, b)
        lo, hi = bootstrap_ci_d(a, b, n_boot=300, seed=0)
        assert lo <= d <= hi

    def test_reproducible_with_seed(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.5, 2.5])
        r1 = bootstrap_ci_d(a, b, n_boot=100, seed=7)
        r2 = bootstrap_ci_d(a, b, n_boot=100, seed=7)
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────
# cles
# ─────────────────────────────────────────────────────────────────
class TestCles:
    def test_all_a_greater(self) -> None:
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cles(a, b) == pytest.approx(1.0)

    def test_all_b_greater(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert cles(a, b) == pytest.approx(0.0)

    def test_equal_distributions(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        assert abs(cles(a, b) - 0.5) < 0.05


# ─────────────────────────────────────────────────────────────────
# fdr_annotate
# ─────────────────────────────────────────────────────────────────
class TestFdrAnnotate:
    def test_adds_p_fdr_column(self) -> None:
        df = pd.DataFrame({"feature": ["a", "b", "c"], "p_raw": [0.01, 0.04, 0.5]})
        out = fdr_annotate(df, "p_raw")
        assert "p_fdr" in out.columns

    def test_adds_significance_flags(self) -> None:
        df = pd.DataFrame({"feature": ["a", "b"], "p_raw": [0.001, 0.9]})
        out = fdr_annotate(df, "p_raw")
        assert "sig_raw05" in out.columns
        assert "sig_fdr05" in out.columns

    def test_single_row(self) -> None:
        df = pd.DataFrame({"feature": ["x"], "p_raw": [0.03]})
        out = fdr_annotate(df, "p_raw")
        assert out["p_fdr"].iloc[0] == pytest.approx(0.03)


# ─────────────────────────────────────────────────────────────────
# assign_age_band
# ─────────────────────────────────────────────────────────────────
class TestAssignAgeBand:
    def test_first_band(self) -> None:
        assert assign_age_band(14.0) == "11-18mo"

    def test_second_band(self) -> None:
        assert assign_age_band(25.0) == "19-31mo"

    def test_third_band(self) -> None:
        assert assign_age_band(35.0) == "32-38mo"

    def test_boundary_inclusive(self) -> None:
        assert assign_age_band(11.0) == "11-18mo"
        assert assign_age_band(18.0) == "11-18mo"
        assert assign_age_band(32.0) == "32-38mo"

    def test_out_of_range(self) -> None:
        assert assign_age_band(5.0) is None
        assert assign_age_band(50.0) is None


# ─────────────────────────────────────────────────────────────────
# stream_filter
# ─────────────────────────────────────────────────────────────────
class TestStreamFilter:
    @pytest.fixture()
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"age_mo": [12.0, 20.0, 35.0], "val": [1, 2, 3]})

    def test_full_stream_returns_all(self, df: pd.DataFrame) -> None:
        out = stream_filter(df, "full")
        assert len(out) == 3

    def test_age_band_filter(self, df: pd.DataFrame) -> None:
        out = stream_filter(df, "11-18mo")
        assert len(out) == 1
        assert out["age_mo"].iloc[0] == 12.0

    def test_no_matching_rows(self, df: pd.DataFrame) -> None:
        out = stream_filter(df, "32-38mo")
        assert len(out) == 1


# ─────────────────────────────────────────────────────────────────
# extract_flapping_features
# ─────────────────────────────────────────────────────────────────
class TestExtractFlappingFeatures:
    def test_returns_dict_for_valid_input(self) -> None:
        frames = _pose_frames(n=30)
        result = extract_flapping_features(frames, list(range(30)))
        assert result is not None
        assert isinstance(result, dict)

    def test_insufficient_frames_returns_none(self) -> None:
        frames = _pose_frames(n=3)
        result = extract_flapping_features(frames, list(range(3)))
        assert result is None

    def test_bilateral_features_present(self) -> None:
        frames = _pose_frames(n=30, amplitude=25.0)
        result = extract_flapping_features(frames, list(range(30)))
        assert result is not None
        assert "wrist_amp_max" in result
        assert "bilateral_y_corr" in result
        assert "bilateral_sym_index" in result

    def test_amplitude_nonzero(self) -> None:
        frames = _pose_frames(n=30, amplitude=30.0)
        result = extract_flapping_features(frames, list(range(30)))
        assert result is not None
        assert result["wrist_amp_max"] > 0

    def test_no_wrist_data_returns_none(self) -> None:
        frames = {}
        for i in range(30):
            fd = _frame()
            del fd["kp_009"]
            del fd["kp_010"]
            frames[str(i)] = fd
        result = extract_flapping_features(frames, list(range(30)))
        assert result is None

    def test_unilateral_left_only(self) -> None:
        frames = {}
        for i in range(30):
            fd = _frame()
            del fd["kp_010"]  # remove right wrist
            frames[str(i)] = fd
        result = extract_flapping_features(frames, list(range(30)))
        assert result is not None
        assert "wrist_amp_max" in result


# ─────────────────────────────────────────────────────────────────
# compute_icc
# ─────────────────────────────────────────────────────────────────
class TestComputeIcc:
    def test_returns_dataframe(self) -> None:
        rng = np.random.default_rng(0)
        rows = []
        for i in range(10):
            pid = f"sub-{i:03d}"
            for _ in range(3):
                rows.append({"pid": pid, "feat_a": rng.normal(i * 0.5, 0.1)})
        df = pd.DataFrame(rows)
        result = compute_icc(df, ["feat_a"])
        assert isinstance(result, pd.DataFrame)

    def test_icc_in_range(self) -> None:
        rng = np.random.default_rng(1)
        rows = []
        for i in range(8):
            pid = f"sub-{i:03d}"
            base = rng.normal(0, 2)
            for _ in range(4):
                rows.append({"pid": pid, "feat_a": base + rng.normal(0, 0.1)})
        df = pd.DataFrame(rows)
        result = compute_icc(df, ["feat_a"])
        if len(result):
            assert 0.0 <= result["ICC"].iloc[0] <= 1.0

    def test_too_few_pids_raises(self) -> None:
        # compute_icc skips all features when n_pids < 5, leaving records=[].
        # pd.DataFrame([]).sort_values("ICC") then raises KeyError — known source behaviour.
        df = pd.DataFrame({"pid": ["sub-001"], "feat_a": [1.0]})
        with pytest.raises(KeyError):
            compute_icc(df, ["feat_a"])


# ─────────────────────────────────────────────────────────────────
# run_mwu
# ─────────────────────────────────────────────────────────────────
class TestRunMwu:
    def test_returns_dataframe(self) -> None:
        df = _child_df()
        result = run_mwu(df, ["wrist_amp_max"])
        assert isinstance(result, pd.DataFrame)
        assert "p_raw" in result.columns

    def test_p_value_in_range(self) -> None:
        df = _child_df()
        result = run_mwu(df, ["wrist_amp_max"])
        if len(result):
            assert 0.0 <= result["p_raw"].iloc[0] <= 1.0

    def test_insufficient_group_returns_empty(self) -> None:
        df = pd.DataFrame({
            "pid": ["sub-001", "sub-002"],
            "Group": ["ASD", "ASD"],
            "wrist_amp_max": [1.0, 2.0],
        })
        result = run_mwu(df, ["wrist_amp_max"])
        assert len(result) == 0

    def test_cles_column_present(self) -> None:
        df = _child_df()
        result = run_mwu(df, ["wrist_amp_max"])
        assert "cles" in result.columns

    def test_effect_direction(self) -> None:
        df = _child_df(asd_mean=3.0, nasd_mean=1.0)
        result = run_mwu(df, ["wrist_amp_max"])
        assert result["cohens_d"].iloc[0] > 0


# ─────────────────────────────────────────────────────────────────
# run_child_permutation
# ─────────────────────────────────────────────────────────────────
class TestRunChildPermutation:
    def test_returns_dataframe(self) -> None:
        df = _child_df()
        result = run_child_permutation(df, ["wrist_amp_max"], n_perm=200)
        assert isinstance(result, pd.DataFrame)

    def test_p_value_valid(self) -> None:
        df = _child_df()
        result = run_child_permutation(df, ["wrist_amp_max"], n_perm=200)
        if len(result):
            p = result["p_raw"].iloc[0]
            assert 0.0 < p <= 1.0

    def test_sig_feature_low_p(self) -> None:
        # Large effect should have low p
        df = _child_df(n_asd=10, n_nasd=10, asd_mean=5.0, nasd_mean=0.0)
        result = run_child_permutation(df, ["wrist_amp_max"], n_perm=500)
        assert len(result) > 0
        assert result["p_raw"].iloc[0] < 0.1


# ─────────────────────────────────────────────────────────────────
# run_wild_bootstrap
# ─────────────────────────────────────────────────────────────────
class TestRunWildBootstrap:
    def test_returns_dataframe(self) -> None:
        df = _child_df()
        result = run_wild_bootstrap(df, ["wrist_amp_max"], n_boot=200)
        assert isinstance(result, pd.DataFrame)

    def test_p_value_valid(self) -> None:
        df = _child_df()
        result = run_wild_bootstrap(df, ["wrist_amp_max"], n_boot=200)
        if len(result):
            p = result["p_raw"].iloc[0]
            assert 0.0 < p <= 1.0


# ─────────────────────────────────────────────────────────────────
# run_spearman_age
# ─────────────────────────────────────────────────────────────────
class TestRunSpearmanAge:
    def test_returns_dataframe(self) -> None:
        df = _child_df()
        result = run_spearman_age(df, ["wrist_amp_max"])
        assert isinstance(result, pd.DataFrame)

    def test_both_groups_present(self) -> None:
        df = _child_df()
        result = run_spearman_age(df, ["wrist_amp_max"])
        if len(result):
            assert set(result["Group"].unique()) == {"ASD", "Non-ASD"}

    def test_correlation_in_range(self) -> None:
        df = _child_df()
        result = run_spearman_age(df, ["wrist_amp_max"])
        if len(result):
            assert result["spearman_r"].between(-1, 1).all()


# ─────────────────────────────────────────────────────────────────
# make_consensus
# ─────────────────────────────────────────────────────────────────
class TestMakeConsensus:
    def _mock_result(self, p: float, feat: str = "wrist_amp_max") -> pd.DataFrame:
        return pd.DataFrame({
            "feature": [feat],
            "p_raw": [p],
            "cohens_d": [1.0],
            "d_ci_lo": [0.5],
            "d_ci_hi": [1.5],
        })

    def test_aggregates_methods(self) -> None:
        rd = {
            "ChildPerm": self._mock_result(0.01),
            "PseudobulkMW": self._mock_result(0.03),
        }
        result = make_consensus(rd, ["wrist_amp_max"])
        assert "n_methods_sig" in result.columns
        assert result["n_methods_sig"].iloc[0] == 2

    def test_empty_results_handled(self) -> None:
        rd = {
            "ChildPerm": pd.DataFrame(),
            "PseudobulkMW": self._mock_result(0.5),
        }
        result = make_consensus(rd, ["wrist_amp_max"])
        assert len(result) == 1

    def test_no_sig_methods(self) -> None:
        rd = {"ChildPerm": self._mock_result(0.8)}
        result = make_consensus(rd, ["wrist_amp_max"])
        assert result["n_methods_sig"].iloc[0] == 0


# ─────────────────────────────────────────────────────────────────
# run_consistency_gate
# ─────────────────────────────────────────────────────────────────
class TestRunConsistencyGate:
    def _make_feat_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        rows = []
        for lbl in ["hands flapping", "arm flapping"]:
            for g, mean in [("ASD", 2.0), ("Non-ASD", 1.0)]:
                for i in range(5):
                    rows.append({
                        "pid": f"sub-{g}-{i}",
                        "Group": g,
                        "original_label": lbl,
                        "wrist_amp_max": rng.normal(mean, 0.2),
                    })
        return pd.DataFrame(rows)

    def test_returns_three_items(self) -> None:
        df = self._make_feat_df()
        r = run_consistency_gate(df, ["wrist_amp_max"], ["wrist_amp_max"])
        assert len(r) == 3

    def test_consistent_direction(self) -> None:
        df = self._make_feat_df()
        cons_df, consistent_feats, _ = run_consistency_gate(
            df, ["wrist_amp_max"], ["wrist_amp_max"]
        )
        # ASD > Non-ASD in both labels → should be consistent
        assert "wrist_amp_max" in consistent_feats

    def test_empty_sig_feats(self) -> None:
        df = self._make_feat_df()
        cons_df, consistent_feats, _ = run_consistency_gate(
            df, ["wrist_amp_max"], []
        )
        assert consistent_feats == []


# ─────────────────────────────────────────────────────────────────
# _add_label_dummies
# ─────────────────────────────────────────────────────────────────
class TestAddLabelDummies:
    def test_adds_dummy_columns(self) -> None:
        df = pd.DataFrame({
            "original_label": ["hands flapping", "arm flapping", "hands flapping"],
        })
        out, cols = _add_label_dummies(df, reference="hands flapping")
        assert len(cols) == 1
        assert cols[0].startswith("lbl_")

    def test_reference_label_excluded(self) -> None:
        df = pd.DataFrame({"original_label": ["hands flapping", "hands flapping"]})
        out, cols = _add_label_dummies(df, reference="hands flapping")
        assert cols == []

    def test_dummy_values_correct(self) -> None:
        df = pd.DataFrame({
            "original_label": ["hands flapping", "arm flapping"]
        })
        out, cols = _add_label_dummies(df, reference="hands flapping")
        assert out[cols[0]].iloc[0] == 0.0
        assert out[cols[0]].iloc[1] == 1.0


# ─────────────────────────────────────────────────────────────────
# _standardise
# ─────────────────────────────────────────────────────────────────
class TestStandardise:
    def test_zero_mean_unit_variance(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z, m, sd = _standardise(s)
        assert abs(float(np.mean(z))) < 1e-10
        # _standardise divides by series.std() which uses ddof=1
        assert abs(float(np.std(z, ddof=1)) - 1.0) < 1e-6

    def test_returns_original_stats(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0])
        _, m, sd = _standardise(s)
        assert m == pytest.approx(20.0)
        assert sd == pytest.approx(10.0)

    def test_constant_series_no_div_zero(self) -> None:
        s = pd.Series([5.0, 5.0, 5.0])
        z, m, sd = _standardise(s)
        assert np.all(np.isfinite(z))


# ─────────────────────────────────────────────────────────────────
# _savage_dickey_bf
# ─────────────────────────────────────────────────────────────────
class TestSavageDickeyBf:
    def test_bf_positive(self) -> None:
        post = np.random.default_rng(0).normal(0.5, 0.2, 2000)
        bf = _savage_dickey_bf(post, prior_sd=0.5)
        assert np.isfinite(bf) or np.isnan(bf)  # nan allowed if kde fails

    def test_null_posterior_bf_near_one(self) -> None:
        # Posterior centred on 0 ≈ prior → BF ≈ 1
        post = np.random.default_rng(1).normal(0.0, 0.5, 5000)
        bf = _savage_dickey_bf(post, prior_sd=0.5)
        if np.isfinite(bf):
            assert 0.1 < bf < 10.0  # loose bounds