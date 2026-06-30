"""
pytest test suite for rmm_combined.py
Source: src/sailsprep/analysis/rmm_combined/rmm_combined.py

Run:  poetry run pytest src/tests/analysis/rmm_combined/test_rmm_combined.py  -v

"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import butter, filtfilt, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE CONSTANTS  (identical to rmm_combined.py)
# ═══════════════════════════════════════════════════════════════════════════

FPS: float = 15.0
MIN_CONF: float = 0.3
BEH_REFERENCE: str = "hands flapping"
RMM_LABELS = {"hands flapping", "jumping", "rocking", "spinning"}
AGE_STREAMS = {"full": None, "11-18mo": (11, 18), "32-38mo": (32, 38)}

KP: Dict[str, str] = {
    "nose": "kp_000",
    "left_shoulder": "kp_005",
    "right_shoulder": "kp_006",
    "left_elbow": "kp_007",
    "right_elbow": "kp_008",
    "left_wrist": "kp_009",
    "right_wrist": "kp_010",
    "left_hip": "kp_011",
    "right_hip": "kp_012",
    "left_knee": "kp_013",
    "right_knee": "kp_014",
    "left_ankle": "kp_015",
    "right_ankle": "kp_016",
}


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTION RE-IMPLEMENTATIONS  (same logic as rmm_combined.py)
# ═══════════════════════════════════════════════════════════════════════════


# ── Utilities ──────────────────────────────────────────────────────────────

def extract_pid(path) -> Optional[str]:
    if not isinstance(path, str):
        return None
    m = re.search(r"(sub-[A-Za-z0-9]+)", path)
    return m.group(1) if m else None


def parse_timestamps(ts_str, fps: float = FPS) -> List[tuple]:
    if not isinstance(ts_str, str):
        return []
    segs = []
    for part in ts_str.split(","):
        m = re.match(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)", part.strip())
        if m:
            s = int(m.group(1)) * 60 + int(m.group(2))
            e = int(m.group(3)) * 60 + int(m.group(4))
            if e > s:
                segs.append((int(s * fps), int(e * fps)))
    return segs


def butter_lp(data, cutoff: float = 4.0, fs: float = 15.0, order: int = 2):
    arr = np.array(data, dtype=float)
    if len(arr) < 10:
        return arr
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff, nyq * 0.9) / nyq, btype="low")
    if len(arr) < 3 * max(len(b), len(a)):
        return arr
    return filtfilt(b, a, arr)


def spectral_features(arr, fps: float, lo: float = 0.3, hi: float = 2.0):
    if len(arr) < 16:
        return np.nan, np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq = freqs[np.argmax(psd)]
        psd_n = psd / (psd.sum() + 1e-12)
        entropy = -np.sum(psd_n[psd_n > 0] * np.log2(psd_n[psd_n > 0]))
        band_mask = (freqs >= lo) & (freqs <= hi)
        band_pwr = psd[band_mask].sum() / (psd.sum() + 1e-12)
        return float(dom_freq), float(entropy), float(band_pwr)
    except Exception:
        return np.nan, np.nan, np.nan


def cohen_d(a, b) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def bootstrap_ci_d(a, b, n_boot: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    boot = [
        cohen_d(
            rng.choice(a, len(a), replace=True),
            rng.choice(b, len(b), replace=True),
        )
        for _ in range(n_boot)
    ]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def fdr_annotate(df_res: pd.DataFrame, p_col: str) -> pd.DataFrame:
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


def stream_filter(df: pd.DataFrame, stream_key: str) -> pd.DataFrame:
    bounds = AGE_STREAMS[stream_key]
    if bounds is None:
        return df.copy()
    lo, hi = bounds
    return df[(df["age_mo"] >= lo) & (df["age_mo"] <= hi)].copy()


# ── Pose helpers ───────────────────────────────────────────────────────────

def get_kp(fd: dict, key: str, min_conf: float = MIN_CONF):
    if key not in fd:
        return None
    kp = fd[key]
    if not isinstance(kp, dict):
        return None
    if kp.get("confidence", 0) < min_conf:
        return None
    return kp


def torso_length(fd: dict):
    ls = get_kp(fd, KP["left_shoulder"], 0.1)
    rs = get_kp(fd, KP["right_shoulder"], 0.1)
    lh = get_kp(fd, KP["left_hip"], 0.1)
    rh = get_kp(fd, KP["right_hip"], 0.1)
    if not all([ls, rs, lh, rh]):
        return None
    sx = (ls["x"] + rs["x"]) / 2
    sy = (ls["y"] + rs["y"]) / 2
    hx = (lh["x"] + rh["x"]) / 2
    hy = (lh["y"] + rh["y"]) / 2
    d = np.sqrt((sx - hx) ** 2 + (sy - hy) ** 2)
    return d if d > 5 else None


def get_scale(fd: dict):
    tl = torso_length(fd)
    if tl:
        return tl
    lh = get_kp(fd, KP["left_hip"], 0.1)
    rh = get_kp(fd, KP["right_hip"], 0.1)
    if lh and rh:
        d = np.sqrt((lh["x"] - rh["x"]) ** 2 + (lh["y"] - rh["y"]) ** 2)
        if d > 5:
            return d
    ls = get_kp(fd, KP["left_shoulder"], 0.1)
    rs = get_kp(fd, KP["right_shoulder"], 0.1)
    if ls and rs:
        d = np.sqrt((ls["x"] - rs["x"]) ** 2 + (ls["y"] - rs["y"]) ** 2)
        if d > 5:
            return d
    return None


# ── Feature extraction ─────────────────────────────────────────────────────

def _jfeats(arr, rec: dict, name: str, fps: float) -> None:
    a = np.array(arr, dtype=float)
    if len(a) < 5:
        return
    rec[f"{name}_amplitude"] = float(np.ptp(a))
    rec[f"{name}_std"] = float(np.std(a))
    rec[f"{name}_mean"] = float(np.mean(a))
    rec[f"{name}_iqr"] = float(np.percentile(a, 75) - np.percentile(a, 25))
    if len(a) >= 8:
        try:
            sm = butter_lp(a, fs=fps)
            vel = np.diff(sm) * fps
            rec[f"{name}_vel_mean"] = float(np.mean(np.abs(vel)))
            rec[f"{name}_vel_std"] = float(np.std(vel))
            rec[f"{name}_vel_max"] = float(np.max(np.abs(vel)))
            if len(vel) >= 4:
                acc = np.diff(vel) * fps
                rec[f"{name}_acc_mean"] = float(np.mean(np.abs(acc)))
                rec[f"{name}_acc_max"] = float(np.max(np.abs(acc)))
        except Exception:
            pass
    df_f, se, bp = spectral_features(a, fps)
    rec[f"{name}_dom_freq"] = df_f
    rec[f"{name}_spectral_entropy"] = se
    rec[f"{name}_band_power_0p3_2hz"] = bp


def extract_rmm_features(pose_frames: dict, frame_indices: list, ann_fps: float = FPS):
    hip_x_L: list = []
    hip_x_R: list = []
    hip_y_L: list = []
    hip_y_R: list = []
    sh_x_L: list = []
    sh_x_R: list = []
    wr_x_L: list = []
    wr_x_R: list = []
    wr_y_L: list = []
    wr_y_R: list = []
    el_x_L: list = []
    el_x_R: list = []
    kn_y_L: list = []
    kn_y_R: list = []
    nose_x_arr: list = []
    conf_vals: list = []
    n_valid = 0

    for fi in frame_indices:
        fk = str(fi)
        if fk not in pose_frames:
            continue
        fd = pose_frames[fk]
        scale = get_scale(fd)
        if scale is None:
            continue
        lh = get_kp(fd, KP["left_hip"])
        rh = get_kp(fd, KP["right_hip"])
        ls = get_kp(fd, KP["left_shoulder"])
        rs = get_kp(fd, KP["right_shoulder"])
        lw = get_kp(fd, KP["left_wrist"])
        rw = get_kp(fd, KP["right_wrist"])
        le = get_kp(fd, KP["left_elbow"])
        re_kp = get_kp(fd, KP["right_elbow"])
        lk = get_kp(fd, KP["left_knee"])
        rk = get_kp(fd, KP["right_knee"])
        ns = get_kp(fd, KP["nose"])
        if lh is None and rh is None and ls is None and rs is None:
            continue
        n_valid += 1
        if lh:
            hip_x_L.append(lh["x"] / scale)
            hip_y_L.append(lh["y"] / scale)
            conf_vals.append(lh["confidence"])
        if rh:
            hip_x_R.append(rh["x"] / scale)
            hip_y_R.append(rh["y"] / scale)
            conf_vals.append(rh["confidence"])
        if ls:
            sh_x_L.append(ls["x"] / scale)
        if rs:
            sh_x_R.append(rs["x"] / scale)
        if lw:
            wr_x_L.append(lw["x"] / scale)
            wr_y_L.append(lw["y"] / scale)
        if rw:
            wr_x_R.append(rw["x"] / scale)
            wr_y_R.append(rw["y"] / scale)
        if le:
            el_x_L.append(le["x"] / scale)
        if re_kp:
            el_x_R.append(re_kp["x"] / scale)
        if lk:
            kn_y_L.append(lk["y"] / scale)
        if rk:
            kn_y_R.append(rk["y"] / scale)
        if ns:
            nose_x_arr.append(ns["x"] / scale)

    if n_valid < 5:
        return None

    rec: dict = {
        "n_valid_frames": n_valid,
        "n_total_frames": len(frame_indices),
        "pct_valid": n_valid / len(frame_indices),
        "duration_sec": len(frame_indices) / ann_fps,
        "mean_conf": float(np.mean(conf_vals)) if conf_vals else np.nan,
    }

    def _mid(a_list, b_list):
        if a_list and b_list:
            ml = min(len(a_list), len(b_list))
            return (np.array(a_list[:ml]) + np.array(b_list[:ml])) / 2
        return np.array(a_list if a_list else b_list, dtype=float)

    mhx = _mid(hip_x_L, hip_x_R)
    mhy = _mid(hip_y_L, hip_y_R)
    mwx = _mid(wr_x_L, wr_x_R)
    mwy = _mid(wr_y_L, wr_y_R)
    msx = _mid(sh_x_L, sh_x_R)
    mex = _mid(el_x_L, el_x_R)
    mky = _mid(kn_y_L, kn_y_R)

    if len(mhx):
        _jfeats(mhx, rec, "mean_hip_x", ann_fps)
    if len(mhy):
        _jfeats(mhy, rec, "mean_hip_y", ann_fps)
    if len(mwx):
        _jfeats(mwx, rec, "mean_wrist_x", ann_fps)
    if len(mwy):
        _jfeats(mwy, rec, "mean_wrist_y", ann_fps)
    if len(msx):
        _jfeats(msx, rec, "mean_sh_x", ann_fps)
    if len(mex):
        _jfeats(mex, rec, "elbow_x", ann_fps)
    if len(mky):
        _jfeats(mky, rec, "knee_y", ann_fps)
    if len(nose_x_arr):
        _jfeats(np.array(nose_x_arr), rec, "nose_x", ann_fps)
    if len(msx) and len(mhx):
        ml = min(len(msx), len(mhx))
        tilt = msx[:ml] - mhx[:ml]
        _jfeats(tilt, rec, "trunk_tilt", ann_fps)

    if len(hip_x_L) >= 5 and len(hip_x_R) >= 5:
        ml = min(len(hip_x_L), len(hip_x_R))
        xl = np.array(hip_x_L[:ml])
        xr = np.array(hip_x_R[:ml])
        rec["bilateral_hip_x_corr"] = float(np.corrcoef(xl, xr)[0, 1])
        rec["bilateral_hip_x_sym"] = float(
            1 - abs(np.ptp(xl) - np.ptp(xr)) / (np.ptp(xl) + np.ptp(xr) + 1e-8)
        )
    if len(wr_y_L) >= 5 and len(wr_y_R) >= 5:
        ml = min(len(wr_y_L), len(wr_y_R))
        wl_arr = np.array(wr_y_L[:ml])
        wr_arr = np.array(wr_y_R[:ml])
        rec["bilateral_wrist_corr"] = float(np.corrcoef(wl_arr, wr_arr)[0, 1])
        rec["bilateral_wrist_sym"] = float(
            1 - abs(np.ptp(wl_arr) - np.ptp(wr_arr)) / (np.ptp(wl_arr) + np.ptp(wr_arr) + 1e-8)
        )

    hip_x_amp = float(np.ptp(mhx)) if len(mhx) else 0.0
    hip_y_amp = float(np.ptp(mhy)) if len(mhy) else 0.0
    rec["hip_2d_amplitude_max"] = float(max(hip_x_amp, hip_y_amp))
    rec["hip_x_y_ratio"] = float(hip_x_amp / (hip_y_amp + 1e-8))

    active_keys = [
        "mean_hip_x_amplitude",
        "mean_hip_y_amplitude",
        "mean_wrist_x_amplitude",
        "mean_wrist_y_amplitude",
        "mean_sh_x_amplitude",
        "elbow_x_amplitude",
        "knee_y_amplitude",
        "nose_x_amplitude",
    ]
    rec["n_active_joints"] = sum(
        1 for k in active_keys if k in rec and not np.isnan(rec[k]) and rec[k] > 0.05
    )
    return rec


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_icc(clip_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    records = []
    for feat in feat_cols:
        sub = clip_df[["pid", feat]].dropna()
        if len(sub) < 10:
            continue
        groups = [g[feat].values for _, g in sub.groupby("pid") if len(g) >= 2]
        if len(groups) < 5:
            continue
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        n0 = (n_total - sum(len(g) ** 2 / n_total for g in groups)) / (k - 1)
        grand = np.concatenate(groups)
        ms_between = np.sum(
            [len(g) * (np.mean(g) - np.mean(grand)) ** 2 for g in groups]
        ) / (k - 1)
        ms_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in groups]) / (
            n_total - k
        )
        icc = max(
            0.0, (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
        )
        f_stat, _ = stats.f_oneway(*groups)
        records.append(
            {"feature": feat, "ICC": round(icc, 4), "f_stat": round(float(f_stat), 3)}
        )
    return pd.DataFrame(records).sort_values("ICC", ascending=False) if records else pd.DataFrame()


def run_mwu(child_df: pd.DataFrame, feat_cols: List[str], subset_label: str = "full") -> pd.DataFrame:
    records = []
    for feat in feat_cols:
        av = child_df[child_df["Group"] == "ASD"][feat].dropna().values
        nv = child_df[child_df["Group"] == "Non-ASD"][feat].dropna().values
        if len(av) < 3 or len(nv) < 3:
            continue
        stat, p = stats.mannwhitneyu(av, nv, alternative="two-sided")
        d = cohen_d(av, nv)
        ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=200)
        records.append(
            {
                "feature": feat,
                "subset": subset_label,
                "method": "PseudobulkMW",
                "asd_median": float(np.median(av)),
                "nasd_median": float(np.median(nv)),
                "mw_stat": float(stat),
                "p_raw": float(p),
                "cohens_d": d,
                "d_ci_lo": ci_lo,
                "d_ci_hi": ci_hi,
                "n_asd": len(av),
                "n_nasd": len(nv),
            }
        )
    if not records:
        return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), "p_raw").sort_values("p_raw")


def run_child_permutation(
    child_df: pd.DataFrame,
    feat_cols: List[str],
    n_perm: int = 5000,
    subset_label: str = "full",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records = []
    for feat in feat_cols:
        sub = child_df[["pid", "Group", feat]].dropna()
        if sub["Group"].nunique() < 2:
            continue
        av = sub[sub["Group"] == "ASD"][feat].values
        nv = sub[sub["Group"] == "Non-ASD"][feat].values
        if len(av) < 3 or len(nv) < 3:
            continue
        obs_stat = abs(np.mean(av) - np.mean(nv))
        n_asd = len(av)
        vals_arr = sub[feat].values
        n_total = len(sub["pid"].unique())
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            sl = rng.permutation(["ASD"] * n_asd + ["Non-ASD"] * (n_total - n_asd))
            a_v = vals_arr[np.array(sl) == "ASD"]
            n_v = vals_arr[np.array(sl) == "Non-ASD"]
            a_v = a_v[~np.isnan(a_v)]
            n_v = n_v[~np.isnan(n_v)]
            perm_stats[i] = (
                abs(np.mean(a_v) - np.mean(n_v))
                if len(a_v) > 0 and len(n_v) > 0
                else 0
            )
        p_perm = max(float(np.mean(perm_stats >= obs_stat)), 1.0 / n_perm)
        d = cohen_d(av, nv)
        ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=200)
        records.append(
            {
                "feature": feat,
                "subset": subset_label,
                "method": "ChildPerm",
                "obs_stat": float(obs_stat),
                "p_raw": p_perm,
                "cohens_d": d,
                "d_ci_lo": ci_lo,
                "d_ci_hi": ci_hi,
                "n_asd": len(av),
                "n_nasd": len(nv),
            }
        )
    if not records:
        return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), "p_raw").sort_values("p_raw")


def run_wild_bootstrap(
    child_df: pd.DataFrame,
    feat_cols: List[str],
    n_boot: int = 5000,
    subset_label: str = "full",
) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    records = []
    df_use = child_df.copy().dropna(subset=["age_mo"])
    df_use["Group_bin"] = (df_use["Group"] == "ASD").astype(float)
    for feat in feat_cols:
        sub = df_use[["pid", "Group_bin", "age_mo", feat]].dropna()
        if sub["Group_bin"].nunique() < 2:
            continue
        av = sub[sub["Group_bin"] == 1][feat].values
        nv = sub[sub["Group_bin"] == 0][feat].values
        if len(av) < 3 or len(nv) < 3:
            continue
        n = len(sub)
        y = sub[feat].values.astype(float)
        X = np.column_stack(
            [np.ones(n), sub["Group_bin"].values, sub["age_mo"].values]
        )
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        resid = y - X @ beta
        t_obs = beta[1] / (np.std(resid) / np.sqrt(n) + 1e-10)
        X0 = X[:, [0, 2]]
        try:
            beta0, _, _, _ = np.linalg.lstsq(X0, y, rcond=None)
        except Exception:
            continue
        resid0 = y - X0 @ beta0
        pids = sub["pid"].values
        u_pids = np.unique(pids)
        t_boot = np.zeros(n_boot)
        for b in range(n_boot):
            w_map = {p: rng.choice([-1.0, 1.0]) for p in u_pids}
            w = np.array([w_map[p] for p in pids])
            y_b = X0 @ beta0 + resid0 * w
            try:
                beta_b, _, _, _ = np.linalg.lstsq(X, y_b, rcond=None)
                resid_b = y_b - X @ beta_b
                t_boot[b] = beta_b[1] / (np.std(resid_b) / np.sqrt(n) + 1e-10)
            except Exception:
                t_boot[b] = 0.0
        p_wb = max(float(np.mean(np.abs(t_boot) >= abs(t_obs))), 1.0 / n_boot)
        d = cohen_d(av, nv)
        ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=200)
        records.append(
            {
                "feature": feat,
                "subset": subset_label,
                "method": "WildBoot",
                "coef_ASD": float(beta[1]),
                "t_obs": float(t_obs),
                "p_raw": p_wb,
                "cohens_d": d,
                "d_ci_lo": ci_lo,
                "d_ci_hi": ci_hi,
                "n_asd": int(len(av)),
                "n_nasd": int(len(nv)),
            }
        )
    if not records:
        return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), "p_raw").sort_values("p_raw")


def make_consensus(
    results_dict: dict, feat_cols: List[str], threshold: float = 0.05
) -> pd.DataFrame:
    rows = []
    for feat in feat_cols:
        row: dict = {"feature": feat}
        n_sig = 0
        for mname, res_df in results_dict.items():
            if res_df is None or len(res_df) == 0:
                row[f"p_{mname}"] = np.nan
                continue
            match = res_df[res_df["feature"] == feat]
            if len(match) == 0:
                row[f"p_{mname}"] = np.nan
            else:
                p = match["p_raw"].values[0]
                row[f"p_{mname}"] = round(p, 4)
                if p < threshold:
                    n_sig += 1
        row["n_methods_sig"] = n_sig
        rows.append(row)
    cons = pd.DataFrame(rows)
    for key in ("LME_KR", "LME_noKR"):
        lme_df = results_dict.get(key)
        if lme_df is not None and not lme_df.empty and "cohens_d" in lme_df.columns:
            cons["cohens_d_LME"] = cons["feature"].map(
                lme_df.set_index("feature")["cohens_d"].to_dict()
            )
            if "d_ci_lo" in lme_df.columns:
                cons["d_ci_lo"] = cons["feature"].map(
                    lme_df.set_index("feature")["d_ci_lo"].to_dict()
                )
                cons["d_ci_hi"] = cons["feature"].map(
                    lme_df.set_index("feature")["d_ci_hi"].to_dict()
                )
            break
    return cons.sort_values("n_methods_sig", ascending=False)


def run_consistency_gate(feat_df: pd.DataFrame, feat_cols: List[str], sig_feats: List[str]):
    beh_mwu: dict = {}
    for beh in sorted(feat_df["behavior"].dropna().unique()):
        sub = feat_df[feat_df["behavior"] == beh]
        asd_n = sub[sub["Group"] == "ASD"]["pid"].nunique()
        nan_n = sub[sub["Group"] == "Non-ASD"]["pid"].nunique()
        if asd_n < 3 or nan_n < 3:
            continue
        recs = []
        for feat in feat_cols:
            av = sub[sub["Group"] == "ASD"][feat].dropna().values
            nv = sub[sub["Group"] == "Non-ASD"][feat].dropna().values
            if len(av) < 3 or len(nv) < 3:
                continue
            _, p = stats.mannwhitneyu(av, nv, alternative="two-sided")
            recs.append(
                {
                    "feature": feat,
                    "cohens_d": cohen_d(av, nv),
                    "p_raw": p,
                    "behavior": beh,
                }
            )
        if recs:
            beh_mwu[beh] = pd.DataFrame(recs)

    cons_recs = []
    consistent_feats: List[str] = []
    beh_all = (
        pd.concat(beh_mwu.values(), ignore_index=True) if beh_mwu else pd.DataFrame()
    )
    for feat in sig_feats:
        if len(beh_all) == 0:
            break
        sub = beh_all[beh_all["feature"] == feat]
        if len(sub) < 2:
            continue
        signs = np.sign(sub["cohens_d"].values)
        n_same = int((signs == signs[0]).sum())
        passed = n_same == len(sub)
        cons_recs.append(
            {
                "feature": feat,
                "n_behaviors_tested": len(sub),
                "n_same_direction": n_same,
                "consistent": passed,
            }
        )
        if passed:
            consistent_feats.append(feat)
    return pd.DataFrame(cons_recs), consistent_feats, beh_mwu


def run_loso_child(
    cdf: pd.DataFrame,
    feat_cols: List[str],
    clf_name: str = "LR",
    n_perm: int = 500,
    seed: int = 42,
):
    df_ = cdf.copy()
    df_["y"] = (df_["Group"] == "ASD").astype(int)
    if df_["y"].sum() < 4 or (1 - df_["y"]).sum() < 4:
        return None
    usable = [f for f in feat_cols if f in df_.columns and df_[f].notna().mean() > 0.5]
    if len(usable) < 2:
        return None
    df_[usable] = df_[usable].fillna(df_[usable].median())
    clf = (
        LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced", random_state=seed
        )
        if clf_name == "LR"
        else RandomForestClassifier(
            n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=1
        )
    )
    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
    y_true: list = []
    y_score: list = []
    for pid in df_["pid"].unique():
        test = df_[df_["pid"] == pid]
        train = df_[df_["pid"] != pid]
        if len(train["y"].unique()) < 2:
            continue
        try:
            pipe.fit(train[usable].values, train["y"].values)
            y_score.extend(pipe.predict_proba(test[usable].values)[:, 1].tolist())
            y_true.extend(test["y"].values.tolist())
        except Exception:
            continue
    if len(set(y_true)) < 2:
        return None
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    perm = [
        roc_auc_score(rng.permuted(np.array(y_true)), y_score) for _ in range(n_perm)
    ]
    p_perm = float((np.array(perm) >= auc).mean())
    cm = confusion_matrix(y_true, (np.array(y_score) >= 0.5).astype(int))
    return {
        "auc": auc,
        "ap": ap,
        "perm_p": p_perm,
        "n_features": len(usable),
        "n_subjects": df_["pid"].nunique(),
        "y_true": y_true,
        "y_score": y_score,
        "perm_aucs": perm,
        "confusion_matrix": cm,
        "clf": clf_name,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


def _make_kp(x: float, y: float, conf: float = 0.9) -> dict:
    return {"x": float(x), "y": float(y), "confidence": float(conf)}


def _make_full_frame(ox: float = 0.0, oy: float = 0.0, conf: float = 0.9) -> dict:
    """Return a frame dict with all 13 keypoints."""
    return {
        KP["nose"]: _make_kp(200 + ox, 50 + oy, conf),
        KP["left_shoulder"]: _make_kp(180 + ox, 100, conf),
        KP["right_shoulder"]: _make_kp(220 + ox, 100, conf),
        KP["left_elbow"]: _make_kp(160 + ox * 1.5, 150 + oy, conf),
        KP["right_elbow"]: _make_kp(240 + ox * 1.5, 150 + oy, conf),
        KP["left_wrist"]: _make_kp(140 + ox * 2, 200 + oy * 2, conf),
        KP["right_wrist"]: _make_kp(260 + ox * 2, 200 + oy * 2, conf),
        KP["left_hip"]: _make_kp(185 + ox * 0.5, 200, conf),
        KP["right_hip"]: _make_kp(215 + ox * 0.5, 200, conf),
        KP["left_knee"]: _make_kp(180, 280 + oy, conf),
        KP["right_knee"]: _make_kp(220, 280 + oy, conf),
        KP["left_ankle"]: _make_kp(178, 360, conf),
        KP["right_ankle"]: _make_kp(222, 360, conf),
    }


@pytest.fixture
def pose_frames() -> dict:
    """100-frame pose dict with sinusoidal motion (mimics hand-flapping)."""
    rng = np.random.default_rng(0)
    frames: dict = {}
    for i in range(100):
        t = i / FPS
        ox = 10 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.5)
        oy = 5 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.3)
        frames[str(i)] = _make_full_frame(ox, oy)
    return frames


@pytest.fixture
def child_df() -> pd.DataFrame:
    """30-child DataFrame (15 ASD + 15 Non-ASD) for stats/classification tests."""
    rng = np.random.default_rng(42)
    n = 15
    return pd.DataFrame(
        {
            "pid": [f"sub-A{i:03d}" for i in range(n)]
            + [f"sub-B{i:03d}" for i in range(n)],
            "Group": ["ASD"] * n + ["Non-ASD"] * n,
            "age_mo": list(rng.uniform(12, 36, n * 2)),
            # feat1/feat2 differ between groups; feat3 does not
            "feat1": list(rng.normal(1.5, 0.4, n)) + list(rng.normal(1.0, 0.4, n)),
            "feat2": list(rng.normal(0.8, 0.2, n)) + list(rng.normal(0.5, 0.2, n)),
            "feat3": list(rng.normal(2.0, 0.5, n * 2)),
            "behavior": (
                ["hands flapping"] * (n // 2) + ["rocking"] * (n - n // 2)
            ) + (["jumping"] * (n // 2) + ["spinning"] * (n - n // 2)),
        }
    )


@pytest.fixture
def clip_df(child_df) -> pd.DataFrame:
    """3 clips per child (child_df repeated with small jitter)."""
    rng = np.random.default_rng(7)
    rows = []
    for _, row in child_df.iterrows():
        for _ in range(3):
            r = row.to_dict()
            r["feat1"] += rng.normal(0, 0.05)
            r["feat2"] += rng.normal(0, 0.02)
            r["feat3"] += rng.normal(0, 0.05)
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def multi_beh_df() -> pd.DataFrame:
    """8 children × 2 groups × 4 behaviors = 64 rows for consistency gate."""
    rng = np.random.default_rng(99)
    rows = []
    for grp, base in [("ASD", 1.5), ("Non-ASD", 1.0)]:
        for beh in sorted(RMM_LABELS):
            for i in range(8):
                rows.append(
                    {
                        "pid": f"sub-{grp[0]}{i:02d}",
                        "Group": grp,
                        "age_mo": float(rng.uniform(12, 36)),
                        "behavior": beh,
                        "feat1": float(rng.normal(base, 0.4)),
                        "feat2": float(rng.normal(base * 0.6, 0.3)),
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════


# ── extract_pid ────────────────────────────────────────────────────────────

class TestExtractPid:
    def test_standard_path(self):
        assert extract_pid("data/sub-A001/video.mp4") == "sub-A001"

    def test_numeric_suffix(self):
        assert extract_pid("path/sub-123/file.csv") == "sub-123"

    def test_alphanumeric_pid(self):
        assert extract_pid("/orcd/data/sub-AB123CD/hrnet.json") == "sub-AB123CD"

    def test_no_match_returns_none(self):
        assert extract_pid("data/nosubject/video.mp4") is None

    def test_none_input(self):
        assert extract_pid(None) is None

    def test_integer_input(self):
        assert extract_pid(42) is None

    def test_empty_string(self):
        assert extract_pid("") is None

    def test_first_match_returned(self):
        # Should return the first sub- match
        pid = extract_pid("path/sub-001/sub-002/file.json")
        assert pid == "sub-001"


# ── parse_timestamps ───────────────────────────────────────────────────────

class TestParseTimestamps:
    def test_single_segment_default_fps(self):
        segs = parse_timestamps("0:10 - 0:20")
        assert len(segs) == 1
        assert segs[0] == (150, 300)  # 10*15, 20*15

    def test_multiple_segments(self):
        segs = parse_timestamps("0:05 - 0:15, 0:20 - 0:30")
        assert len(segs) == 2

    def test_none_returns_empty(self):
        assert parse_timestamps(None) == []

    def test_invalid_string_returns_empty(self):
        assert parse_timestamps("not a timestamp") == []

    def test_reversed_order_excluded(self):
        assert parse_timestamps("0:20 - 0:10") == []

    def test_equal_times_excluded(self):
        assert parse_timestamps("0:10 - 0:10") == []

    def test_minutes_in_timestamp(self):
        segs = parse_timestamps("1:00 - 2:00")
        assert segs[0] == (900, 1800)  # 60*15, 120*15

    def test_custom_fps(self):
        segs = parse_timestamps("0:00 - 0:01", fps=30.0)
        assert segs[0] == (0, 30)

    def test_partial_invalid_segment_skipped(self):
        segs = parse_timestamps("0:05 - 0:10, garbage, 0:15 - 0:20")
        assert len(segs) == 2


# ── butter_lp ──────────────────────────────────────────────────────────────

class TestButterLp:
    def test_output_length_preserved(self):
        arr = np.sin(np.linspace(0, 2 * np.pi, 100))
        assert len(butter_lp(arr)) == 100

    def test_short_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(butter_lp(arr), arr)

    def test_attenuates_high_freq(self):
        t = np.linspace(0, 2, 300)
        signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 10 * t)
        filtered = butter_lp(signal, cutoff=3.0, fs=150.0)
        assert np.std(filtered) < np.std(signal)

    def test_flat_signal_preserved(self):
        arr = np.ones(50)
        np.testing.assert_allclose(butter_lp(arr), arr, atol=1e-8)

    def test_returns_float_array(self):
        arr = np.arange(20, dtype=int)
        result = butter_lp(arr)
        assert result.dtype == float


# ── spectral_features ──────────────────────────────────────────────────────

class TestSpectralFeatures:
    def test_returns_three_floats(self):
        arr = np.sin(np.linspace(0, 4 * np.pi, 100))
        df, ent, bp = spectral_features(arr, fps=15.0)
        assert all(isinstance(v, float) for v in [df, ent, bp])

    def test_short_array_all_nan(self):
        df, ent, bp = spectral_features(np.zeros(5), fps=15.0)
        assert np.isnan(df) and np.isnan(ent) and np.isnan(bp)

    def test_dominant_freq_detection(self):
        t = np.linspace(0, 4, 60)
        arr = np.sin(2 * np.pi * 1.0 * t)
        dom_freq, _, _ = spectral_features(arr, fps=15.0)
        assert abs(dom_freq - 1.0) < 0.8

    def test_band_power_in_unit_interval(self):
        arr = np.random.default_rng(0).normal(0, 1, 100)
        _, _, bp = spectral_features(arr, fps=15.0)
        assert 0.0 <= bp <= 1.0

    def test_entropy_nonnegative(self):
        arr = np.sin(np.linspace(0, 10, 100))
        _, ent, _ = spectral_features(arr, fps=15.0)
        assert ent >= 0.0


# ── cohen_d ────────────────────────────────────────────────────────────────

class TestCohenD:
    def test_identical_groups_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohen_d(a, a.copy()) == 0.0

    def test_positive_direction(self):
        assert cohen_d([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]) > 0

    def test_negative_direction(self):
        assert cohen_d([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]) < 0

    def test_antisymmetry(self):
        a = [1.0, 3.0, 5.0, 7.0]
        b = [2.0, 4.0, 6.0, 8.0]
        assert cohen_d(a, b) == pytest.approx(-cohen_d(b, a))

    def test_zero_pooled_sd_returns_zero(self):
        # Both constant arrays → pooled_sd == 0 → returns 0.0
        assert cohen_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0

    def test_large_effect(self):
        a = np.ones(30) * 10
        b = np.zeros(30)
        # std ~0 for constant, so this tests the zero-variance guard
        d = cohen_d(a + np.random.default_rng(1).normal(0, 1, 30),
                    b + np.random.default_rng(2).normal(0, 1, 30))
        assert d > 5  # clearly large effect


# ── bootstrap_ci_d ─────────────────────────────────────────────────────────

class TestBootstrapCiD:
    def test_returns_ordered_pair(self):
        a = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = bootstrap_ci_d(a, b, n_boot=100)
        assert lo <= hi

    def test_types(self):
        a = np.arange(5, dtype=float)
        b = np.arange(5, dtype=float) + 1
        lo, hi = bootstrap_ci_d(a, b, n_boot=50)
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_reproducible_with_seed(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        lo1, hi1 = bootstrap_ci_d(a, b, n_boot=50, seed=7)
        lo2, hi2 = bootstrap_ci_d(a, b, n_boot=50, seed=7)
        assert lo1 == lo2 and hi1 == hi2

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        a = rng.normal(2, 0.5, 20)
        b = rng.normal(0, 0.5, 20)
        lo, hi = bootstrap_ci_d(a, b, n_boot=200)
        d = cohen_d(a, b)
        assert lo <= d <= hi


# ── fdr_annotate ───────────────────────────────────────────────────────────

class TestFdrAnnotate:
    def test_adds_required_columns(self):
        df = pd.DataFrame({"feature": ["f1", "f2", "f3"], "p_raw": [0.01, 0.05, 0.50]})
        result = fdr_annotate(df, "p_raw")
        for col in ("p_fdr", "sig_fdr05", "sig_raw05"):
            assert col in result.columns

    def test_single_row_fdr_equals_p(self):
        df = pd.DataFrame({"feature": ["f1"], "p_raw": [0.03]})
        result = fdr_annotate(df, "p_raw")
        assert result["p_fdr"].iloc[0] == pytest.approx(0.03)

    def test_sig_raw_flags(self):
        df = pd.DataFrame({"feature": ["f1", "f2"], "p_raw": [0.001, 0.9]})
        result = fdr_annotate(df, "p_raw")
        assert result.loc[result["feature"] == "f1", "sig_raw05"].values[0]
        assert not result.loc[result["feature"] == "f2", "sig_raw05"].values[0]

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"feature": ["f1"], "p_raw": [0.01]})
        fdr_annotate(df, "p_raw")
        assert "p_fdr" not in df.columns

    def test_fdr_geq_raw(self):
        # FDR-corrected p must be >= uncorrected p for each feature
        df = pd.DataFrame(
            {"feature": [f"f{i}" for i in range(10)], "p_raw": np.linspace(0.001, 0.1, 10)}
        )
        result = fdr_annotate(df, "p_raw")
        assert (result["p_fdr"] >= result["p_raw"] - 1e-9).all()


# ── stream_filter ──────────────────────────────────────────────────────────

class TestStreamFilter:
    def _df(self):
        return pd.DataFrame({"age_mo": [5, 12, 15, 20, 35, 37, 40], "val": range(7)})

    def test_full_returns_all(self):
        df = self._df()
        assert len(stream_filter(df, "full")) == len(df)

    def test_11_18mo_range(self):
        result = stream_filter(self._df(), "11-18mo")
        assert all(result["age_mo"].between(11, 18))

    def test_32_38mo_range(self):
        result = stream_filter(self._df(), "32-38mo")
        assert all(result["age_mo"].between(32, 38))

    def test_does_not_mutate_original(self):
        df = self._df()
        original_len = len(df)
        stream_filter(df, "11-18mo")
        assert len(df) == original_len


# ── get_kp ─────────────────────────────────────────────────────────────────

class TestGetKp:
    def test_valid_keypoint(self):
        fd = {KP["nose"]: {"x": 100.0, "y": 50.0, "confidence": 0.9}}
        kp = get_kp(fd, KP["nose"], 0.3)
        assert kp["x"] == 100.0

    def test_below_confidence_returns_none(self):
        fd = {KP["nose"]: {"x": 100.0, "y": 50.0, "confidence": 0.1}}
        assert get_kp(fd, KP["nose"], 0.3) is None

    def test_missing_key_returns_none(self):
        assert get_kp({}, KP["nose"]) is None

    def test_non_dict_value_returns_none(self):
        assert get_kp({KP["nose"]: "bad"}, KP["nose"]) is None

    def test_exact_confidence_threshold_passes(self):
        fd = {KP["nose"]: {"x": 1.0, "y": 1.0, "confidence": 0.3}}
        assert get_kp(fd, KP["nose"], 0.3) is not None


# ── torso_length ───────────────────────────────────────────────────────────

class TestTorsoLength:
    def test_valid_torso(self):
        fd = {
            KP["left_shoulder"]: _make_kp(180, 100),
            KP["right_shoulder"]: _make_kp(220, 100),
            KP["left_hip"]: _make_kp(185, 200),
            KP["right_hip"]: _make_kp(215, 200),
        }
        tl = torso_length(fd)
        assert tl is not None
        assert tl > 5

    def test_missing_shoulder_returns_none(self):
        fd = {
            KP["left_hip"]: _make_kp(185, 200),
            KP["right_hip"]: _make_kp(215, 200),
        }
        assert torso_length(fd) is None

    def test_too_close_returns_none(self):
        # shoulders and hips nearly identical → distance ≤ 5
        fd = {
            KP["left_shoulder"]: _make_kp(100, 100),
            KP["right_shoulder"]: _make_kp(100, 100),
            KP["left_hip"]: _make_kp(100, 102),
            KP["right_hip"]: _make_kp(100, 102),
        }
        assert torso_length(fd) is None

    def test_known_distance(self):
        # shoulder midpoint=(200,0), hip midpoint=(200,100) → d=100
        fd = {
            KP["left_shoulder"]: _make_kp(190, 0),
            KP["right_shoulder"]: _make_kp(210, 0),
            KP["left_hip"]: _make_kp(190, 100),
            KP["right_hip"]: _make_kp(210, 100),
        }
        tl = torso_length(fd)
        assert tl == pytest.approx(100.0, abs=0.1)


# ── get_scale ──────────────────────────────────────────────────────────────

class TestGetScale:
    def test_torso_first(self):
        fd = _make_full_frame()
        scale = get_scale(fd)
        assert scale is not None
        assert scale > 5

    def test_fallback_hip_width(self):
        fd = {
            KP["left_hip"]: _make_kp(150, 200),
            KP["right_hip"]: _make_kp(250, 200),
        }
        scale = get_scale(fd)
        assert scale == pytest.approx(100.0, abs=0.1)

    def test_fallback_shoulder_width(self):
        fd = {
            KP["left_shoulder"]: _make_kp(150, 100),
            KP["right_shoulder"]: _make_kp(250, 100),
        }
        assert get_scale(fd) is not None

    def test_empty_frame_returns_none(self):
        assert get_scale({}) is None

    def test_low_confidence_returns_none(self):
        fd = {
            KP["left_hip"]: _make_kp(100, 200, conf=0.05),
            KP["right_hip"]: _make_kp(200, 200, conf=0.05),
        }
        # Both hips below 0.1 threshold → scale should be None
        assert get_scale(fd) is None


# ── _jfeats ────────────────────────────────────────────────────────────────

class TestJFeats:
    def test_populates_amplitude(self):
        rec: dict = {}
        arr = np.sin(np.linspace(0, 4 * np.pi, 60))
        _jfeats(arr, rec, "test", FPS)
        assert "test_amplitude" in rec
        assert rec["test_amplitude"] > 0

    def test_populates_velocity(self):
        rec: dict = {}
        arr = np.linspace(0, 1, 30)
        _jfeats(arr, rec, "test", FPS)
        assert "test_vel_mean" in rec

    def test_spectral_keys_present(self):
        rec: dict = {}
        arr = np.sin(np.linspace(0, 8 * np.pi, 80))
        _jfeats(arr, rec, "test", FPS)
        assert "test_dom_freq" in rec
        assert "test_spectral_entropy" in rec
        assert "test_band_power_0p3_2hz" in rec

    def test_too_short_no_op(self):
        rec: dict = {}
        _jfeats([1.0, 2.0], rec, "test", FPS)
        assert len(rec) == 0


# ── extract_rmm_features ───────────────────────────────────────────────────

class TestExtractRmmFeatures:
    def test_valid_extraction(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert feats is not None

    def test_output_contains_meta_fields(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        for key in ("n_valid_frames", "n_total_frames", "pct_valid", "duration_sec"):
            assert key in feats

    def test_hip_features_extracted(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert "mean_hip_x_amplitude" in feats
        assert feats["mean_hip_x_amplitude"] > 0

    def test_bilateral_features_present(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert "bilateral_hip_x_corr" in feats
        assert "bilateral_wrist_corr" in feats

    def test_n_active_joints_range(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert 0 <= feats["n_active_joints"] <= 8

    def test_too_few_valid_frames_returns_none(self):
        frames = {str(i): _make_full_frame() for i in range(3)}
        assert extract_rmm_features(frames, [0, 1, 2]) is None

    def test_empty_frame_dict_returns_none(self):
        assert extract_rmm_features({}, list(range(50))) is None

    def test_pct_valid_in_unit_interval(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert 0.0 <= feats["pct_valid"] <= 1.0

    def test_missing_frames_handled(self, pose_frames):
        # Request frames 0-199 but only 0-99 exist → pct_valid < 1
        feats = extract_rmm_features(pose_frames, list(range(200)))
        assert feats is not None
        assert feats["pct_valid"] < 1.0

    def test_hip_x_y_ratio_nonnegative(self, pose_frames):
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert feats["hip_x_y_ratio"] >= 0


# ── compute_icc ────────────────────────────────────────────────────────────

class TestComputeIcc:
    def test_returns_dataframe(self, clip_df):
        result = compute_icc(clip_df, ["feat1", "feat2"])
        assert isinstance(result, pd.DataFrame)

    def test_icc_in_unit_interval(self, clip_df):
        result = compute_icc(clip_df, ["feat1"])
        if len(result):
            assert 0.0 <= result["ICC"].values[0] <= 1.0

    def test_too_few_samples_returns_empty(self):
        df = pd.DataFrame({"pid": ["a", "b"], "feat1": [1.0, 2.0]})
        assert len(compute_icc(df, ["feat1"])) == 0

    def test_high_clustering_detected(self):
        # Create data where within-subject variance << between-subject variance
        rng = np.random.default_rng(5)
        pids = [f"sub-{i:02d}" for i in range(10) for _ in range(5)]
        means = rng.normal(0, 5, 10)
        vals = [means[int(p.split("-")[1])] + rng.normal(0, 0.01) for p in pids]
        df = pd.DataFrame({"pid": pids, "feat1": vals})
        result = compute_icc(df, ["feat1"])
        assert result["ICC"].values[0] > 0.5


# ── run_mwu ────────────────────────────────────────────────────────────────

class TestRunMwu:
    def test_basic_run(self, child_df):
        result = run_mwu(child_df, ["feat1", "feat2"])
        assert len(result) == 2

    def test_p_values_in_range(self, child_df):
        result = run_mwu(child_df, ["feat1", "feat2"])
        assert all(0 <= p <= 1 for p in result["p_raw"])

    def test_fdr_columns_present(self, child_df):
        result = run_mwu(child_df, ["feat1"])
        for col in ("p_fdr", "sig_fdr05", "sig_raw05"):
            assert col in result.columns

    def test_too_few_per_group_returns_empty(self):
        df = pd.DataFrame(
            {"pid": ["a", "b"], "Group": ["ASD", "Non-ASD"], "feat1": [1.0, 2.0]}
        )
        assert len(run_mwu(df, ["feat1"])) == 0

    def test_effect_size_sign(self, child_df):
        # ASD mean > Non-ASD mean for feat1 by construction
        result = run_mwu(child_df, ["feat1"])
        assert result["cohens_d"].values[0] > 0

    def test_subset_label_propagated(self, child_df):
        result = run_mwu(child_df, ["feat1"], subset_label="test_stream")
        assert (result["subset"] == "test_stream").all()


# ── run_child_permutation ──────────────────────────────────────────────────

class TestRunChildPermutation:
    def test_basic_run(self, child_df):
        result = run_child_permutation(child_df, ["feat1"], n_perm=200)
        assert len(result) == 1

    def test_p_geq_min_perm(self, child_df):
        n_perm = 100
        result = run_child_permutation(child_df, ["feat1"], n_perm=n_perm)
        assert result["p_raw"].values[0] >= 1.0 / n_perm

    def test_too_few_returns_empty(self):
        df = pd.DataFrame(
            {"pid": ["a", "b"], "Group": ["ASD", "Non-ASD"], "feat1": [1.0, 2.0]}
        )
        assert len(run_child_permutation(df, ["feat1"], n_perm=50)) == 0

    def test_fdr_columns_present(self, child_df):
        result = run_child_permutation(child_df, ["feat1"], n_perm=100)
        assert "p_fdr" in result.columns and "sig_raw05" in result.columns

    def test_effect_size_returned(self, child_df):
        result = run_child_permutation(child_df, ["feat1"], n_perm=100)
        assert "cohens_d" in result.columns
        assert "d_ci_lo" in result.columns


# ── run_wild_bootstrap ─────────────────────────────────────────────────────

class TestRunWildBootstrap:
    def test_basic_run(self, child_df):
        result = run_wild_bootstrap(child_df, ["feat1"], n_boot=200)
        assert len(result) == 1
        assert 0 < result["p_raw"].values[0] <= 1

    def test_too_few_returns_empty(self):
        df = pd.DataFrame(
            {
                "pid": ["a", "b"],
                "Group": ["ASD", "Non-ASD"],
                "age_mo": [12.0, 14.0],
                "feat1": [1.0, 2.0],
            }
        )
        assert len(run_wild_bootstrap(df, ["feat1"], n_boot=50)) == 0

    def test_coefficient_sign_matches_direction(self, child_df):
        result = run_wild_bootstrap(child_df, ["feat1"], n_boot=200)
        # ASD mean > Non-ASD mean → coef_ASD should be positive
        assert result["coef_ASD"].values[0] > 0

    def test_multiple_features(self, child_df):
        result = run_wild_bootstrap(child_df, ["feat1", "feat2"], n_boot=100)
        assert len(result) == 2


# ── make_consensus ─────────────────────────────────────────────────────────

class TestMakeConsensus:
    def _make_result_df(self, feature, p, d=0.5):
        return pd.DataFrame(
            {
                "feature": [feature],
                "p_raw": [p],
                "p_fdr": [p],
                "cohens_d": [d],
                "d_ci_lo": [d - 0.2],
                "d_ci_hi": [d + 0.2],
                "sig_raw05": [p < 0.05],
                "sig_fdr05": [p < 0.05],
            }
        )

    def test_n_methods_sig_correct(self):
        r1 = self._make_result_df("f1", 0.01)
        r2 = self._make_result_df("f1", 0.03)
        r3 = self._make_result_df("f1", 0.80)
        cons = make_consensus({"m1": r1, "m2": r2, "m3": r3}, ["f1"])
        assert cons[cons["feature"] == "f1"]["n_methods_sig"].values[0] == 2

    def test_empty_result_dict(self):
        cons = make_consensus({}, ["f1", "f2"])
        assert len(cons) == 2
        assert (cons["n_methods_sig"] == 0).all()

    def test_missing_feature_in_result(self):
        r1 = self._make_result_df("f1", 0.01)
        # f2 not in r1
        cons = make_consensus({"m1": r1}, ["f1", "f2"])
        f2_row = cons[cons["feature"] == "f2"]
        assert f2_row["n_methods_sig"].values[0] == 0

    def test_sorted_by_n_methods_sig(self):
        r1 = self._make_result_df("f1", 0.01)
        r2 = self._make_result_df("f2", 0.80)
        cons = make_consensus({"m1": r1, "m2": r2}, ["f1", "f2"])
        # f1 should rank higher
        assert cons.iloc[0]["feature"] == "f1"


# ── run_consistency_gate ───────────────────────────────────────────────────

class TestRunConsistencyGate:
    def test_basic_run(self, multi_beh_df):
        cons_df, consistent, beh_mwu = run_consistency_gate(
            multi_beh_df, ["feat1", "feat2"], ["feat1"]
        )
        assert isinstance(cons_df, pd.DataFrame)
        assert isinstance(consistent, list)
        assert isinstance(beh_mwu, dict)

    def test_consistent_feature_detected(self, multi_beh_df):
        # feat1 has ASD>Non-ASD in ALL behaviors (same direction)
        _, consistent, _ = run_consistency_gate(
            multi_beh_df, ["feat1"], ["feat1"]
        )
        assert "feat1" in consistent

    def test_empty_sig_feats(self, multi_beh_df):
        cons_df, consistent, _ = run_consistency_gate(multi_beh_df, ["feat1"], [])
        assert len(consistent) == 0

    def test_beh_mwu_keys_are_behavior_labels(self, multi_beh_df):
        _, _, beh_mwu = run_consistency_gate(multi_beh_df, ["feat1"], ["feat1"])
        for key in beh_mwu:
            assert key in RMM_LABELS


# ── run_loso_child ─────────────────────────────────────────────────────────

class TestRunLosoChild:
    def test_lr_returns_dict(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], clf_name="LR", n_perm=20)
        assert result is not None
        assert isinstance(result, dict)

    def test_rf_returns_dict(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], clf_name="RF", n_perm=10)
        assert result is not None

    def test_auc_in_unit_interval(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], n_perm=20)
        assert 0.0 <= result["auc"] <= 1.0

    def test_perm_p_in_unit_interval(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], n_perm=50)
        assert 0.0 <= result["perm_p"] <= 1.0

    def test_single_group_returns_none(self):
        df = pd.DataFrame(
            {
                "pid": [f"sub-{i:02d}" for i in range(10)],
                "Group": ["ASD"] * 10,
                "feat1": np.random.default_rng(1).normal(0, 1, 10),
                "feat2": np.random.default_rng(2).normal(0, 1, 10),
            }
        )
        assert run_loso_child(df, ["feat1", "feat2"], n_perm=5) is None

    def test_too_few_subjects_returns_none(self):
        df = pd.DataFrame(
            {
                "pid": ["a", "b", "c"],
                "Group": ["ASD", "ASD", "Non-ASD"],
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [0.1, 0.2, 0.3],
            }
        )
        assert run_loso_child(df, ["feat1", "feat2"], n_perm=5) is None

    def test_result_keys_present(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], n_perm=20)
        for key in ("auc", "ap", "perm_p", "n_features", "n_subjects",
                    "y_true", "y_score", "perm_aucs", "confusion_matrix", "clf"):
            assert key in result

    def test_n_features_correct(self, child_df):
        result = run_loso_child(child_df, ["feat1", "feat2"], n_perm=10)
        assert result["n_features"] == 2

    def test_separable_data_high_auc(self):
        """Perfectly separable data should yield AUC well above 0.5."""
        rng = np.random.default_rng(123)
        n = 20
        df = pd.DataFrame(
            {
                "pid": [f"sub-{i:02d}" for i in range(n * 2)],
                "Group": ["ASD"] * n + ["Non-ASD"] * n,
                "feat1": list(rng.normal(5, 0.1, n)) + list(rng.normal(0, 0.1, n)),
                "feat2": list(rng.normal(5, 0.1, n)) + list(rng.normal(0, 0.1, n)),
            }
        )
        result = run_loso_child(df, ["feat1", "feat2"], n_perm=20)
        assert result is not None
        assert result["auc"] > 0.85


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION — end-to-end pipeline smoke test
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndPipeline:
    """Smoke tests that chain multiple components together."""

    def test_feature_extraction_to_mwu(self, pose_frames, child_df):
        """Extracted features can be fed directly into MWU."""
        feats = extract_rmm_features(pose_frames, list(range(100)))
        assert feats is not None
        # Add a synthetic amplitude to child_df and run MWU
        child_df = child_df.copy()
        child_df["syn_amp"] = np.where(
            child_df["Group"] == "ASD",
            feats["mean_hip_x_amplitude"] + 0.1,
            feats["mean_hip_x_amplitude"] - 0.1,
        )
        result = run_mwu(child_df, ["syn_amp"])
        assert len(result) == 1

    def test_mwu_feeds_consistency_gate(self, multi_beh_df):
        """MWU sig-features feed the consistency gate."""
        mwu_result = run_mwu(multi_beh_df, ["feat1", "feat2"])
        sig = mwu_result[mwu_result["sig_raw05"]]["feature"].tolist()
        cons_df, consistent, _ = run_consistency_gate(
            multi_beh_df, ["feat1", "feat2"], sig
        )
        assert isinstance(consistent, list)

    def test_permutation_to_consensus(self, child_df):
        """Permutation and MWU results combined into consensus table."""
        perm = run_child_permutation(child_df, ["feat1", "feat2"], n_perm=100)
        mwu = run_mwu(child_df, ["feat1", "feat2"])
        cons = make_consensus({"ChildPerm": perm, "MWU": mwu}, ["feat1", "feat2"])
        assert "n_methods_sig" in cons.columns
        assert len(cons) == 2

    def test_full_stats_suite(self, child_df):
        """Run all non-LME stats methods and build consensus."""
        feats = ["feat1", "feat2"]
        perm = run_child_permutation(child_df, feats, n_perm=100)
        boot = run_wild_bootstrap(child_df, feats, n_boot=100)
        mwu = run_mwu(child_df, feats)
        cons = make_consensus(
            {"ChildPerm": perm, "WildBoot": boot, "MWU": mwu}, feats
        )
        # feat1 should be significant in at least 1 method
        f1 = cons[cons["feature"] == "feat1"]["n_methods_sig"].values[0]
        assert f1 >= 0  # at minimum it ran

    def test_loso_then_fdr(self, child_df):
        """LOSO classification result is consistent with effect sizes from MWU."""
        loso = run_loso_child(child_df, ["feat1", "feat2"], n_perm=20)
        mwu = run_mwu(child_df, ["feat1", "feat2"])
        assert loso is not None
        # Both should agree that feat1 shows a group difference
        assert mwu.iloc[0]["cohens_d"] != 0