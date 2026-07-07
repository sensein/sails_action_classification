#!/usr/bin/env python3

import json
import os
import re
import traceback
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, welch
from scipy.stats import gaussian_kde
from scipy.stats import norm as spnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Optional heavy dependencies ──────────────────────────────────────
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    _lme4 = importr('lme4'); _lmerTest = importr('lmerTest')
    _RPY2_OK = True
    print("[rpy2] lme4+lmerTest available — Kenward-Roger enabled")
except Exception:
    _RPY2_OK = False
    print("[rpy2] NOT available — statsmodels LME fallback (no KR)")

try:
    import arviz as az
    import pymc as pm
    _PYMC_OK = True
    print("[PyMC] available")
except Exception:
    _PYMC_OK = False
    print("[PyMC] NOT available — skipping Bayesian section")

try:
    from wildboottest.wildboottest import WildboottestHC
    _WBT_OK = True
    print("[wildboottest] available — CR2 enabled")
except Exception:
    _WBT_OK = False
    print("[wildboottest] NOT available — skipping CR2")

# ═══════════════════════════════════════════════════════════════════
# SHARED CONFIG
# ═══════════════════════════════════════════════════════════════════
MAIN_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
RMM_CSV  = "/home/aparnabg/orcd/scratch/all_project_files/phase_2_analyais/clip_to_csv_matching.csv"
BASE_DIR = (
    "/orcd/data/satra/002/projects/SAILS/action_outputs_features/analysis/handflapping/v3"
)



FPS      = 15.0
MIN_CONF = 0.3

# Bayesian settings
BAYES_DRAWS  = 2000
BAYES_TUNE   = 1000
BAYES_CHAINS = 4
RUN_BAYESIAN = True
PRIOR_SDS    = [0.3, 0.5, 1.0]   # prior sensitivity widths

KP = {
    'left_shoulder':    'kp_005',
    'right_shoulder':   'kp_006',
    'left_elbow':       'kp_007',
    'right_elbow':      'kp_008',
    'left_wrist':       'kp_009',
    'right_wrist':      'kp_010',
    'left_hip':         'kp_011',
    'right_hip':        'kp_012',
    'left_hand_wrist':  'kp_091',
    'right_hand_wrist': 'kp_112',
}

AGE_STREAMS = {'full': None, '11-18mo': (11, 18), '32-38mo': (32, 38)}
AGE_BANDS   = {'11-18mo': (11, 18), '19-31mo': (19, 31), '32-38mo': (32, 38)}
STAT_BANDS  = ['11-18mo', '32-38mo']
LABEL_REFERENCE = 'hands flapping'   # reference category for LME dummies

ASD_COLOR    = '#E05C5C'; NONASD_COLOR = '#5B8DB8'
ASD_LIGHT    = '#F2AEAE'; NONASD_LIGHT = '#A8C8E8'
COLORS       = {'ASD': ASD_COLOR, 'Non-ASD': NONASD_COLOR}
COLORS_LIGHT = {'ASD': ASD_LIGHT, 'Non-ASD': NONASD_LIGHT}
STREAM_COLORS = {'full': '#555555', '11-18mo': '#7B5EA7', '32-38mo': '#D47C2A'}
GROUPS = ['ASD', 'Non-ASD']

SHORT_LABELS = {
    'wrist_amp_max': 'Wrist Amp (max)',
    'wrist_amp_mean': 'Wrist Amp (mean)',
    'wrist_vel_max': 'Wrist Vel (max)',
    'wrist_vel_mean': 'Wrist Vel (mean)',
    'wrist_y_L_amplitude': 'Amp L(y)',
    'wrist_y_R_amplitude': 'Amp R(y)',
    'wrist_y_L_vel_mean': 'Vel L mean',
    'wrist_y_R_vel_mean': 'Vel R mean',
    'wrist_y_L_acc_mean': 'Acc L mean',
    'wrist_y_R_acc_mean': 'Acc R mean',
    'wrist_y_L_dom_freq': 'Dom Freq L',
    'wrist_y_R_dom_freq': 'Dom Freq R',
    'bilateral_amp_diff': 'Bilat Amp Diff',
    'bilateral_y_corr': 'Bilat Y Corr',
    'bilateral_sym_index': 'Bilat Symmetry',
    'bilateral_phase_lag_sec': 'Phase Lag',
    'elbow_y_L_amplitude': 'Elbow Amp L',
    'elbow_y_R_amplitude': 'Elbow Amp R',
    'wrist_y_L_spectral_entropy': 'Entropy L',
    'wrist_y_R_spectral_entropy': 'Entropy R',
}

VARIANTS = [
    {
        'name':            'hands_and_onehand',
        'flapping_labels': {'hands flapping', 'one hand flap'},
        'output_dir':      os.path.join(BASE_DIR, 'v3_hands_and_onehand'),
    },
    {
        'name':            'full_analysis',
        'flapping_labels': {'hands flapping', 'arm flapping', 'arms flapping',
                            'one hand flap', 'one arm flap'},
        'output_dir':      os.path.join(BASE_DIR, 'v3_full_analysis'),
    },
    {
        'name':            'hands_only',
        'flapping_labels': {'hands flapping'},
        'output_dir':      os.path.join(BASE_DIR, 'v3_hands_only'),
    },
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

# ═══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════
def hr(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def save_fig(fig, name, figure_dir):
    fig.savefig(os.path.join(figure_dir, name))
    plt.close(fig)
    print(f"  Saved {name}")

def extract_pid(path):
    if not isinstance(path, str): return None
    m = re.search(r'(sub-[A-Za-z0-9]+)', path)
    return m.group(1) if m else None

def parse_timestamps(ts_str, fps=FPS):
    if not isinstance(ts_str, str): return []
    segs = []
    for part in ts_str.split(','):
        m = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)', part.strip())
        if m:
            s = int(m.group(1))*60 + int(m.group(2))
            e = int(m.group(3))*60 + int(m.group(4))
            if e > s:
                segs.append((int(s*fps), int(e*fps)))
    return segs

def get_kp(fd, key, min_conf=MIN_CONF):
    if key not in fd: return None
    kp = fd[key]
    if not isinstance(kp, dict): return None
    if kp.get('confidence', 0) < min_conf: return None
    return kp

def butter_lp(data, cutoff=6.0, fs=15.0, order=2):
    arr = np.array(data, dtype=float)
    if len(arr) < 10: return arr
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff, nyq*0.9)/nyq, btype='low')
    if len(arr) < 3*max(len(b), len(a)): return arr
    return filtfilt(b, a, arr)

def torso_length(fd):
    ls = get_kp(fd, KP['left_shoulder'],  min_conf=0.1)
    rs = get_kp(fd, KP['right_shoulder'], min_conf=0.1)
    lh = get_kp(fd, KP['left_hip'],       min_conf=0.1)
    rh = get_kp(fd, KP['right_hip'],      min_conf=0.1)
    if not all([ls, rs, lh, rh]): return None
    sx = (ls['x'] + rs['x']) / 2; sy = (ls['y'] + rs['y']) / 2
    hx = (lh['x'] + rh['x']) / 2; hy = (lh['y'] + rh['y']) / 2
    d  = np.sqrt((sx-hx)**2 + (sy-hy)**2)
    return d if d > 5 else None

def spectral_features(arr, fps):
    if len(arr) < 16: return np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq = freqs[np.argmax(psd)]
        psd_n = psd / (psd.sum() + 1e-12)
        entropy = -np.sum(psd_n[psd_n>0] * np.log2(psd_n[psd_n>0]))
        return float(dom_freq), float(entropy)
    except:
        return np.nan, np.nan

def cohen_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def bootstrap_ci_d(a, b, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    boot = [cohen_d(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def cles(a, b):
    a, b = np.array(a), np.array(b)
    count = sum(1 for ai in a for bi in b if ai > bi)
    return count / (len(a) * len(b))

def fdr_annotate(df_res, p_col):
    if len(df_res) > 1:
        _, p_fdr, _, _ = multipletests(df_res[p_col].fillna(1), method='fdr_bh')
        df_res = df_res.copy(); df_res['p_fdr'] = p_fdr
    else:
        df_res = df_res.copy(); df_res['p_fdr'] = df_res[p_col]
    df_res['sig_fdr05'] = df_res['p_fdr'] < 0.05
    df_res['sig_raw05'] = df_res[p_col] < 0.05
    return df_res

def add_sig_bar(ax, x1, x2, y, p, h=0.02):
    label = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col = '#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], lw=1.2, color='black')
    ax.text((x1+x2)/2, y+h*1.05, label, ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')

def assign_age_band(age_mo):
    for band, (lo, hi) in AGE_BANDS.items():
        if lo <= age_mo <= hi: return band
    return None

def stream_filter(df, stream_key):
    bounds = AGE_STREAMS[stream_key]
    if bounds is None: return df.copy()
    lo, hi = bounds
    return df[(df['age_mo'] >= lo) & (df['age_mo'] <= hi)].copy()

def _add_label_dummies(df, reference=LABEL_REFERENCE):
    """Add dummy columns for non-reference labels (for LME formula)."""
    df = df.copy()
    labels = sorted(df['original_label'].dropna().unique())
    non_ref = [l for l in labels if l != reference]
    for lb in non_ref:
        col = 'lbl_' + re.sub(r'[^A-Za-z0-9]', '_', lb)
        df[col] = (df['original_label'] == lb).astype(float)
    dummy_cols = ['lbl_' + re.sub(r'[^A-Za-z0-9]', '_', lb) for lb in non_ref]
    return df, dummy_cols

# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════
def extract_flapping_features(pose_frames, frame_indices, ann_fps=FPS):
    wrist_y_L, wrist_y_R = [], []
    wrist_x_L, wrist_x_R = [], []
    elbow_y_L, elbow_y_R = [], []
    conf_L, conf_R = [], []
    torso_lens = []
    n_valid = 0

    for fi in frame_indices:
        fk = str(fi)
        if fk not in pose_frames: continue
        fd = pose_frames[fk]
        tl = torso_length(fd)
        if tl is None: continue
        lw = get_kp(fd, KP['left_wrist']) or get_kp(fd, KP['left_hand_wrist'])
        rw = get_kp(fd, KP['right_wrist']) or get_kp(fd, KP['right_hand_wrist'])
        le = get_kp(fd, KP['left_elbow'])
        re = get_kp(fd, KP['right_elbow'])
        if lw is None and rw is None: continue
        torso_lens.append(tl)
        n_valid += 1
        if lw:
            wrist_y_L.append(lw['y'] / tl)
            wrist_x_L.append(lw['x'] / tl)
            conf_L.append(lw['confidence'])
        if rw:
            wrist_y_R.append(rw['y'] / tl)
            wrist_x_R.append(rw['x'] / tl)
            conf_R.append(rw['confidence'])
        if le: elbow_y_L.append(le['y'] / tl)
        if re: elbow_y_R.append(re['y'] / tl)

    if n_valid < 5: return None
    has_L = len(wrist_y_L) >= 5
    has_R = len(wrist_y_R) >= 5
    if not has_L and not has_R: return None

    rec = {
        'n_valid_frames':    n_valid,
        'n_total_frames':    len(frame_indices),
        'pct_valid':         n_valid / len(frame_indices),
        'duration_sec':      len(frame_indices) / ann_fps,
        'mean_torso_length': np.mean(torso_lens),
        'mean_conf_L':       np.mean(conf_L) if conf_L else np.nan,
        'mean_conf_R':       np.mean(conf_R) if conf_R else np.nan,
    }

    def wrist_feats(arr, name):
        a = np.array(arr, dtype=float)
        rec[f'{name}_amplitude'] = float(np.ptp(a))
        rec[f'{name}_std']       = float(np.std(a))
        rec[f'{name}_mean']      = float(np.mean(a))
        rec[f'{name}_iqr']       = float(np.percentile(a,75) - np.percentile(a,25))
        if len(a) >= 8:
            try:
                sm  = butter_lp(a, fs=ann_fps)
                vel = np.diff(sm) * ann_fps
                rec[f'{name}_vel_mean'] = float(np.mean(np.abs(vel)))
                rec[f'{name}_vel_std']  = float(np.std(vel))
                rec[f'{name}_vel_max']  = float(np.max(np.abs(vel)))
                if len(vel) >= 4:
                    acc = np.diff(vel) * ann_fps
                    rec[f'{name}_acc_mean'] = float(np.mean(np.abs(acc)))
                    rec[f'{name}_acc_max']  = float(np.max(np.abs(acc)))
            except: pass
        df_f, se = spectral_features(a, ann_fps)
        rec[f'{name}_dom_freq']         = df_f
        rec[f'{name}_spectral_entropy'] = se

    if has_L: wrist_feats(wrist_y_L, 'wrist_y_L'); wrist_feats(wrist_x_L, 'wrist_x_L')
    if has_R: wrist_feats(wrist_y_R, 'wrist_y_R'); wrist_feats(wrist_x_R, 'wrist_x_R')

    for name, arr in [('elbow_y_L', elbow_y_L), ('elbow_y_R', elbow_y_R)]:
        a = np.array(arr, dtype=float)
        if len(a) >= 5:
            rec[f'{name}_amplitude'] = float(np.ptp(a))
            rec[f'{name}_std']       = float(np.std(a))

    if has_L and has_R:
        ml = min(len(wrist_y_L), len(wrist_y_R))
        yl = np.array(wrist_y_L[:ml]); yr = np.array(wrist_y_R[:ml])
        rec['bilateral_amp_diff']  = float(abs(np.ptp(yl) - np.ptp(yr)))
        rec['bilateral_amp_ratio'] = float(min(np.ptp(yl), np.ptp(yr)) /
                                           (max(np.ptp(yl), np.ptp(yr)) + 1e-8))
        rec['bilateral_y_corr']    = float(np.corrcoef(yl, yr)[0,1])
        rec['bilateral_sym_index'] = float(1 - abs(np.ptp(yl)-np.ptp(yr)) /
                                                  (np.ptp(yl)+np.ptp(yr)+1e-8))
        try:
            xcorr = np.correlate(yl-yl.mean(), yr-yr.mean(), mode='full')
            lags  = np.arange(-(ml-1), ml)
            rec['bilateral_phase_lag_sec'] = float(abs(lags[np.argmax(xcorr)]) / ann_fps)
        except: pass
        rec['wrist_amp_max']  = float(max(np.ptp(yl), np.ptp(yr)))
        rec['wrist_amp_mean'] = float((np.ptp(yl) + np.ptp(yr)) / 2)
        if 'wrist_y_L_vel_mean' in rec and 'wrist_y_R_vel_mean' in rec:
            rec['wrist_vel_max']  = float(max(rec['wrist_y_L_vel_mean'], rec['wrist_y_R_vel_mean']))
            rec['wrist_vel_mean'] = float((rec['wrist_y_L_vel_mean'] + rec['wrist_y_R_vel_mean']) / 2)
    elif has_L:
        rec['wrist_amp_max']  = rec.get('wrist_y_L_amplitude', np.nan)
        rec['wrist_amp_mean'] = rec.get('wrist_y_L_amplitude', np.nan)
        rec['wrist_vel_max']  = rec.get('wrist_y_L_vel_mean',  np.nan)
        rec['wrist_vel_mean'] = rec.get('wrist_y_L_vel_mean',  np.nan)
    elif has_R:
        rec['wrist_amp_max']  = rec.get('wrist_y_R_amplitude', np.nan)
        rec['wrist_amp_mean'] = rec.get('wrist_y_R_amplitude', np.nan)
        rec['wrist_vel_max']  = rec.get('wrist_y_R_vel_mean',  np.nan)
        rec['wrist_vel_mean'] = rec.get('wrist_y_R_vel_mean',  np.nan)

    return rec


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL METHODS — FULL SUITE
# ═══════════════════════════════════════════════════════════════════

# ── Step 0: ICC ──────────────────────────────────────────────────
def compute_icc(clip_df, feat_cols):
    """Intraclass correlation — quantifies within-child clustering."""
    records = []
    for feat in feat_cols:
        sub = clip_df[['pid', feat]].dropna()
        if len(sub) < 10: continue
        groups = [g[feat].values for _, g in sub.groupby('pid') if len(g) >= 2]
        if len(groups) < 5: continue
        n_total = sum(len(g) for g in groups); k = len(groups)
        n0 = (n_total - sum(len(g)**2/n_total for g in groups)) / (k-1)
        grand = np.concatenate(groups)
        ms_between = np.sum([len(g)*(np.mean(g)-np.mean(grand))**2
                             for g in groups]) / (k-1)
        ms_within  = np.sum([np.sum((g-np.mean(g))**2)
                             for g in groups]) / (n_total-k)
        icc = max(0.0, (ms_between-ms_within) /
                       (ms_between + (n0-1)*ms_within))
        records.append({'feature': feat, 'ICC': round(icc,4)})
    return pd.DataFrame(records).sort_values('ICC', ascending=False)


# ── Step 1: LME + Kenward-Roger ──────────────────────────────────
def run_lme_kr(clip_df, feat_cols, subset_label='full'):
    """
    LME with Kenward-Roger df correction via rpy2/lmerTest.
    Falls back to statsmodels MixedLM if rpy2 unavailable.
    Model: feature ~ Group_bin + age_mo_c + label_dummies + (1|pid)
    """
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)

    for feat in feat_cols:
        keep = ['pid', 'Group_bin', 'age_mo_c', 'original_label', feat] + dummy_cols
        sub  = df_use[[c for c in keep if c in df_use.columns]].dropna(
                  subset=['pid', 'Group_bin', feat])
        if sub['Group_bin'].nunique() < 2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min() < 3: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        d  = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        p_val = np.nan; coef = np.nan; se = np.nan
        method_used = 'none'; converged = False

        if _RPY2_OK:
            try:
                safe = re.sub(r'[^A-Za-z0-9_]', '_', feat)
                sub2 = sub.rename(columns={feat: safe})
                bterm = ' + '.join(dummy_cols) if dummy_cols else ''
                formula = (f'{safe} ~ Group_bin + age_mo_c'
                           + (f' + {bterm}' if bterm else '')
                           + ' + (1|pid)')
                r_df = pandas2ri.py2rpy(sub2); ro.globalenv['r_df'] = r_df
                ro.r(f'fit <- lmerTest::lmer({formula}, data=r_df, REML=TRUE)')
                summ = pandas2ri.rpy2py(
                    ro.r('as.data.frame(coef(summary(fit,ddf="Kenward-Roger")))'))
                if 'Group_bin' in summ.index:
                    coef  = float(summ.loc['Group_bin', 'Estimate'])
                    se    = float(summ.loc['Group_bin', 'Std. Error'])
                    p_val = float(summ.loc['Group_bin', 'Pr(>|t|)'])
                    method_used = 'LME_KR'; converged = True
            except: pass

        if method_used == 'none':
            try:
                bterm = '+'.join(dummy_cols) if dummy_cols else ''
                formula_sm = (f'{feat} ~ Group_bin + age_mo_c'
                              + (f' + {bterm}' if bterm else ''))
                mdf = smf.mixedlm(formula_sm, sub, groups=sub['pid']).fit(
                          method=['lbfgs'], reml=True, maxiter=300)
                coef  = float(mdf.params.get('Group_bin', np.nan))
                se    = float(mdf.bse.get('Group_bin', np.nan))
                p_val = float(mdf.pvalues.get('Group_bin', np.nan))
                method_used = 'LME_noKR'; converged = bool(mdf.converged)
            except: pass

        records.append({
            'feature': feat, 'subset': subset_label, 'method': method_used,
            'coef_ASD': coef, 'se': se, 'p_raw': p_val,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'converged': converged,
            'n_asd': sub[sub['Group_bin']==1]['pid'].nunique(),
            'n_nasd': sub[sub['Group_bin']==0]['pid'].nunique(),
            'n_clips': len(sub),
        })

    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 2: CR2 bias-reduced (wildboottest) ───────────────────────
def run_cr2(clip_df, feat_cols, subset_label='full'):
    """CR2 cluster-robust inference — handles small number of clusters."""
    if not _WBT_OK:
        print("  [CR2] skipped — wildboottest not installed"); return pd.DataFrame()
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)

    for feat in feat_cols:
        sub = df_use[['pid', 'Group_bin', 'age_mo_c', feat] + dummy_cols].dropna(
                  subset=['pid', 'Group_bin', feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 10: continue
        X_cols = ['Group_bin', 'age_mo_c'] + [c for c in dummy_cols
                                               if sub[c].std() > 1e-8]
        X = sub[X_cols].values.astype(float)
        y = sub[feat].values.astype(float)
        clusters = sub['pid'].values
        try:
            wbt = WildboottestHC(X=X, y=y, cluster=clusters,
                                 R=np.eye(len(X_cols))[[0],:], B=999,
                                 bootstrap_type='WCR11')
            wbt.get_wildboottest()
            records.append({'feature': feat, 'subset': subset_label,
                            'method': 'CR2', 'p_raw': float(wbt.pvalue),
                            'n_clips': len(sub)})
        except: continue

    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 3: GEE ──────────────────────────────────────────────────
def run_gee(clip_df, feat_cols, subset_label='full'):
    """GEE with exchangeable working correlation — supplementary robustness."""
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)
    pid_map = {p: i for i, p in enumerate(df_use['pid'].unique())}
    df_use['pid_int'] = df_use['pid'].map(pid_map)

    for feat in feat_cols:
        sub = df_use[['pid_int', 'Group_bin', 'age_mo_c', feat] + dummy_cols].dropna(
                  subset=['pid_int', 'Group_bin', feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 20: continue
        counts = sub.groupby('pid_int').size()
        sub = sub[sub['pid_int'].isin(counts[counts >= 2].index)]
        if len(sub) < 20: continue
        try:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', feat)
            sub2 = sub.rename(columns={feat: safe})
            bterm = '+'.join([c for c in dummy_cols if sub2[c].std() > 1e-8])
            formula = (f'{safe} ~ Group_bin + age_mo_c'
                       + (f' + {bterm}' if bterm else ''))
            res = GEE.from_formula(formula, 'pid_int', data=sub2,
                                   family=Gaussian(), cov_struct=Exchangeable()).fit(maxiter=100)
            av = sub[sub['Group_bin']==1][feat].values
            nv = sub[sub['Group_bin']==0][feat].values
            records.append({
                'feature': feat, 'subset': subset_label, 'method': 'GEE',
                'coef_ASD': float(res.params.get('Group_bin', np.nan)),
                'p_raw': float(res.pvalues.get('Group_bin', np.nan)),
                'cohens_d': cohen_d(av, nv), 'n_clips': len(sub),
            })
        except: continue

    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 4: Child-level permutation ──────────────────────────────
def run_child_permutation(child_df, feat_cols, n_perm=5000, subset_label='full'):
    """Permutes group labels at child level — controls for pseudoreplication."""
    rng = np.random.default_rng(42); records = []
    for feat in feat_cols:
        sub = child_df[['pid', 'Group', feat]].dropna()
        if sub['Group'].nunique() < 2: continue
        av = sub[sub['Group']=='ASD'][feat].values
        nv = sub[sub['Group']=='Non-ASD'][feat].values
        if len(av) < 3 or len(nv) < 3: continue
        obs_stat = abs(np.mean(av) - np.mean(nv))
        n_asd = len(av); vals_arr = sub[feat].values
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            sl = rng.permutation(['ASD']*n_asd + ['Non-ASD']*(len(sub)-n_asd))
            a_v = vals_arr[np.array(sl)=='ASD']
            n_v = vals_arr[np.array(sl)=='Non-ASD']
            a_v = a_v[~np.isnan(a_v)]; n_v = n_v[~np.isnan(n_v)]
            perm_stats[i] = (abs(np.mean(a_v)-np.mean(n_v))
                             if len(a_v)>0 and len(n_v)>0 else 0)
        p_perm = max(float(np.mean(perm_stats >= obs_stat)), 1.0/n_perm)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'ChildPerm',
            'obs_stat': float(obs_stat), 'p_raw': p_perm,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': len(av), 'n_nasd': len(nv),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 5: Wild cluster bootstrap ───────────────────────────────
def run_wild_bootstrap(child_df, feat_cols, n_boot=5000, subset_label='full'):
    """Wild cluster bootstrap — small-cluster robust p-values at child level."""
    rng = np.random.default_rng(99); records = []
    df_use = child_df.copy().dropna(subset=['age_mo'])
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)

    for feat in feat_cols:
        sub = df_use[['pid', 'Group_bin', 'age_mo', feat]].dropna()
        if sub['Group_bin'].nunique() < 2: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        if len(av) < 3 or len(nv) < 3: continue
        n = len(sub); y = sub[feat].values.astype(float)
        X = np.column_stack([np.ones(n), sub['Group_bin'].values, sub['age_mo'].values])
        try: beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except: continue
        resid = y - X @ beta; t_obs = beta[1] / (np.std(resid)/np.sqrt(n) + 1e-10)
        X0 = X[:, [0, 2]]
        try: beta0, _, _, _ = np.linalg.lstsq(X0, y, rcond=None)
        except: continue
        resid0 = y - X0 @ beta0
        pids = sub['pid'].values; u_pids = np.unique(pids)
        t_boot = np.zeros(n_boot)
        for b in range(n_boot):
            w_map = {p: rng.choice([-1.0, 1.0]) for p in u_pids}
            w = np.array([w_map[p] for p in pids])
            y_b = X0 @ beta0 + resid0 * w
            try:
                beta_b, _, _, _ = np.linalg.lstsq(X, y_b, rcond=None)
                resid_b = y_b - X @ beta_b
                t_boot[b] = beta_b[1] / (np.std(resid_b)/np.sqrt(n) + 1e-10)
            except: t_boot[b] = 0.0
        p_wb = max(float(np.mean(np.abs(t_boot) >= abs(t_obs))), 1.0/n_boot)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'WildBoot',
            'coef_ASD': float(beta[1]), 't_obs': float(t_obs), 'p_raw': p_wb,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': int(len(av)), 'n_nasd': int(len(nv)),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 6: Pseudo-bulk MWU + CLES ───────────────────────────────
def run_mwu(child_df, feat_cols, subset_label='full'):
    """Mann-Whitney U on child-level means with CLES and bootstrap CI."""
    records = []
    for feat in feat_cols:
        av = child_df[child_df['Group']=='ASD'][feat].dropna().values
        nv = child_df[child_df['Group']=='Non-ASD'][feat].dropna().values
        if len(av) < 3 or len(nv) < 3: continue
        stat, p = stats.mannwhitneyu(av, nv, alternative='two-sided')
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        cl = cles(av, nv)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'PseudobulkMW',
            'asd_median': float(np.median(av)), 'nasd_median': float(np.median(nv)),
            'asd_mean': float(np.mean(av)), 'nasd_mean': float(np.mean(nv)),
            'asd_std': float(np.std(av)), 'nasd_std': float(np.std(nv)),
            'mw_stat': float(stat), 'p_raw': float(p),
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'cles': cl, 'n_asd': len(av), 'n_nasd': len(nv),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 7: Label × Group interaction (LME) ──────────────────────
def run_label_group_interaction(clip_df, feat_cols, subset_label='full'):
    """
    Tests whether the Group effect DIFFERS by flapping label.
    A significant interaction means the ASD/Non-ASD difference is
    label-specific, not a general flapping signature.
    """
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)

    for feat in feat_cols:
        keep = ['pid', 'Group_bin', 'age_mo_c', 'original_label', feat] + dummy_cols
        sub  = df_use[[c for c in keep if c in df_use.columns]].dropna(
                  subset=['pid', 'Group_bin', feat])
        if sub['Group_bin'].nunique() < 2 or sub['pid'].nunique() < 5: continue
        if not dummy_cols: continue
        beh_x_grp = [f'Group_bin:{c}' for c in dummy_cols]
        formula = (f'{feat} ~ Group_bin + age_mo_c + ' + '+'.join(dummy_cols)
                   + ' + ' + '+'.join(beh_x_grp))
        try:
            mdf = smf.mixedlm(formula, sub, groups=sub['pid']).fit(
                      method=['lbfgs'], reml=True, maxiter=300)
            for bxg in beh_x_grp:
                if bxg in mdf.pvalues.index:
                    records.append({
                        'feature': feat, 'interaction_term': bxg,
                        'coef': float(mdf.params.get(bxg, np.nan)),
                        'p_raw': float(mdf.pvalues.get(bxg, np.nan)),
                    })
        except: continue

    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')


# ── Step 8: Consensus ─────────────────────────────────────────────
def make_consensus(results_dict, feat_cols, threshold=0.05):
    """Aggregates p-values across all methods into one summary table."""
    rows = []
    for feat in feat_cols:
        row = {'feature': feat}; n_sig = 0
        for mname, res_df in results_dict.items():
            if res_df is None or len(res_df) == 0:
                row[f'p_{mname}'] = np.nan; continue
            match = res_df[res_df['feature'] == feat]
            if len(match) == 0:
                row[f'p_{mname}'] = np.nan
            else:
                p = match['p_raw'].values[0]; row[f'p_{mname}'] = round(p, 4)
                if p < threshold: n_sig += 1
        row['n_methods_sig'] = n_sig; rows.append(row)
    cons = pd.DataFrame(rows)
    # attach Cohen's d from best available LME
    for lme_key in ['LME_KR', 'LME_noKR', 'PseudobulkMW']:
        ldf = results_dict.get(lme_key)
        if ldf is not None and len(ldf) and 'cohens_d' in ldf.columns:
            cons['cohens_d'] = cons['feature'].map(
                ldf.set_index('feature')['cohens_d'].to_dict())
            if 'd_ci_lo' in ldf.columns:
                cons['d_ci_lo'] = cons['feature'].map(
                    ldf.set_index('feature')['d_ci_lo'].to_dict())
                cons['d_ci_hi'] = cons['feature'].map(
                    ldf.set_index('feature')['d_ci_hi'].to_dict())
            break
    return cons.sort_values('n_methods_sig', ascending=False)


# ── Step 9: Consistency gate ──────────────────────────────────────
def run_consistency_gate(feat_df, feat_cols, sig_feats):
    """
    Checks that the direction of the Group effect (ASD>Non-ASD or vice-versa)
    is the same across all flapping labels. Inconsistent direction = interpret
    the main effect with caution.
    """
    label_mwu = {}
    for lbl in sorted(feat_df['original_label'].dropna().unique()):
        sub = feat_df[feat_df['original_label'] == lbl]
        asd_n  = sub[sub['Group']=='ASD']['pid'].nunique()
        nasd_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        if asd_n < 3 or nasd_n < 3: continue
        recs = []
        for feat in feat_cols:
            av = sub[sub['Group']=='ASD'][feat].dropna().values
            nv = sub[sub['Group']=='Non-ASD'][feat].dropna().values
            if len(av) < 3 or len(nv) < 3: continue
            _, p = stats.mannwhitneyu(av, nv, alternative='two-sided')
            recs.append({'feature': feat, 'cohens_d': cohen_d(av, nv),
                         'p_raw': p, 'label': lbl})
        if recs: label_mwu[lbl] = pd.DataFrame(recs)

    cons_recs = []; consistent_feats = []
    lbl_all = (pd.concat(label_mwu.values(), ignore_index=True)
               if label_mwu else pd.DataFrame())
    for feat in sig_feats:
        if len(lbl_all) == 0: break
        sub = lbl_all[lbl_all['feature'] == feat]
        if len(sub) < 2: continue
        signs = np.sign(sub['cohens_d'].values)
        n_same = int((signs == signs[0]).sum())
        passed = (n_same == len(sub))
        cons_recs.append({'feature': feat, 'n_labels_tested': len(sub),
                          'n_same_direction': n_same, 'consistent': passed})
        if passed: consistent_feats.append(feat)

    return pd.DataFrame(cons_recs), consistent_feats, label_mwu


# ── Step 10: Spearman age correlations ───────────────────────────
def run_spearman_age(clip_df, feat_cols):
    """Spearman correlation of each feature with age, per group."""
    records = []
    for grp in GROUPS:
        sub = clip_df[clip_df['Group'] == grp]
        for feat in feat_cols:
            vals = sub[['age_mo', feat]].dropna()
            if len(vals) < 5: continue
            r, p = stats.spearmanr(vals['age_mo'], vals[feat])
            records.append({'Group': grp, 'feature': feat,
                            'spearman_r': float(r), 'p_raw': float(p),
                            'n': len(vals)})
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records); df['sig_p05'] = df['p_raw'] < 0.05
    return df


# ── Step 11: Full LME suite (age-strat, within-ASD, growth curve) ─
def run_lme_suite(clip_df, feat_cols, subset_label='full'):
    """
    Runs age-stratified LME, within-ASD trajectory, and
    Group × Age growth-curve interaction.
    Returns dict of DataFrames keyed by analysis name.
    """
    results = {}

    # 11a: within-ASD trajectory
    asd_df = clip_df[clip_df['Group'] == 'ASD'].copy()
    recs = []
    if asd_df['pid'].nunique() >= 4:
        asd_df['age_mo_c'] = asd_df['age_mo'] - asd_df['age_mo'].mean()
        for feat in feat_cols:
            sub = asd_df[['pid', 'age_mo_c', feat]].dropna()
            if len(sub) < 6 or sub['pid'].nunique() < 3: continue
            try:
                mdf = smf.mixedlm(f'{feat} ~ age_mo_c', sub,
                                  groups=sub['pid']).fit(
                          method=['lbfgs'], reml=True, maxiter=300)
                recs.append({'feature': feat, 'coef_age': float(mdf.params.get('age_mo_c', np.nan)),
                             'p_raw': float(mdf.pvalues.get('age_mo_c', np.nan)),
                             'n_obs': len(sub), 'n_pids': sub['pid'].nunique()})
            except: continue
    if recs:
        results['asd_trajectory'] = fdr_annotate(pd.DataFrame(recs), 'p_raw').sort_values('p_raw')

    # 11b: Group × Age growth curve
    clip_df2 = clip_df.copy()
    clip_df2['Group_bin'] = (clip_df2['Group'] == 'ASD').astype(float)
    clip_df2['age_mo_c']  = clip_df2['age_mo'] - clip_df2['age_mo'].mean()
    recs = []
    for feat in feat_cols:
        sub = clip_df2[['pid', 'Group_bin', 'age_mo_c', feat]].dropna()
        if sub['Group_bin'].nunique() < 2 or sub['pid'].nunique() < 5: continue
        try:
            mdf = smf.mixedlm(f'{feat} ~ Group_bin * age_mo_c', sub,
                              groups=sub['pid']).fit(
                      method=['lbfgs'], reml=True, maxiter=300)
            key = 'Group_bin:age_mo_c'
            if key in mdf.pvalues.index:
                recs.append({'feature': feat,
                             'coef_interact': float(mdf.params.get(key, np.nan)),
                             'p_raw': float(mdf.pvalues.get(key, np.nan)),
                             'n_obs': len(sub)})
        except: continue
    if recs:
        results['growth_curve'] = fdr_annotate(pd.DataFrame(recs), 'p_raw').sort_values('p_raw')

    # 11c: Age-stratified LME per band
    for band in STAT_BANDS:
        sub_band = clip_df[clip_df['age_band'] == band].copy()
        asd_n  = sub_band[sub_band['Group']=='ASD']['pid'].nunique()
        nasd_n = sub_band[sub_band['Group']=='Non-ASD']['pid'].nunique()
        if asd_n < 3 or nasd_n < 3: continue
        band_recs = []
        for feat in feat_cols:
            r = run_lme_kr(sub_band, [feat], subset_label=band)
            if len(r): band_recs.append(r.iloc[0].to_dict())
        if band_recs:
            results[f'age_strat_{band}'] = pd.DataFrame(band_recs)

    return results


# ═══════════════════════════════════════════════════════════════════
# BAYESIAN HIERARCHICAL LMM
# ═══════════════════════════════════════════════════════════════════
def _standardise(series):
    m, s = series.mean(), series.std()
    s = s if s > 1e-10 else 1.0
    return ((series - m) / s).values, m, s

def _savage_dickey_bf(post, prior_sd=0.5):
    prior_at_0 = spnorm.pdf(0, 0, prior_sd)
    try:
        post_at_0 = gaussian_kde(post)(0)[0]
        return float(prior_at_0 / post_at_0) if post_at_0 > 0 else np.nan
    except: return np.nan

def _build_bayes_df(df, feat):
    cols = ['pid', 'Group', 'age_mo', 'original_label', feat]
    tmp  = df[[c for c in cols if c in df.columns]].dropna().copy()
    if len(tmp) < 8 or tmp['pid'].nunique() < 4: return None
    tmp['Group_bin'] = (tmp['Group'] == 'ASD').astype(float)
    tmp['age_c']     = tmp['age_mo'] - tmp['age_mo'].mean()
    # label dummies for model
    labels = sorted(tmp['original_label'].unique()) if 'original_label' in tmp.columns else []
    non_ref = [l for l in labels if l != LABEL_REFERENCE]
    beh_mat = (np.column_stack([(tmp['original_label']==l).astype(float).values
                                for l in non_ref])
               if non_ref else np.zeros((len(tmp), 0)))
    y_z, ym, ys = _standardise(tmp[feat])
    pid_labels, pid_idx = np.unique(tmp['pid'].values, return_inverse=True)
    return {'df': tmp, 'y_z': y_z.astype(float),
            'group_bin': tmp['Group_bin'].values.astype(float),
            'age_c': tmp['age_c'].values.astype(float),
            'beh_mat': beh_mat, 'n_beh_dum': beh_mat.shape[1],
            'pid_idx': pid_idx, 'n_pids': len(pid_labels),
            'y_mean': ym, 'y_std': ys, 'n_obs': len(tmp)}

def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                    chains=BAYES_CHAINS, seed=42):
    """Main effect Bayesian model with random intercepts per child."""
    with pm.Model():
        alpha     = pm.Normal('alpha', 0, 1)
        b_group   = pm.Normal('b_group', 0, prior_sd)
        b_age     = pm.Normal('b_age', 0, 0.5)
        beh_contrib = 0.0
        if bd['n_beh_dum'] > 0:
            b_beh = pm.Normal('b_beh', 0, 0.5, shape=bd['n_beh_dum'])
            beh_contrib = pm.math.dot(bd['beh_mat'], b_beh)
        sigma_pid = pm.HalfNormal('sigma_pid', 1)
        sigma     = pm.HalfNormal('sigma', 1)
        alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu = (alpha + alpha_pid[bd['pid_idx']] + beh_contrib
              + b_group * bd['group_bin'] + b_age * bd['age_c'])
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    b_post = idata.posterior['b_group'].values.flatten()
    hdi    = az.hdi(idata, var_names=['b_group'], hdi_prob=0.94)['b_group'].values
    diag   = az.summary(idata, var_names=['b_group'], hdi_prob=0.94)
    rhat   = float(diag['r_hat'].values[0])
    ess    = float(diag['ess_bulk'].values[0])
    n_div  = int(idata.sample_stats['diverging'].values.sum())
    bf10   = _savage_dickey_bf(b_post, prior_sd)
    return idata, {
        'b_group_mean': float(b_post.mean()), 'b_group_sd': float(b_post.std()),
        'hdi94_lo': float(hdi[0]), 'hdi94_hi': float(hdi[1]),
        'p_pos': float((b_post > 0).mean()), 'bf10': bf10,
        'rhat': rhat, 'ess_bulk': ess, 'n_divergences': n_div,
        'converged': bool(rhat < 1.05 and ess > 400 and n_div == 0),
        'prior_sd': prior_sd,
    }

def _prior_predictive_check(bd, feat, prior_sd=0.5):
    """Check that prior covers the observed data range."""
    with pm.Model():
        b_group = pm.Normal('b_group', 0, prior_sd)
        b_age   = pm.Normal('b_age', 0, 0.5)
        sigma   = pm.HalfNormal('sigma', 1)
        alpha   = pm.Normal('alpha', 0, 1)
        mu = alpha + b_group * bd['group_bin'] + b_age * bd['age_c']
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        ppc = pm.sample_prior_predictive(samples=200, random_seed=42)
    prior_ys = ppc.prior_predictive['y_obs'].values.flatten()
    obs_range   = (bd['y_z'].min(), bd['y_z'].max())
    prior_range = (float(np.percentile(prior_ys, 1)),
                   float(np.percentile(prior_ys, 99)))
    return {'feature': feat, 'obs_min': obs_range[0], 'obs_max': obs_range[1],
            'prior_p1': prior_range[0], 'prior_p99': prior_range[1],
            'plausible': prior_range[0] <= obs_range[0] and prior_range[1] >= obs_range[1]}

def run_bayesian_suite(feat_df, primary_feats, perm_results, output_dir):
    """Full Bayesian suite: main effect + prior sensitivity + prior predictive."""
    hr("BAYESIAN HIERARCHICAL LMM")
    if not RUN_BAYESIAN or not _PYMC_OK:
        print("  Bayesian skipped"); return {}

    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    # Top features by permutation p-value
    bayes_feats = (perm_results.sort_values('p_raw').head(15)['feature'].tolist()
                   if len(perm_results) else primary_feats[:10])

    print(f"\nRunning Bayesian models on {len(bayes_feats)} features...")
    ppc_records = []; sensitivity_records = []; bayes_records = []
    bayes_main_results = {}

    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        # Prior predictive check
        try:
            ppc_rec = _prior_predictive_check(bd, feat)
            ppc_records.append(ppc_rec)
            if not ppc_rec['plausible']: print(f"  ⚠ PPC narrow for {feat}")
        except: pass
        # Prior sensitivity
        bf_vals = {}
        for psd in PRIOR_SDS:
            try:
                _, summ = _fit_bayes_main(bd, prior_sd=psd)
                summ['feature'] = feat; summ['prior_sd'] = psd
                sensitivity_records.append(summ); bf_vals[psd] = summ['bf10']
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")
        # Main result (prior_sd=0.5)
        if 0.5 in bf_vals:
            match = [r for r in sensitivity_records
                     if r['feature'] == feat and r['prior_sd'] == 0.5]
            if match:
                rec = match[-1].copy()
                bfs = [bf_vals[p] for p in PRIOR_SDS
                       if p in bf_vals and not np.isnan(bf_vals[p])]
                rec['bf_robust'] = bool(len(bfs) >= 2 and
                                        all((b>1) == (bfs[0]>1) for b in bfs))
                bayes_records.append(rec)
                flag = '✓' if rec.get('converged') else '⚠'
                bf_str = ' | '.join([f"sd={p}:BF={bf_vals.get(p,np.nan):.2f}"
                                     for p in PRIOR_SDS])
                print(f"  {feat:<45} {bf_str} {flag}")

    if ppc_records:
        pd.DataFrame(ppc_records).to_csv(
            os.path.join(output_dir, 'bayes_ppc.csv'), index=False)
    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            os.path.join(output_dir, 'bayes_sensitivity.csv'), index=False)
    if bayes_records:
        bayes_df = pd.DataFrame(bayes_records).sort_values('bf10', ascending=False)
        bayes_df.to_csv(os.path.join(output_dir, 'bayes_main.csv'), index=False)
        bayes_main_results = bayes_df
        print(f"\n  BF10>3:   {(bayes_df['bf10']>3).sum()}/{len(bayes_df)}")
        print(f"  BF10>10:  {(bayes_df['bf10']>10).sum()}/{len(bayes_df)}")
        print(f"  Robust:   {bayes_df['bf_robust'].sum()}/{len(bayes_df)}")
        bad = bayes_df[~bayes_df['converged']]
        if len(bad):
            print(f"  ⚠ {len(bad)} convergence issues:")
            for _, r in bad.iterrows():
                print(f"    {r['feature']}  rhat={r.get('rhat',np.nan):.3f}")

    return bayes_main_results


# ═══════════════════════════════════════════════════════════════════
# CLASSIFICATION — LOSO (LR + RF)
# ═══════════════════════════════════════════════════════════════════
def run_loso_child(cdf, feat_cols, clf_name='LR', n_perm=500, seed=42):
    """Leave-one-subject-out CV with permutation p-value. LR or RF."""
    df_ = cdf.copy(); df_['y'] = (df_['Group'] == 'ASD').astype(int)
    if df_['y'].sum() < 4 or (1-df_['y']).sum() < 4: return None
    usable = [f for f in feat_cols if f in df_.columns
              and df_[f].notna().mean() > 0.5]
    if len(usable) < 2: return None
    df_[usable] = df_[usable].fillna(df_[usable].median())

    if clf_name == 'LR':
        clf = LogisticRegression(max_iter=1000, C=0.1,
                                 class_weight='balanced', random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     random_state=seed, n_jobs=-1)
    pipe = Pipeline([('sc', StandardScaler()), ('clf', clf)])
    y_true, y_score = [], []
    for pid in df_['pid'].unique():
        test  = df_[df_['pid'] == pid]
        train = df_[df_['pid'] != pid]
        if len(train['y'].unique()) < 2: continue
        try:
            pipe.fit(train[usable].values, train['y'].values)
            y_score.extend(pipe.predict_proba(test[usable].values)[:,1].tolist())
            y_true.extend(test['y'].values.tolist())
        except: continue
    if len(set(y_true)) < 2: return None
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    perm = [roc_auc_score(rng.permuted(np.array(y_true)), y_score)
            for _ in range(n_perm)]
    p_perm = float((np.array(perm) >= auc).mean())
    cm = confusion_matrix(y_true, (np.array(y_score) >= 0.5).astype(int))
    print(f"  [{clf_name}] AUC={auc:.3f}  AP={ap:.3f}  p_perm={p_perm:.4f}  "
          f"n_feat={len(usable)}")
    return {'auc': auc, 'ap': ap, 'perm_p': p_perm, 'n_features': len(usable),
            'n_subjects': df_['pid'].nunique(), 'y_true': y_true, 'y_score': y_score,
            'perm_aucs': perm, 'confusion_matrix': cm, 'clf': clf_name}


# ═══════════════════════════════════════════════════════════════════
# FIGURES — FULL SUITE
# ═══════════════════════════════════════════════════════════════════
def run_figures(feat_df, child_df, stream_clip_dfs, stream_child_dfs,
                stream_lme_results, consensus_all, icc_df, cons_df,
                beh_mwu_dict, clf_results, bayes_results, sp_df,
                perm_all, feat_importance_df, output_dir, variant_name):
    """Generate the full figure suite."""
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    def _sl(f): return SHORT_LABELS.get(f, f)

    # ── Fig 1: Sample overview ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f'Sample Overview — {variant_name}', fontweight='bold')
    ax = axes[0]
    for i, sk in enumerate(AGE_STREAMS):
        cdf = stream_child_dfs[sk]
        for j, grp in enumerate(GROUPS):
            n = (cdf['Group'] == grp).sum()
            ax.bar(i*3+j, n, color=COLORS[grp], alpha=0.85, edgecolor='white')
            ax.text(i*3+j, n+0.1, str(n), ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks([1,4,7]); ax.set_xticklabels(list(AGE_STREAMS.keys()))
    ax.set_title('(a) Children per stream'); ax.set_ylabel('N')
    ax.legend(handles=[mpatches.Patch(color=COLORS[g], label=g) for g in GROUPS])
    ax = axes[1]
    lc = feat_df.groupby(['original_label', 'Group']).size().unstack(fill_value=0)
    x  = np.arange(len(lc)); w = 0.35
    for i, grp in enumerate(GROUPS):
        if grp in lc.columns:
            ax.bar(x+i*w, lc[grp], w, color=COLORS[grp], label=grp,
                   alpha=0.85, edgecolor='white')
    ax.set_xticks(x+w/2)
    ax.set_xticklabels([b.replace(' ', '\n') for b in lc.index], fontsize=8)
    ax.set_title('(b) Clips per label'); ax.set_ylabel('N'); ax.legend(fontsize=8)
    ax = axes[2]
    for grp in GROUPS:
        ax.hist(stream_child_dfs['full'][stream_child_dfs['full']['Group']==grp]['age_mo'],
                bins=12, alpha=0.6, color=COLORS[grp], label=grp, edgecolor='white')
    ax.axvspan(11,18,alpha=0.12,color=STREAM_COLORS['11-18mo'],label='11-18mo')
    ax.axvspan(32,38,alpha=0.12,color=STREAM_COLORS['32-38mo'],label='32-38mo')
    ax.set_xlabel('Age (months)'); ax.set_ylabel('N')
    ax.set_title('(c) Age distribution'); ax.legend(fontsize=7)
    plt.tight_layout(); save_fig(fig, 'fig01_sample_overview.png', figure_dir)

    # ── Fig 2: Effect size forest across streams ──────────────────
    all_stream_rows = []
    for sk, res in stream_lme_results.items():
        if len(res): r = res.copy(); r['stream'] = sk; all_stream_rows.append(r)
    if all_stream_rows:
        combined = pd.concat(all_stream_rows, ignore_index=True)
        top_feats = (combined.groupby('feature')['cohens_d']
                     .apply(lambda x: x.abs().max())
                     .sort_values(ascending=False).head(20).index.tolist())
        sub_c = combined[combined['feature'].isin(top_feats)].copy()
        fig, ax = plt.subplots(figsize=(13, max(6, len(top_feats)*0.6)))
        y_pos = {f: i for i, f in enumerate(top_feats)}
        offsets = {'full': 0.0, '11-18mo': 0.22, '32-38mo': -0.22}
        for sk in AGE_STREAMS:
            sub = sub_c[sub_c['stream'] == sk]
            for _, row in sub.iterrows():
                y = y_pos[row['feature']] + offsets.get(sk, 0)
                ax.scatter(row['cohens_d'], y, color=STREAM_COLORS[sk], s=60, zorder=5)
                if 'd_ci_lo' in row.index and not np.isnan(row.get('d_ci_lo', np.nan)):
                    ax.plot([row['d_ci_lo'], row['d_ci_hi']], [y,y],
                            color=STREAM_COLORS[sk], lw=2, alpha=0.7)
                if row.get('sig_fdr05'):
                    ax.scatter(row['cohens_d'], y, color=STREAM_COLORS[sk],
                               s=120, marker='*', zorder=6)
        ax.axvline(0, color='black', lw=0.8)
        for t, ls_ in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
            ax.axvline(t, color='gray', lw=0.6, ls=ls_, alpha=0.4)
            ax.axvline(-t, color='gray', lw=0.6, ls=ls_, alpha=0.4)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([_sl(f) for f in top_feats], fontsize=8)
        ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)")
        ax.set_title("Effect Sizes Across Streams (child level)\n"
                     "★=FDR sig  Line=95% bootstrap CI", fontweight='bold')
        ax.legend(handles=[mpatches.Patch(color=STREAM_COLORS[sk], label=sk)
                           for sk in AGE_STREAMS])
        plt.tight_layout(); save_fig(fig, 'fig02_effect_sizes_streams.png', figure_dir)

    # ── Fig 3: Violin plots by stream ─────────────────────────────
    DISP_FEATS = [f for f in ['wrist_amp_max', 'wrist_vel_max',
                               'bilateral_y_corr', 'bilateral_sym_index',
                               'bilateral_amp_diff', 'bilateral_phase_lag_sec',
                               'elbow_y_L_amplitude', 'elbow_y_R_amplitude']
                  if f in feat_df.columns]
    if DISP_FEATS:
        nf = len(DISP_FEATS); ns = len(AGE_STREAMS)
        fig, axes = plt.subplots(nf, ns, figsize=(5*ns, 4*nf))
        fig.suptitle('Feature Distributions — ASD vs Non-ASD (Columns=streams)',
                     fontweight='bold')
        for ci, sk in enumerate(AGE_STREAMS):
            sdf = stream_child_dfs[sk]
            r   = stream_lme_results.get(sk, pd.DataFrame())
            for ri, feat in enumerate(DISP_FEATS):
                ax = axes[ri][ci]
                dg = [sdf.loc[sdf['Group']==g, feat].dropna().values for g in GROUPS]
                if all(len(d)==0 for d in dg): ax.set_visible(False); continue
                parts = ax.violinplot([d if len(d)>1 else [0] for d in dg],
                                      positions=[0,1], showmedians=True, showextrema=False)
                for j, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
                parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
                for j, vals in enumerate(dg):
                    if len(vals):
                        ax.scatter(j+np.random.uniform(-0.07,0.07,len(vals)), vals,
                                   color=list(COLORS.values())[j], alpha=0.3, s=15, zorder=3)
                if len(r):
                    row_s = r[r['feature']==feat]
                    if len(row_s):
                        p  = row_s['p_raw'].values[0]; pf = row_s['p_fdr'].values[0]
                        d  = row_s['cohens_d'].values[0]
                        col = '#cc0000' if pf<0.05 else ('#ff8800' if p<0.05 else 'gray')
                        ax.text(0.5, 0.97, f'p={p:.3f}|FDR={pf:.3f}|d={d:.2f}',
                                transform=ax.transAxes, ha='center', va='top',
                                fontsize=7, color=col)
                        if all(len(d2)>0 for d2 in dg):
                            ymax = max(np.percentile(d2,95) for d2 in dg)
                            yr   = ymax - min(np.percentile(d2,5) for d2 in dg)
                            add_sig_bar(ax, 0, 1, ymax+yr*0.05, p, h=max(yr*0.04,1e-6))
                if ri == 0: ax.set_title(sk, fontsize=9, fontweight='bold',
                                         color=STREAM_COLORS[sk])
                if ci == 0: ax.set_ylabel(_sl(feat)[:20], fontsize=7)
                ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=7)
        plt.tight_layout(); save_fig(fig, 'fig03_violins_by_stream.png', figure_dir)

    # ── Fig 4: Consensus heatmap ──────────────────────────────────
    if len(consensus_all) > 0:
        p_cols    = [c for c in consensus_all.columns if c.startswith('p_')]
        heat_data = consensus_all.set_index('feature')[p_cols].head(20)
        heat_log  = -np.log10(heat_data.clip(lower=1e-5, upper=1.0).astype(float))
        fig, ax   = plt.subplots(figsize=(len(p_cols)*2+2, max(6, len(heat_data)*0.4)))
        im = ax.imshow(heat_log.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=4)
        ax.set_xticks(range(len(p_cols)))
        ax.set_xticklabels([c.replace('p_','') for c in p_cols], rotation=30, ha='right')
        ax.set_yticks(range(len(heat_data)))
        ax.set_yticklabels([_sl(f) for f in heat_data.index], fontsize=9)
        for i in range(heat_log.shape[0]):
            for j in range(heat_log.shape[1]):
                raw_p = heat_data.values[i,j]
                ax.text(j, i, f'{raw_p:.3f}{"*" if raw_p<0.05 else ""}',
                        ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, label='-log10(p)')
        ax.set_title('Consensus p-values across methods (top 20 features)',
                     fontweight='bold')
        plt.tight_layout(); save_fig(fig, 'fig04_consensus_heatmap.png', figure_dir)

    # ── Fig 5: Consistency gate ───────────────────────────────────
    if len(cons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(cons_df)*0.45)))
        cols_cg = [ASD_COLOR if v else NONASD_COLOR for v in cons_df['consistent']]
        ax.barh(cons_df['feature'].map(SHORT_LABELS).fillna(cons_df['feature']),
                cons_df['n_same_direction'] / cons_df['n_labels_tested'],
                color=cols_cg, edgecolor='white', height=0.6)
        ax.axvline(1.0, color='green', lw=1.5, ls='--', label='All consistent')
        ax.axvline(0.5, color='orange', lw=1, ls=':', label='50%')
        ax.set_xlim(0, 1.15); ax.set_xlabel('Fraction of labels with same direction')
        ax.set_title('Consistency Gate\nRed=failed (interpret carefully)', fontweight='bold')
        ax.legend(); plt.tight_layout()
        save_fig(fig, 'fig05_consistency_gate.png', figure_dir)

    # ── Fig 6: Label × behavior heatmap (Cohen's d) ───────────────
    if beh_mwu_dict and len(perm_all):
        sig_feats_plot = perm_all[perm_all['sig_raw05']]['feature'].tolist()[:15]
        lbl_all = pd.concat(beh_mwu_dict.values(), ignore_index=True)
        sub_b   = lbl_all[lbl_all['feature'].isin(sig_feats_plot)]
        if len(sub_b):
            lbls_avail = sorted(sub_b['label'].unique())
            pivot_d = sub_b.pivot_table(index='feature', columns='label',
                                        values='cohens_d')
            pivot_d.index = [_sl(f) for f in pivot_d.index]
            fig, ax = plt.subplots(
                figsize=(max(8, len(lbls_avail)*2.5), max(5, len(sig_feats_plot)*0.6)))
            im = ax.imshow(pivot_d.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(lbls_avail)))
            ax.set_xticklabels([b.replace(' ','\n') for b in lbls_avail], fontsize=9)
            ax.set_yticks(range(len(pivot_d)))
            ax.set_yticklabels(pivot_d.index, fontsize=8)
            for i in range(pivot_d.shape[0]):
                for j in range(pivot_d.shape[1]):
                    v = pivot_d.values[i,j]
                    if not np.isnan(v): ax.text(j, i, f'{v:.2f}', ha='center',
                                                va='center', fontsize=7)
            plt.colorbar(im, ax=ax, label="Cohen's d", fraction=0.03)
            ax.set_title("Cohen's d by Flapping Label\n"
                         "Consistent color = robust group effect", fontweight='bold')
            plt.tight_layout(); save_fig(fig, 'fig06_label_heatmap.png', figure_dir)

    # ── Fig 7: ICC bar ────────────────────────────────────────────
    if len(icc_df) > 0:
        top_icc = icc_df.head(20)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top_icc)*0.4)))
        cols_icc = ['#2ecc71' if v>0.1 else '#e74c3c' for v in top_icc['ICC']]
        ax.barh(top_icc['feature'].map(SHORT_LABELS).fillna(top_icc['feature']),
                top_icc['ICC'], color=cols_icc, edgecolor='white', height=0.65)
        ax.axvline(0.1, color='orange', lw=1.5, ls='--',
                   label='ICC=0.10 (clustering threshold)')
        ax.set_xlabel('ICC'); ax.set_title('Intraclass Correlation (within-child)\n'
                                           'Green=clustering matters', fontweight='bold')
        ax.legend(); plt.tight_layout()
        save_fig(fig, 'fig07_icc.png', figure_dir)

    # ── Fig 8: Bayesian forest ────────────────────────────────────
    if len(bayes_results) > 0:
        bdf = bayes_results.copy()
        bdf['label'] = bdf['feature'].map(SHORT_LABELS).fillna(bdf['feature'])
        bdf = bdf.sort_values('b_group_mean')
        fig, ax = plt.subplots(figsize=(13, max(5, len(bdf)*0.5)))
        for j, (_, row) in enumerate(bdf.iterrows()):
            col = ASD_COLOR if row['b_group_mean'] > 0 else NONASD_COLOR
            ax.plot([row['hdi94_lo'], row['hdi94_hi']], [j,j], color=col, lw=2.5, alpha=0.8)
            ax.scatter(row['b_group_mean'], j, color=col, s=70, zorder=5)
            ax.plot(row['hdi94_lo'], j, '|', color=col, markersize=8)
            ax.plot(row['hdi94_hi'], j, '|', color=col, markersize=8)
            bf = float(row['bf10']) if not np.isnan(float(row['bf10'])) else 0
            lbl = f"BF={bf:.1f}"
            if not row.get('converged', True): lbl += ' ⚠'
            if not row.get('bf_robust', True): lbl += ' [prior-sensitive]'
            ax.text(row['hdi94_hi']+0.01, j, lbl, va='center', fontsize=7)
        ax.axvline(0, color='black', lw=1.2, ls='--')
        ax.set_yticks(range(len(bdf))); ax.set_yticklabels(bdf['label'], fontsize=9)
        ax.set_xlabel('Posterior mean  |  94% HDI  (standardised)')
        ax.set_title('Bayesian Hierarchical LMM\n⚠=convergence  [prior-sensitive]=BF changed',
                     fontweight='bold')
        ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                           mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
        plt.tight_layout(); save_fig(fig, 'fig08_bayes_forest.png', figure_dir)

    # ── Fig 9: Prior sensitivity ──────────────────────────────────
    sens_path = os.path.join(output_dir, 'bayes_sensitivity.csv')
    if os.path.isfile(sens_path):
        sens = pd.read_csv(sens_path)
        if len(sens):
            feats_s = sens['feature'].unique()[:12]
            nrows   = int(np.ceil(len(feats_s)/3))
            fig, axes = plt.subplots(nrows, 3, figsize=(15, 4*nrows))
            fig.suptitle('Prior Sensitivity — BF10 across prior widths', fontweight='bold')
            axes = axes.flatten()
            for i, feat in enumerate(feats_s):
                ax  = axes[i]; sub = sens[sens['feature']==feat].sort_values('prior_sd')
                ax.plot(sub['prior_sd'], sub['bf10'], marker='o', color=ASD_COLOR, lw=2)
                ax.axhline(3, color='green', lw=1, ls='--', label='BF=3')
                ax.axhline(1, color='gray', lw=0.8, ls=':')
                ax.set_xlabel('Prior SD'); ax.set_ylabel('BF10')
                ax.set_title(_sl(feat)[:25], fontsize=9); ax.legend(fontsize=7)
            for j in range(len(feats_s), len(axes)): axes[j].set_visible(False)
            plt.tight_layout(); save_fig(fig, 'fig09_prior_sensitivity.png', figure_dir)

    # ── Fig 10: Classification ROC ────────────────────────────────
    if clf_results:
        keys  = list(clf_results.keys()); n = len(keys)
        ncols = min(n, 4); nrows = int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        if nrows*ncols == 1: axes = np.array([[axes]])
        elif nrows == 1: axes = axes.reshape(1,-1)
        fig.suptitle('Classification ROC — Child-Level LOSO', fontweight='bold')
        for i, key in enumerate(keys):
            r = clf_results[key]; ax = axes[i//ncols][i%ncols]
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_score'])
            ax.plot(fpr, tpr, color=ASD_COLOR, lw=2,
                    label=f"AUC={r['auc']:.3f}  AP={r['ap']:.3f}")
            ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.1, color=ASD_COLOR)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f"{key}\np_perm={r['perm_p']:.3f}", fontsize=8)
            ax.legend(fontsize=8)
            if r.get('perm_aucs'):
                axins = ax.inset_axes([0.55,0.05,0.4,0.28])
                axins.hist(r['perm_aucs'], bins=20, color='gray', alpha=0.7)
                axins.axvline(r['auc'], color=ASD_COLOR, lw=2)
                axins.set_title('Null', fontsize=6); axins.tick_params(labelsize=5)
        for i in range(len(keys), nrows*ncols):
            axes[i//ncols][i%ncols].set_visible(False)
        plt.tight_layout(); save_fig(fig, 'fig10_roc.png', figure_dir)

    # ── Fig 11: RF importances ────────────────────────────────────
    if len(feat_importance_df) > 0:
        top20 = feat_importance_df.head(20)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(top20['label'], top20['importance'], color=ASD_COLOR,
                edgecolor='white', height=0.65, alpha=0.85)
        ax.set_xlabel('Mean decrease in impurity')
        ax.set_title('RF Feature Importances (full stream, child level)', fontweight='bold')
        ax.axvline(top20['importance'].mean(), color='gray', lw=1, ls='--', label='Mean')
        ax.legend(); plt.tight_layout()
        save_fig(fig, 'fig11_rf_importances.png', figure_dir)

    # ── Fig 12: Developmental trajectories ───────────────────────
    TRAJ = [f for f in ['wrist_amp_max','wrist_vel_max','bilateral_sym_index',
                         'bilateral_y_corr','elbow_y_L_amplitude','elbow_y_R_amplitude']
            if f in feat_df.columns]
    if TRAJ:
        ncols = 3; nrows = int(np.ceil(len(TRAJ)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
        fig.suptitle('Developmental Trajectories (full stream)', fontweight='bold')
        axes = axes.flatten()
        for i, feat in enumerate(TRAJ):
            ax = axes[i]
            for grp in GROUPS:
                sub = feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
                ax.scatter(sub['age_mo'], sub[feat], color=COLORS[grp], alpha=0.25, s=12)
                if len(sub) >= 5:
                    m, b, r, p, _ = stats.linregress(sub['age_mo'], sub[feat])
                    xr = np.linspace(sub['age_mo'].min(), sub['age_mo'].max(), 100)
                    ax.plot(xr, m*xr+b, color=COLORS[grp], lw=2.5,
                            label=f'{grp} r={r:.2f} p={p:.3f}')
            ax.axvspan(11,18,alpha=0.08,color=STREAM_COLORS['11-18mo'])
            ax.axvspan(32,38,alpha=0.08,color=STREAM_COLORS['32-38mo'])
            ax.set_xlabel('Age (months)'); ax.set_ylabel(_sl(feat)[:20], fontsize=8)
            ax.set_title(_sl(feat), fontsize=9, fontweight='bold'); ax.legend(fontsize=7)
        for j in range(len(TRAJ), len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); save_fig(fig, 'fig12_trajectories.png', figure_dir)

    # ── Fig 13: Child-level boxplots ──────────────────────────────
    BOX = [f for f in ['wrist_amp_max','wrist_vel_max',
                        'bilateral_sym_index','bilateral_y_corr']
           if f in child_df.columns]
    if BOX:
        fig, axes = plt.subplots(len(BOX), len(AGE_STREAMS),
                                 figsize=(5*len(AGE_STREAMS), 4*len(BOX)), sharey='row')
        fig.suptitle('Child-Level Boxplots — Each Dot = 1 Child', fontweight='bold')
        for ci, sk in enumerate(AGE_STREAMS):
            cdf = stream_child_dfs[sk]
            for ri, feat in enumerate(BOX):
                ax = axes[ri][ci]
                for j, grp in enumerate(GROUPS):
                    vals = cdf[cdf['Group']==grp][feat].dropna().values
                    if not len(vals): continue
                    bp = ax.boxplot(vals, positions=[j], widths=0.45,
                                    patch_artist=True, showfliers=False,
                                    medianprops={'color':'black','linewidth':2})
                    bp['boxes'][0].set_facecolor(COLORS_LIGHT[grp])
                    bp['boxes'][0].set_edgecolor(COLORS[grp])
                    bp['boxes'][0].set_linewidth(1.5)
                    ax.scatter(j+np.random.uniform(-0.12,0.12,len(vals)), vals,
                               color=COLORS[grp], alpha=0.65, s=28, zorder=4)
                da = cdf[cdf['Group']=='ASD'][feat].dropna().values
                dn = cdf[cdf['Group']=='Non-ASD'][feat].dropna().values
                if len(da)>=3 and len(dn)>=3:
                    _, p = stats.mannwhitneyu(da, dn, alternative='two-sided')
                    ymax = cdf[feat].dropna().quantile(0.97)
                    add_sig_bar(ax, 0, 1, ymax, p, h=abs(ymax)*0.04+1e-6)
                if ri == 0: ax.set_title(sk, fontsize=9, fontweight='bold',
                                         color=STREAM_COLORS[sk])
                if ci == 0: ax.set_ylabel(_sl(feat)[:18], fontsize=7)
                ax.set_xticks([0,1]); ax.set_xticklabels(['ASD','NASD'], fontsize=8)
                ax.text(0.5,-0.18,f'n={len(da)}/{len(dn)}',
                        transform=ax.transAxes, ha='center', fontsize=7, color='gray')
        plt.tight_layout(); save_fig(fig, 'fig13_child_boxplots.png', figure_dir)

    # ── Fig 14: Spearman age correlations ─────────────────────────
    if sp_df is not None and len(sp_df):
        pivot_r = sp_df.pivot_table(index='feature', columns='Group', values='spearman_r')
        pivot_p = sp_df.pivot_table(index='feature', columns='Group', values='p_raw')
        top_feats_sp = pivot_r.abs().max(axis=1).sort_values(ascending=False).head(20).index
        pr = pivot_r.loc[top_feats_sp]; pp = pivot_p.loc[top_feats_sp]
        fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(pr)*0.4)))
        fig.suptitle('Spearman Correlation with Age', fontweight='bold')
        for ci, grp in enumerate(GROUPS):
            if grp not in pr.columns: continue
            ax = axes[ci]
            cols_sp = [ASD_COLOR if v > 0 else NONASD_COLOR for v in pr[grp]]
            ax.barh([_sl(f) for f in pr.index], pr[grp],
                    color=cols_sp, edgecolor='white', height=0.65, alpha=0.85)
            # mark significant
            for j, feat in enumerate(pr.index):
                if pp.loc[feat, grp] < 0.05:
                    ax.text(pr.loc[feat, grp] + 0.01, j, '*', va='center',
                            fontsize=12, color='gold')
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel('Spearman r'); ax.set_title(f'{grp}', fontweight='bold')
        plt.tight_layout(); save_fig(fig, 'fig14_spearman_age.png', figure_dir)

    # ── Fig 15: Three-way method comparison ───────────────────────
    # MW vs LME vs Bayesian
    if (len(consensus_all) and len(bayes_results) and
            'p_LME_KR' in consensus_all.columns and 'p_PseudobulkMW' in consensus_all.columns):
        cmp = consensus_all.merge(
            bayes_results[['feature','b_group_mean','bf10']],
            on='feature', how='inner')
        if len(cmp) >= 3:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Three-Way Method Comparison', fontweight='bold')
            ax = axes[0]
            ax.scatter(-np.log10(cmp['p_PseudobulkMW']+1e-10),
                       -np.log10(cmp.get('p_LME_KR', cmp.get('p_LME_noKR',
                                                               pd.Series(np.nan)))+1e-10),
                       c=cmp['bf10'].clip(upper=20), cmap='RdYlGn', s=50,
                       alpha=0.8, edgecolors='gray', lw=0.5)
            ax.set_xlabel('-log10(p) Mann-Whitney'); ax.set_ylabel('-log10(p) LME')
            ax.set_title('MW vs LME (color=BF10)')
            ax.axhline(-np.log10(0.05), color='gray', lw=0.7, ls='--')
            ax.axvline(-np.log10(0.05), color='gray', lw=0.7, ls='--')
            ax = axes[1]
            ax.scatter(cmp.get('cohens_d', 0),
                       np.log10(cmp['bf10'].clip(lower=0.01)),
                       c=[ASD_COLOR if d>0 else NONASD_COLOR
                          for d in cmp.get('cohens_d', [0]*len(cmp))],
                       s=50, alpha=0.8)
            ax.axhline(np.log10(3),  color='orange', lw=1, ls='--', label='BF=3')
            ax.axhline(np.log10(10), color='green',  lw=1, ls='-',  label='BF=10')
            ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel("Cohen's d"); ax.set_ylabel('log10(BF10)')
            ax.set_title("Cohen's d vs Bayes Factor"); ax.legend(fontsize=8)
            plt.tight_layout(); save_fig(fig, 'fig15_three_way_comparison.png', figure_dir)

    print(f"  All figures saved to {figure_dir}")


# ═══════════════════════════════════════════════════════════════════
# MAIN PER-VARIANT RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_variant(variant, df_main, pid_info, video_to_hrnet):
    name            = variant['name']
    FLAPPING_LABELS = variant['flapping_labels']
    OUTPUT_DIR      = variant['output_dir']
    FIGURE_DIR      = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    hr(f"VARIANT: {name}  |  labels: {FLAPPING_LABELS}")

    # ── PART 0: LOAD DATA ─────────────────────────────────────────
    hr(f"PART 0: LOAD DATA [{name}]")
    rmm = pd.read_csv(RMM_CSV)
    rmm['pid'] = rmm['csv_bids_processed'].apply(extract_pid)
    rmm = rmm.merge(pid_info, on='pid', how='left')
    rmm = rmm[rmm['Group'].isin(['ASD','Non-ASD'])].copy()
    rmm['label_lower'] = rmm['matched_label'].str.strip().str.lower()
    flap_rmm = rmm[rmm['label_lower'].isin(FLAPPING_LABELS)].copy()
    flap_rmm['hrnet_path'] = flap_rmm['csv_bids_processed'].map(video_to_hrnet)
    flap_rmm = flap_rmm[
        flap_rmm['hrnet_path'].apply(lambda p: isinstance(p,str) and os.path.isfile(p))
    ].copy()
    flap_rmm['age_band']      = flap_rmm['age_mo'].apply(assign_age_band)
    flap_rmm['original_label']= flap_rmm['matched_label'].str.strip().str.lower()

    print(f"Flapping clips with pose file: {len(flap_rmm)}")
    print(flap_rmm.groupby(['original_label','Group']).size()
          .reset_index(name='n').to_string(index=False))

    for sk, bounds in AGE_STREAMS.items():
        if bounds is None: flap_rmm[f'stream_{sk}'] = True
        else:
            lo, hi = bounds
            flap_rmm[f'stream_{sk}'] = ((flap_rmm['age_mo']>=lo) &
                                        (flap_rmm['age_mo']<=hi))

    # ── PART 1: FEATURE EXTRACTION ────────────────────────────────
    hr(f"PART 1: FEATURE EXTRACTION [{name}]")
    all_features = []
    n_ok = n_fail_pose = n_fail_ts = n_fail_kp = 0

    for _, row in flap_rmm.iterrows():
        ts_str = str(row.get('matched_ts', ''))
        segs   = parse_timestamps(ts_str)
        if not segs: n_fail_ts += 1; continue
        try:
            with open(row['hrnet_path'], 'r') as f:
                pose_data = json.load(f)
            pose_frames = pose_data.get('frames', {})
            ann_fps     = float(pose_data.get('ann_fps', FPS))
            if ann_fps != FPS: segs = parse_timestamps(ts_str, fps=ann_fps)
        except: n_fail_pose += 1; continue
        for seg_start, seg_end in segs:
            frame_idx = list(range(seg_start, seg_end+1))
            if len(frame_idx) < 5: continue
            feats = extract_flapping_features(pose_frames, frame_idx, ann_fps)
            if feats is None: n_fail_kp += 1; continue
            n_ok += 1
            feats.update({
                'pid':            row['pid'],
                'Group':          row['Group'],
                'age_mo':         row['age_mo'],
                'age_band':       row['age_band'],
                'original_label': row['original_label'],
                'clip':           row.get('clip_filename',''),
            })
            all_features.append(feats)

    print(f"Extraction: ok={n_ok}  fail_ts={n_fail_ts}  "
          f"fail_pose={n_fail_pose}  fail_kp={n_fail_kp}")
    if n_ok == 0:
        print(f"ERROR [{name}]: No features extracted. Skipping."); return

    feat_df = pd.DataFrame(all_features)
    feat_df.to_csv(os.path.join(OUTPUT_DIR, 'clip_level_features.csv'), index=False)

    META_COLS = {'pid','Group','age_mo','age_band','original_label','clip',
                 'n_valid_frames','n_total_frames','pct_valid','duration_sec',
                 'mean_torso_length','mean_conf_L','mean_conf_R'}
    FEAT_COLS = [c for c in feat_df.columns if c not in META_COLS]

    PRIMARY_FEATS = [f for f in FEAT_COLS if any(x in f for x in [
        'wrist_amp_max','wrist_amp_mean','wrist_vel_max','wrist_vel_mean',
        'wrist_y_L_amplitude','wrist_y_R_amplitude',
        'wrist_y_L_vel_mean','wrist_y_R_vel_mean',
        'wrist_y_L_acc_mean','wrist_y_R_acc_mean',
        'wrist_y_L_dom_freq','wrist_y_R_dom_freq',
        'wrist_y_L_spectral_entropy','wrist_y_R_spectral_entropy',
        'bilateral_amp_diff','bilateral_y_corr',
        'bilateral_sym_index','bilateral_phase_lag_sec',
        'elbow_y_L_amplitude','elbow_y_R_amplitude',
    ])]
    PRIMARY_FEATS = [f for f in PRIMARY_FEATS if f in feat_df.columns]

    def make_child_df(clip_df):
        fc  = [f for f in PRIMARY_FEATS if f in clip_df.columns]
        agg = clip_df.groupby(['pid','Group'])[fc].mean().reset_index()
        agg['n_clips']  = clip_df.groupby(['pid','Group']).size().values
        agg['age_mo']   = clip_df.groupby(['pid','Group'])['age_mo'].first().values
        agg['age_band'] = clip_df.groupby(['pid','Group'])['age_band'].first().values
        agg['original_label'] = (clip_df.groupby(['pid','Group'])['original_label']
                                 .agg(lambda x: x.mode()[0]).values)
        return agg

    stream_clip_dfs  = {}; stream_child_dfs = {}
    for sk in AGE_STREAMS:
        sdf = stream_filter(feat_df, sk)
        stream_clip_dfs[sk]  = sdf
        stream_child_dfs[sk] = make_child_df(sdf)
        sdf.to_csv(os.path.join(OUTPUT_DIR, f'clip_features_{sk}.csv'), index=False)
        stream_child_dfs[sk].to_csv(os.path.join(OUTPUT_DIR, f'child_features_{sk}.csv'), index=False)
        cdf = stream_child_dfs[sk]
        print(f"  {sk}: {len(sdf)} clips | {len(cdf)} children | "
              f"ASD={cdf[cdf['Group']=='ASD']['pid'].nunique()} "
              f"Non-ASD={cdf[cdf['Group']=='Non-ASD']['pid'].nunique()}")

    cdf_full = stream_child_dfs['full']
    sdf_full = stream_clip_dfs['full']

    # ── PART 2: FULL STATISTICAL BATTERY ─────────────────────────
    hr(f"PART 2: FULL STATISTICAL BATTERY [{name}]")

    # Step 0: ICC
    print("\n--- Step 0: ICC ---")
    icc_df = compute_icc(sdf_full, PRIMARY_FEATS)
    icc_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_icc.csv'), index=False)
    print(icc_df.head(8).to_string(index=False))
    print(f"  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)} features")

    # Step 1: LME + KR
    print("\n--- Step 1: LME + Kenward-Roger ---")
    lme_all = run_lme_kr(sdf_full, PRIMARY_FEATS, 'full')
    if len(lme_all):
        lme_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_kr_full.csv'), index=False)
        print(f"  sig_raw={lme_all['sig_raw05'].sum()}  FDR={lme_all['sig_fdr05'].sum()}")

    # Step 2: CR2
    print("\n--- Step 2: CR2 ---")
    cr2_all = run_cr2(sdf_full, PRIMARY_FEATS, 'full')
    if len(cr2_all):
        cr2_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_cr2_full.csv'), index=False)
        print(f"  sig_raw={cr2_all['sig_raw05'].sum()}  FDR={cr2_all['sig_fdr05'].sum()}")

    # Step 3: GEE
    print("\n--- Step 3: GEE ---")
    gee_all = run_gee(sdf_full, PRIMARY_FEATS, 'full')
    if len(gee_all):
        gee_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_gee_full.csv'), index=False)
        print(f"  sig_raw={gee_all['sig_raw05'].sum()}  FDR={gee_all['sig_fdr05'].sum()}")

    # Step 4: Child-level permutation
    print("\n--- Step 4: Child permutation ---")
    perm_all = run_child_permutation(cdf_full, PRIMARY_FEATS, n_perm=5000, subset_label='full')
    if len(perm_all):
        perm_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_child_perm_full.csv'), index=False)
        print(f"  sig_raw={perm_all['sig_raw05'].sum()}  FDR={perm_all['sig_fdr05'].sum()}")

    # Step 5: Wild cluster bootstrap
    print("\n--- Step 5: Wild cluster bootstrap ---")
    boot_all = run_wild_bootstrap(cdf_full, PRIMARY_FEATS, n_boot=5000, subset_label='full')
    if len(boot_all):
        boot_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_wild_boot_full.csv'), index=False)
        print(f"  sig_raw={boot_all['sig_raw05'].sum()}  FDR={boot_all['sig_fdr05'].sum()}")

    # Step 6: MWU
    print("\n--- Step 6: Pseudo-bulk MWU ---")
    mw_all = run_mwu(cdf_full, PRIMARY_FEATS, 'full')
    if len(mw_all):
        mw_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_mwu_full.csv'), index=False)
        print(f"  sig_raw={mw_all['sig_raw05'].sum()}  FDR={mw_all['sig_fdr05'].sum()}")

    # Step 7: Label × Group interaction
    print("\n--- Step 7: Label × Group interaction ---")
    lgi_all = run_label_group_interaction(sdf_full, PRIMARY_FEATS, 'full')
    if len(lgi_all):
        lgi_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_label_group_interaction.csv'), index=False)
        n_sig_lgi = lgi_all['sig_raw05'].sum()
        if n_sig_lgi > 0:
            print(f"  ⚠ {n_sig_lgi} Label×Group interactions — some effects differ by label:")
            for _, r in lgi_all[lgi_all['sig_raw05']].head(5).iterrows():
                print(f"    {r['feature']:<40} {r['interaction_term']}  p={r['p_raw']:.4f}")
        else:
            print(f"  No significant interactions ({len(lgi_all)} tested)")

    # Step 8: Consensus
    print("\n--- Step 8: Consensus ---")
    all_results = {'LME_KR': lme_all, 'CR2': cr2_all, 'GEE': gee_all,
                   'ChildPerm': perm_all, 'WildBoot': boot_all, 'PseudobulkMW': mw_all}
    consensus_all = make_consensus(all_results, PRIMARY_FEATS)
    consensus_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_consensus_all.csv'), index=False)
    print(f"  Top features by n_methods_sig:")
    for _, r in consensus_all.head(5).iterrows():
        print(f"    {r['feature']:<45} n_sig={r['n_methods_sig']}")

    # Step 9: Consistency gate
    print("\n--- Step 9: Consistency gate ---")
    sig_feats = list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else []
    cons_df, consistent_feats, label_mwu_dict = run_consistency_gate(
        feat_df, PRIMARY_FEATS, sig_feats)
    if len(cons_df):
        cons_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_consistency_gate.csv'), index=False)
        print(f"  {len(consistent_feats)}/{len(sig_feats)} features passed")
        for f in consistent_feats: print(f"    ✓ {f}")

    # Step 10: Spearman age correlations
    print("\n--- Step 10: Spearman age correlations ---")
    sp_df = run_spearman_age(sdf_full, PRIMARY_FEATS)
    if len(sp_df):
        sp_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_spearman_age.csv'), index=False)
        print(f"  Significant: {sp_df['sig_p05'].sum()}/{len(sp_df)}")

    # Step 11: Age-stratified + growth-curve LME
    print("\n--- Step 11: LME suite (age-strat, growth-curve, within-ASD) ---")
    lme_suite = run_lme_suite(sdf_full, PRIMARY_FEATS, 'full')
    for key, res in lme_suite.items():
        if len(res):
            res.to_csv(os.path.join(OUTPUT_DIR, f'stats_lme_{key}.csv'), index=False)
            print(f"  {key}: {len(res)} features")

    # Per-stream LME
    print("\n--- Stream-stratified LME ---")
    stream_lme_results = {}
    for sk in AGE_STREAMS:
        sdf = stream_clip_dfs[sk]; cdf = stream_child_dfs[sk]
        n_asd  = sdf[sdf['Group']=='ASD']['pid'].nunique()
        n_nasd = sdf[sdf['Group']=='Non-ASD']['pid'].nunique()
        print(f"\n  [{sk}] ASD={n_asd} Non-ASD={n_nasd}")
        if n_asd < 3 or n_nasd < 3: print("  → skip"); continue
        slme  = run_lme_kr(sdf, PRIMARY_FEATS, sk)
        sperm = run_child_permutation(cdf, PRIMARY_FEATS, n_perm=2000, subset_label=sk)
        sboot = run_wild_bootstrap(cdf, PRIMARY_FEATS, n_boot=2000, subset_label=sk)
        smw   = run_mwu(cdf, PRIMARY_FEATS, sk)
        sd    = {k:v for k,v in {'LME_KR':slme,'ChildPerm':sperm,
                                  'WildBoot':sboot,'MWU':smw}.items() if len(v)>0}
        if not sd: continue
        scons = make_consensus(sd, PRIMARY_FEATS); scons['stream'] = sk
        scons.to_csv(os.path.join(OUTPUT_DIR, f'stats_{sk.replace("-","_")}_consensus.csv'),
                     index=False)
        stream_lme_results[sk] = slme

    # Per-label MWU
    print("\n--- Label-stratified MWU ---")
    label_mwu_results = {}
    for lbl in sorted(feat_df['original_label'].dropna().unique()):
        sub = sdf_full[sdf_full['original_label']==lbl]
        asd_n  = sub[sub['Group']=='ASD']['pid'].nunique()
        nasd_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        print(f"  {lbl}: ASD={asd_n} Non-ASD={nasd_n}", end='')
        if asd_n >= 3 and nasd_n >= 3:
            r = run_mwu(sub, [f for f in PRIMARY_FEATS if f in sub.columns],
                        subset_label=lbl)
            label_mwu_results[lbl] = r; r['label'] = lbl
            print(f" → sig_raw:{r['sig_raw05'].sum()}  FDR:{r['sig_fdr05'].sum()}")
        else: print(" → too few")
    if label_mwu_results:
        pd.concat(label_mwu_results.values(), ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, 'stats_label_stratified_mwu.csv'), index=False)

    # ── PART 3: BAYESIAN ──────────────────────────────────────────
    hr(f"PART 3: BAYESIAN [{name}]")
    bayes_results = run_bayesian_suite(feat_df, PRIMARY_FEATS,
                                       perm_all, OUTPUT_DIR)
    if isinstance(bayes_results, dict): bayes_results = pd.DataFrame()

    # ── PART 4: CLASSIFICATION ────────────────────────────────────
    hr(f"PART 4: CLASSIFICATION [{name}]")
    clf_results = {}
    for sk in AGE_STREAMS:
        cdf = stream_child_dfs[sk]
        asd_n  = (cdf['Group']=='ASD').sum()
        nasd_n = (cdf['Group']=='Non-ASD').sum()
        print(f"\n--- Stream {sk} (ASD={asd_n} Non-ASD={nasd_n}) ---")
        if asd_n < 4 or nasd_n < 4: print("  Skipped"); continue
        for cname in ['LR', 'RF']:
            r = run_loso_child(cdf, PRIMARY_FEATS, clf_name=cname)
            if r: clf_results[f'{sk}_{cname}'] = r

    for lbl in sorted(FLAPPING_LABELS):
        sub = stream_child_dfs['full'][
            stream_child_dfs['full']['original_label']==lbl]
        asd_n  = (sub['Group']=='ASD').sum()
        nasd_n = (sub['Group']=='Non-ASD').sum()
        print(f"\n--- Label: {lbl} (ASD={asd_n} Non-ASD={nasd_n}) ---")
        if asd_n >= 4 and nasd_n >= 4:
            r = run_loso_child(sub, [f for f in PRIMARY_FEATS if f in sub.columns])
            if r: clf_results[f'lbl_{lbl.replace(" ","_")}_LR'] = r
        else: print("  Skipped")

    if clf_results:
        pd.DataFrame([{'subset':k,'clf':v.get('clf',''),'auc':v['auc'],
                       'ap':v['ap'],'perm_p':v['perm_p'],
                       'n_features':v['n_features'],'n_subjects':v['n_subjects']}
                      for k,v in clf_results.items()]).to_csv(
            os.path.join(OUTPUT_DIR, 'classification_summary.csv'), index=False)

    # RF feature importances
    feat_importance_df = pd.DataFrame()
    try:
        tmp = stream_child_dfs['full'].copy()
        tmp['y'] = (tmp['Group']=='ASD').astype(int)
        usable = [f for f in PRIMARY_FEATS if f in tmp.columns
                  and tmp[f].notna().mean() > 0.5]
        tmp[usable] = tmp[usable].fillna(tmp[usable].median())
        sc = StandardScaler(); X = sc.fit_transform(tmp[usable].values)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X, tmp['y'].values)
        feat_importance_df = pd.DataFrame({
            'feature': usable, 'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        feat_importance_df['label'] = (feat_importance_df['feature']
                                       .map(SHORT_LABELS).fillna(feat_importance_df['feature']))
        feat_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importances.csv'),
                                   index=False)
        print("\nTop 10 RF importances:")
        for _, r in feat_importance_df.head(10).iterrows():
            print(f"  {r['feature']:<45} {r['importance']:.4f}")
    except Exception as e:
        print(f"RF importance failed: {e}")

    # ── PART 5: FIGURES ───────────────────────────────────────────
    hr(f"PART 5: FIGURES [{name}]")
    run_figures(feat_df, cdf_full, stream_clip_dfs, stream_child_dfs,
                stream_lme_results, consensus_all, icc_df, cons_df,
                label_mwu_dict, clf_results, bayes_results, sp_df,
                perm_all, feat_importance_df, OUTPUT_DIR, name)

    # ── PART 6: SUMMARY ───────────────────────────────────────────
    hr(f"SUMMARY [{name}]")
    print(f"\nOutput : {OUTPUT_DIR}")
    print(f"Figures: {os.path.join(OUTPUT_DIR, 'figures')}\n")
    print("--- CSVs ---")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith('.csv'):
            try:
                tmp = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
                print(f"  {fname:<65} {tmp.shape[0]:>5}r × {tmp.shape[1]:>3}c")
            except: print(f"  {fname}")
    print("\n--- KEY RESULTS ---")
    for nm, res in [('LME_KR', lme_all), ('ChildPerm', perm_all),
                    ('WildBoot', boot_all), ('MWU', mw_all)]:
        if len(res): print(f"  {nm}: sig_raw={res['sig_raw05'].sum()}  "
                           f"FDR={res['sig_fdr05'].sum()}")
    print(f"\nConsistency: {len(consistent_feats)}/{len(sig_feats)} features passed")
    for f in consistent_feats: print(f"  ✓ {f}")
    if clf_results:
        print("\nClassification (child-level LOSO):")
        for k, v in clf_results.items():
            print(f"  {k:<50} AUC={v['auc']:.3f}  p_perm={v['perm_p']:.4f}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
hr("LOADING SHARED DATA")

df_main = pd.read_csv(MAIN_CSV)
df_main['pid']    = df_main['video_path'].apply(extract_pid)
df_main['age_mo'] = df_main['Age'] * 12
df_main = df_main[df_main['pid'].notna() &
                  df_main['Group'].isin(['ASD','Non-ASD'])].copy()

video_to_hrnet = dict(zip(df_main['video_path'], df_main['hrnet_full_path']))
pid_info = (df_main.dropna(subset=['pid','Group'])
            .groupby('pid')
            .agg(Group=('Group','first'), age_mo=('age_mo','mean'))
            .reset_index())

print(f"Main CSV: {len(df_main)} rows, {df_main['pid'].nunique()} children")
print(f"  ASD:     {df_main[df_main['Group']=='ASD']['pid'].nunique()}")
print(f"  Non-ASD: {df_main[df_main['Group']=='Non-ASD']['pid'].nunique()}")

for variant in VARIANTS:
    run_variant(variant, df_main, pid_info, video_to_hrnet)

hr("ALL VARIANTS COMPLETE")