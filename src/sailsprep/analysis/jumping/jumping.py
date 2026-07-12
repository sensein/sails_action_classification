#!/usr/bin/env python3
"""
JUMPING KINEMATIC ANALYSIS
"""

import json
import os
import re
import traceback
import warnings
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, welch
from scipy.stats import gaussian_kde
from scipy.stats import norm as spnorm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from sailsprep.analysis.common.banners import hr_v1 as hr
from sailsprep.analysis.common.parsing import extract_pid, parse_timestamps_v1 as parse_timestamps
from sailsprep.analysis.common.keypoints import get_kp, assign_age_band
from sailsprep.analysis.common.effect_size import cohen_d_v3 as cohen_d
from sailsprep.analysis.common.significance import fdr_annotate_v2 as fdr_annotate
from sailsprep.analysis.common.signal_processing import butter_lp_v1 as butter_lp
from sailsprep.analysis.common.bayes import _savage_dickey_bf, _standardise
from sailsprep.analysis.common.misc import run_spearman_age

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Optional heavy deps ──────────────────────────────────────────────
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
# ═══════════════════════════════════════════════════════════════════════
# SHARED CONFIG
# ═══════════════════════════════════════════════════════════════════════
MAIN_CSV = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
RMM_CSV = ( 
    "/home/aparnabg/orcd/scratch/all_project_files/phase_2_analyais"  
    "/clip_to_csv_matching.csv"
)
BASE_DIR = (
    "/orcd/data/satra/002/projects/SAILS/action_outputs_features"
    "/analysis/jumping/v3"
)


FPS      = 15.0
MIN_CONF = 0.3

# Bayesian config
BAYES_DRAWS  = 2000
BAYES_TUNE   = 1000
BAYES_CHAINS = 4
RUN_BAYESIAN = True
PRIOR_SDS    = [0.3, 0.5, 1.0]   # prior sensitivity widths

KP = {
    'left_shoulder':  'kp_005',
    'right_shoulder': 'kp_006',
    'left_elbow':     'kp_007',
    'right_elbow':    'kp_008',
    'left_wrist':     'kp_009',
    'right_wrist':    'kp_010',
    'left_hip':       'kp_011',
    'right_hip':      'kp_012',
    'left_knee':      'kp_013',
    'right_knee':     'kp_014',
    'left_ankle':     'kp_015',
    'right_ankle':    'kp_016',
}

AGE_BANDS = {
    '11-18mo': (11, 18),
    '19-31mo': (19, 31),
    '32-38mo': (32, 38),
}
STAT_BANDS = ['11-18mo', '32-38mo']

ASD_COLOR    = '#E05C5C'
NONASD_COLOR = '#5B8DB8'
ASD_LIGHT    = '#F2AEAE'
NONASD_LIGHT = '#A8C8E8'
COLORS       = {'ASD': ASD_COLOR, 'Non-ASD': NONASD_COLOR}
COLORS_LIGHT = {'ASD': ASD_LIGHT, 'Non-ASD': NONASD_LIGHT}
BAND_COLORS  = {'11-18mo': '#7B5EA7', '19-31mo': '#4A9B6F', '32-38mo': '#D47C2A'}
GROUPS       = ['ASD', 'Non-ASD']

LABEL_REFERENCE = 'jumping'   # reference level for label dummies

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

ALL_JUMP_LABELS = {'jumping', 'bed jumping', 'bouncing', 'knee jumping'}

VARIANTS = [
    {'name': 'jumping_all',  'jump_labels': {'jumping','bed jumping','bouncing','knee jumping'},
     'output_dir': os.path.join(BASE_DIR, 'jumping_all')},
    {'name': 'jumping_core', 'jump_labels': {'jumping','knee jumping'},
     'output_dir': os.path.join(BASE_DIR, 'jumping_core')},
    {'name': 'jumping_3',    'jump_labels': {'jumping','bed jumping','bouncing'},
     'output_dir': os.path.join(BASE_DIR, 'jumping_3')},
    {'name': 'jumping',      'jump_labels': {'jumping'},
     'output_dir': os.path.join(BASE_DIR, 'jumping_only')},
    {'name': 'bouncing',     'jump_labels': {'bouncing'},
     'output_dir': os.path.join(BASE_DIR, 'bouncing_only')},
]

SHORT_LABELS = {
    'mean_hip_y_amplitude':          'Hip Amp (mean)',
    'vertical_excursion_hip_max':    'Hip Excursion Max',
    'mean_hip_dom_freq':             'Hip Dom Freq',
    'mean_hip_band_power_1_4hz':     'Hip Power 1-4Hz',
    'mean_hip_spectral_entropy':     'Hip Spectral Entropy',
    'mean_hip_vel_mean':             'Hip Vel Mean',
    'mean_hip_acc_mean':             'Hip Acc Mean',
    'bilateral_hip_y_corr':          'Bilat Hip Corr',
    'bilateral_hip_sym_index':       'Bilat Hip Symmetry',
    'bilateral_hip_phase_lag':       'Bilat Hip Phase Lag',
    'bilateral_knee_y_corr':         'Bilat Knee Corr',
    'bilateral_knee_sym_index':      'Bilat Knee Symmetry',
    'bilateral_ankle_y_corr':        'Bilat Ankle Corr',
    'bilateral_ankle_sym_index':     'Bilat Ankle Symmetry',
    'ankle_y_L_vel_mean':            'Ankle L Vel',
    'ankle_y_L_mean':                'Ankle L Mean',
    'ankle_y_L_acc_mean':            'Ankle L Acc',
    'shoulder_y_R_dom_freq':         'Shoulder R Dom Freq',
    'shoulder_y_R_band_power_1_4hz': 'Shoulder R 1-4Hz',
    'hip_ankle_amp_ratio':           'Hip/Ankle Ratio',
    'hip_y_L_dom_freq':              'Hip L Dom Freq',
    'hip_y_R_dom_freq':              'Hip R Dom Freq',
}


# ═══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════






def torso_length(fd):
    ls = get_kp(fd, KP['left_shoulder'],  min_conf=0.1)
    rs = get_kp(fd, KP['right_shoulder'], min_conf=0.1)
    lh = get_kp(fd, KP['left_hip'],       min_conf=0.1)
    rh = get_kp(fd, KP['right_hip'],      min_conf=0.1)
    if not all([ls, rs, lh, rh]): return None
    sx = (ls['x']+rs['x'])/2;  sy = (ls['y']+rs['y'])/2
    hx = (lh['x']+rh['x'])/2;  hy = (lh['y']+rh['y'])/2
    d  = np.sqrt((sx-hx)**2 + (sy-hy)**2)
    return d if d > 5 else None

def hip_width(fd):
    lh = get_kp(fd, KP['left_hip'],  min_conf=0.1)
    rh = get_kp(fd, KP['right_hip'], min_conf=0.1)
    if not all([lh, rh]): return None
    d = np.sqrt((lh['x']-rh['x'])**2 + (lh['y']-rh['y'])**2)
    return d if d > 5 else None

def get_scale(fd):
    tl = torso_length(fd)
    if tl: return tl
    hw = hip_width(fd)
    if hw: return hw
    return None

def spectral_features(arr, fps):
    if len(arr) < 16: return np.nan, np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq  = freqs[np.argmax(psd)]
        psd_n     = psd / (psd.sum() + 1e-12)
        entropy   = -np.sum(psd_n[psd_n>0] * np.log2(psd_n[psd_n>0]))
        band_mask = (freqs >= 1.0) & (freqs <= 4.0)
        band_pwr  = psd[band_mask].sum() / (psd.sum() + 1e-12)
        return dom_freq, entropy, band_pwr
    except:
        return np.nan, np.nan, np.nan


def bootstrap_ci_d(a, b, n_boot=500, seed=42):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    boot = [cohen_d(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def cles(a, b):
    a, b = np.array(a), np.array(b)
    return sum(1 for ai in a for bi in b if ai > bi) / (len(a)*len(b))


def add_sig_bar(ax, x1, x2, y, p, h=0.02):
    label = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col   = '#cc0000' if p<0.001 else ('#e06600' if p<0.01 else
            ('#888800' if p<0.05 else '#888888'))
    ax.plot([x1,x1,x2,x2], [y, y+h, y+h, y], lw=1.2, color='black')
    ax.text((x1+x2)/2, y+h*1.05, label, ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')


def _add_label_dummies(df, reference=LABEL_REFERENCE):
    """Add dummy columns for jump labels (for mixed models)."""
    df = df.copy()
    labels = sorted(df['label_lower'].dropna().unique())
    non_ref = [lb for lb in labels if lb != reference]
    for lb in non_ref:
        col = 'lbl_' + re.sub(r'[^A-Za-z0-9]', '_', lb)
        df[col] = (df['label_lower'] == lb).astype(float)
    dummy_cols = ['lbl_' + re.sub(r'[^A-Za-z0-9]', '_', lb) for lb in non_ref]
    return df, dummy_cols


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_jumping_features(pose_frames, frame_indices, ann_fps=FPS):
    hip_y_L, hip_y_R           = [], []
    knee_y_L, knee_y_R         = [], []
    ankle_y_L, ankle_y_R       = [], []
    shoulder_y_L, shoulder_y_R = [], []
    wrist_y_L, wrist_y_R       = [], []
    conf_vals, scale_vals       = [], []
    n_valid = 0

    for fi in frame_indices:
        fk = str(fi)
        if fk not in pose_frames: continue
        fd = pose_frames[fk]
        scale = get_scale(fd)
        if scale is None: continue
        lh = get_kp(fd, KP['left_hip']);    rh = get_kp(fd, KP['right_hip'])
        lk = get_kp(fd, KP['left_knee']);   rk = get_kp(fd, KP['right_knee'])
        la = get_kp(fd, KP['left_ankle']);  ra = get_kp(fd, KP['right_ankle'])
        ls = get_kp(fd, KP['left_shoulder']); rs = get_kp(fd, KP['right_shoulder'])
        lw = get_kp(fd, KP['left_wrist']);  rw = get_kp(fd, KP['right_wrist'])
        if lh is None and rh is None and la is None and ra is None:
            continue
        n_valid += 1
        scale_vals.append(scale)
        if lh: hip_y_L.append(lh['y']/scale);    conf_vals.append(lh['confidence'])
        if rh: hip_y_R.append(rh['y']/scale);    conf_vals.append(rh['confidence'])
        if lk: knee_y_L.append(lk['y']/scale)
        if rk: knee_y_R.append(rk['y']/scale)
        if la: ankle_y_L.append(la['y']/scale)
        if ra: ankle_y_R.append(ra['y']/scale)
        if ls: shoulder_y_L.append(ls['y']/scale)
        if rs: shoulder_y_R.append(rs['y']/scale)
        if lw: wrist_y_L.append(lw['y']/scale)
        if rw: wrist_y_R.append(rw['y']/scale)

    if n_valid < 5: return None

    rec = {
        'n_valid_frames': n_valid,
        'n_total_frames': len(frame_indices),
        'pct_valid':      n_valid / len(frame_indices),
        'duration_sec':   len(frame_indices) / ann_fps,
        'mean_conf':      np.mean(conf_vals) if conf_vals else np.nan,
        'mean_scale':     np.mean(scale_vals),
    }

    def joint_feats(arr, name):
        a = np.array(arr)
        if len(a) < 5: return
        rec[f'{name}_amplitude'] = np.ptp(a)
        rec[f'{name}_std']       = np.std(a)
        rec[f'{name}_mean']      = np.mean(a)
        rec[f'{name}_iqr']       = float(np.percentile(a,75) - np.percentile(a,25))
        if len(a) >= 8:
            try:
                sm  = butter_lp(a, fs=ann_fps)
                vel = np.diff(sm) * ann_fps
                rec[f'{name}_vel_mean'] = np.mean(np.abs(vel))
                rec[f'{name}_vel_std']  = np.std(vel)
                rec[f'{name}_vel_max']  = np.max(np.abs(vel))
                if len(vel) >= 4:
                    acc = np.diff(vel) * ann_fps
                    rec[f'{name}_acc_mean'] = np.mean(np.abs(acc))
                    rec[f'{name}_acc_max']  = np.max(np.abs(acc))
            except: pass
        df_f, se, bp = spectral_features(a, ann_fps)
        rec[f'{name}_dom_freq']          = df_f
        rec[f'{name}_spectral_entropy']  = se
        rec[f'{name}_band_power_1_4hz']  = bp

    for arr, name in [
        (hip_y_L,      'hip_y_L'),   (hip_y_R,      'hip_y_R'),
        (knee_y_L,     'knee_y_L'),  (knee_y_R,     'knee_y_R'),
        (ankle_y_L,    'ankle_y_L'), (ankle_y_R,    'ankle_y_R'),
        (shoulder_y_L, 'shoulder_y_L'), (shoulder_y_R, 'shoulder_y_R'),
        (wrist_y_L,    'wrist_y_L'), (wrist_y_R,    'wrist_y_R'),
    ]:
        joint_feats(arr, name)

    for jname, L, R in [
        ('hip',   hip_y_L,   hip_y_R),
        ('knee',  knee_y_L,  knee_y_R),
        ('ankle', ankle_y_L, ankle_y_R),
    ]:
        if len(L) >= 5 and len(R) >= 5:
            ml = min(len(L), len(R))
            yl = np.array(L[:ml]); yr = np.array(R[:ml])
            rec[f'bilateral_{jname}_amp_diff']  = abs(np.ptp(yl) - np.ptp(yr))
            rec[f'bilateral_{jname}_y_corr']    = float(np.corrcoef(yl, yr)[0,1])
            rec[f'bilateral_{jname}_sym_index'] = (
                1 - abs(np.ptp(yl)-np.ptp(yr)) / (np.ptp(yl)+np.ptp(yr)+1e-8))
            try:
                xcorr = np.correlate(yl-yl.mean(), yr-yr.mean(), mode='full')
                lags  = np.arange(-(ml-1), ml)
                rec[f'bilateral_{jname}_phase_lag'] = abs(lags[np.argmax(xcorr)]) / ann_fps
            except: pass

    has_L = len(hip_y_L) >= 5
    has_R = len(hip_y_R) >= 5
    if has_L and has_R:
        ml = min(len(hip_y_L), len(hip_y_R))
        mean_hip = (np.array(hip_y_L[:ml]) + np.array(hip_y_R[:ml])) / 2
        rec['mean_hip_y_amplitude'] = np.ptp(mean_hip)
        rec['mean_hip_y_std']       = np.std(mean_hip)
        df_f, se, bp = spectral_features(mean_hip, ann_fps)
        rec['mean_hip_dom_freq']         = df_f
        rec['mean_hip_spectral_entropy'] = se
        rec['mean_hip_band_power_1_4hz'] = bp
        if len(mean_hip) >= 8:
            try:
                sm  = butter_lp(mean_hip, fs=ann_fps)
                vel = np.diff(sm) * ann_fps
                rec['mean_hip_vel_mean'] = np.mean(np.abs(vel))
                rec['mean_hip_vel_max']  = np.max(np.abs(vel))
                if len(vel) >= 4:
                    acc = np.diff(vel) * ann_fps
                    rec['mean_hip_acc_mean'] = np.mean(np.abs(acc))
            except: pass
    elif has_L:
        rec['mean_hip_y_amplitude'] = np.ptp(hip_y_L)
        rec['mean_hip_y_std']       = np.std(hip_y_L)
    elif has_R:
        rec['mean_hip_y_amplitude'] = np.ptp(hip_y_R)
        rec['mean_hip_y_std']       = np.std(hip_y_R)

    hip_amp   = max(float(np.ptp(hip_y_L))   if hip_y_L   else 0.0,
                    float(np.ptp(hip_y_R))   if hip_y_R   else 0.0)
    ankle_amp = max(float(np.ptp(ankle_y_L)) if ankle_y_L else 0.0,
                    float(np.ptp(ankle_y_R)) if ankle_y_R else 0.0)
    knee_amp  = max(float(np.ptp(knee_y_L))  if knee_y_L  else 0.0,
                    float(np.ptp(knee_y_R))  if knee_y_R  else 0.0)
    if hip_amp > 0:   rec['vertical_excursion_hip_max']   = hip_amp
    if ankle_amp > 0: rec['vertical_excursion_ankle_max'] = ankle_amp
    if knee_amp > 0:  rec['vertical_excursion_knee_max']  = knee_amp
    if hip_amp > 0 and ankle_amp > 0:
        rec['hip_ankle_amp_ratio'] = hip_amp / (ankle_amp + 1e-8)
    if hip_amp > 0 and knee_amp > 0:
        rec['hip_knee_amp_ratio']  = hip_amp / (knee_amp  + 1e-8)

    return rec


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL METHODS  (unified, matches RMM v3)
# ═══════════════════════════════════════════════════════════════════════

# ── ICC ─────────────────────────────────────────────────────────────
def compute_icc(clip_df, feat_cols):
    """Intraclass correlation — quantifies within-child clustering."""
    records = []
    for feat in feat_cols:
        sub = clip_df[['pid', feat]].dropna()
        if len(sub) < 10: continue
        groups = [g[feat].values for _, g in sub.groupby('pid') if len(g) >= 2]
        if len(groups) < 5: continue
        f_stat, _ = stats.f_oneway(*groups)
        n_total = sum(len(g) for g in groups); k = len(groups)
        n0 = (n_total - sum(len(g)**2/n_total for g in groups)) / (k-1)
        grand = np.concatenate(groups)
        ms_between = np.sum([len(g)*(np.mean(g)-np.mean(grand))**2 for g in groups])/(k-1)
        ms_within  = np.sum([np.sum((g-np.mean(g))**2) for g in groups])/(n_total-k)
        icc = max(0.0, (ms_between-ms_within)/(ms_between+(n0-1)*ms_within))
        records.append({'feature': feat, 'ICC': round(icc,4), 'f_stat': round(f_stat,3)})
    return pd.DataFrame(records).sort_values('ICC', ascending=False)

# ── MWU (clip or child level) ────────────────────────────────────────
def run_mwu_comparison(df, feat_cols, group_col='Group',
                       group_a='ASD', group_b='Non-ASD',
                       level='clip', subset_label='ALL'):
    ga = df[df[group_col]==group_a]; gb = df[df[group_col]==group_b]
    records = []
    for feat in feat_cols:
        av = ga[feat].dropna().values; bv = gb[feat].dropna().values
        if len(av) < 3 or len(bv) < 3: continue
        stat, p = stats.mannwhitneyu(av, bv, alternative='two-sided')
        d       = cohen_d(av, bv)
        ci_lo, ci_hi = bootstrap_ci_d(av, bv)
        records.append({
            'feature': feat, 'subset': subset_label, 'level': level,
            'group_a': group_a, 'group_b': group_b,
            f'{group_a}_n': len(av), f'{group_b}_n': len(bv),
            f'{group_a}_median': np.median(av), f'{group_b}_median': np.median(bv),
            f'{group_a}_mean': np.mean(av),   f'{group_b}_mean': np.mean(bv),
            f'{group_a}_std': np.std(av),     f'{group_b}_std': np.std(bv),
            'mw_stat': stat, 'p_raw': p,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'cles': cles(av, bv),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── LME + KR ────────────────────────────────────────────────────────
def run_lme_kr(clip_df, feat_cols, subset_label='full'):
    """LME with Kenward-Roger via rpy2 (falls back to statsmodels without KR)."""
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)
    for feat in feat_cols:
        keep = ['pid','Group_bin','age_mo_c','label_lower',feat] + dummy_cols
        sub  = df_use[[c for c in keep if c in df_use.columns]].dropna(
                    subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min() < 3: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        d  = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv)
        p_val = np.nan; coef = np.nan; se = np.nan
        method_used = 'none'; converged = False
        if _RPY2_OK:
            try:
                safe = re.sub(r'[^A-Za-z0-9_]', '_', feat)
                sub2 = sub.rename(columns={feat: safe})
                bterm = ' + '.join(dummy_cols) if dummy_cols else ''
                formula = (f'{safe} ~ Group_bin + age_mo_c'
                           + (f' + {bterm}' if bterm else '') + ' + (1|pid)')
                r_df = pandas2ri.py2rpy(sub2); ro.globalenv['r_df'] = r_df
                ro.r(f'fit <- lmerTest::lmer({formula}, data=r_df, REML=TRUE)')
                summ = pandas2ri.rpy2py(
                    ro.r('as.data.frame(coef(summary(fit,ddf="Kenward-Roger")))'))
                if 'Group_bin' in summ.index:
                    coef = float(summ.loc['Group_bin','Estimate'])
                    se   = float(summ.loc['Group_bin','Std. Error'])
                    p_val = float(summ.loc['Group_bin','Pr(>|t|)'])
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
            'n_asd':  sub[sub['Group_bin']==1]['pid'].nunique(),
            'n_nasd': sub[sub['Group_bin']==0]['pid'].nunique(),
            'n_clips': len(sub),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── CR2 ─────────────────────────────────────────────────────────────
def run_cr2(clip_df, feat_cols, subset_label='full'):
    """CR2 bias-reduced linearization (wildboottest)."""
    if not _WBT_OK:
        print("  [CR2] skipped — wildboottest not installed")
        return pd.DataFrame()
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo_c',feat]+dummy_cols].dropna(
                  subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 10: continue
        X_cols = ['Group_bin','age_mo_c'] + [c for c in dummy_cols
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

# ── GEE ─────────────────────────────────────────────────────────────
def run_gee(clip_df, feat_cols, subset_label='full'):
    """GEE (supplementary robustness check)."""
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)
    pid_map = {p: i for i, p in enumerate(df_use['pid'].unique())}
    df_use['pid_int'] = df_use['pid'].map(pid_map)
    for feat in feat_cols:
        sub = df_use[['pid_int','Group_bin','age_mo_c',feat]+dummy_cols].dropna(
                  subset=['pid_int','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 20: continue
        counts = sub.groupby('pid_int').size()
        sub = sub[sub['pid_int'].isin(counts[counts>=2].index)]
        if len(sub) < 20: continue
        try:
            safe  = re.sub(r'[^A-Za-z0-9_]', '_', feat)
            sub2  = sub.rename(columns={feat: safe})
            bterm = '+'.join([c for c in dummy_cols if sub2[c].std() > 1e-8])
            formula = (f'{safe} ~ Group_bin + age_mo_c'
                       + (f' + {bterm}' if bterm else ''))
            res = GEE.from_formula(formula, 'pid_int', data=sub2,
                                   family=Gaussian(),
                                   cov_struct=Exchangeable()).fit(maxiter=100)
            av = sub[sub['Group_bin']==1][feat].values
            nv = sub[sub['Group_bin']==0][feat].values
            records.append({
                'feature': feat, 'subset': subset_label, 'method': 'GEE',
                'coef_ASD': float(res.params.get('Group_bin', np.nan)),
                'p_raw':    float(res.pvalues.get('Group_bin', np.nan)),
                'cohens_d': cohen_d(av, nv), 'n_clips': len(sub),
            })
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Child-level permutation ──────────────────────────────────────────
def run_child_permutation(child_df, feat_cols, n_perm=5000, subset_label='full'):
    """Permute group labels at child level; compare mean difference."""
    rng = np.random.default_rng(42); records = []
    for feat in feat_cols:
        sub = child_df[['pid','Group',feat]].dropna()
        if sub['Group'].nunique() < 2: continue
        av = sub[sub['Group']=='ASD'][feat].values
        nv = sub[sub['Group']=='Non-ASD'][feat].values
        if len(av) < 3 or len(nv) < 3: continue
        obs_stat = abs(np.mean(av) - np.mean(nv)); n_asd = len(av)
        vals_arr = sub[feat].values; n_total = len(sub)
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            sl = rng.permutation(['ASD']*n_asd + ['Non-ASD']*(n_total-n_asd))
            a_v = vals_arr[np.array(sl)=='ASD']; n_v = vals_arr[np.array(sl)=='Non-ASD']
            a_v = a_v[~np.isnan(a_v)]; n_v = n_v[~np.isnan(n_v)]
            perm_stats[i] = abs(np.mean(a_v)-np.mean(n_v)) if len(a_v)>0 and len(n_v)>0 else 0
        p_perm = max(float(np.mean(perm_stats >= obs_stat)), 1.0/n_perm)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'ChildPerm',
            'obs_stat': float(obs_stat), 'p_raw': p_perm,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': len(av), 'n_nasd': len(nv),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Wild cluster bootstrap ───────────────────────────────────────────
def run_wild_bootstrap(child_df, feat_cols, n_boot=5000, subset_label='full'):
    """Wild cluster bootstrap at child level (Rademacher weights)."""
    rng = np.random.default_rng(99); records = []
    df_use = child_df.copy().dropna(subset=['age_mo'])
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo',feat]].dropna()
        if sub['Group_bin'].nunique() < 2: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        if len(av) < 3 or len(nv) < 3: continue
        n = len(sub); y = sub[feat].values.astype(float)
        X = np.column_stack([np.ones(n), sub['Group_bin'].values,
                              sub['age_mo'].values])
        try: beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except: continue
        resid = y - X @ beta; t_obs = beta[1]/(np.std(resid)/np.sqrt(n)+1e-10)
        X0 = X[:,[0,2]]
        try: beta0, _, _, _ = np.linalg.lstsq(X0, y, rcond=None)
        except: continue
        resid0 = y - X0 @ beta0
        pids = sub['pid'].values; u_pids = np.unique(pids)
        t_boot = np.zeros(n_boot)
        for b in range(n_boot):
            w_map = {p: rng.choice([-1.0,1.0]) for p in u_pids}
            w = np.array([w_map[p] for p in pids])
            y_b = X0 @ beta0 + resid0 * w
            try:
                beta_b, _, _, _ = np.linalg.lstsq(X, y_b, rcond=None)
                resid_b = y_b - X @ beta_b
                t_boot[b] = beta_b[1]/(np.std(resid_b)/np.sqrt(n)+1e-10)
            except: t_boot[b] = 0.0
        p_wb = max(float(np.mean(np.abs(t_boot) >= abs(t_obs))), 1.0/n_boot)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'WildBoot',
            'coef_ASD': float(beta[1]), 't_obs': float(t_obs), 'p_raw': p_wb,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': int(len(av)), 'n_nasd': int(len(nv)),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Consensus across methods ─────────────────────────────────────────
def make_consensus(results_dict, feat_cols, threshold=0.05):
    """Aggregate p-values from all methods; count how many are significant."""
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
                p = match['p_raw'].values[0]
                row[f'p_{mname}'] = round(p, 4)
                if p < threshold: n_sig += 1
        row['n_methods_sig'] = n_sig; rows.append(row)
    cons = pd.DataFrame(rows)
    # attach Cohen's d from best available method
    for mname in ['LME_KR','LME_noKR','ChildPerm','WildBoot','MWU_child']:
        res = results_dict.get(mname)
        if res is not None and len(res) and 'cohens_d' in res.columns:
            cons['cohens_d'] = cons['feature'].map(
                res.set_index('feature')['cohens_d'].to_dict())
            if 'd_ci_lo' in res.columns:
                cons['d_ci_lo'] = cons['feature'].map(
                    res.set_index('feature')['d_ci_lo'].to_dict())
                cons['d_ci_hi'] = cons['feature'].map(
                    res.set_index('feature')['d_ci_hi'].to_dict())
            break
    return cons.sort_values('n_methods_sig', ascending=False)

# ── Consistency gate ─────────────────────────────────────────────────
def run_consistency_gate(feat_df, feat_cols, sig_feats, label_col='label_lower'):
    """Check whether Group effect direction is consistent across jump labels."""
    label_mwu = {}
    for lbl in sorted(feat_df[label_col].dropna().unique()):
        sub = feat_df[feat_df[label_col] == lbl]
        asd_n = sub[sub['Group']=='ASD']['pid'].nunique()
        nan_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        if asd_n < 3 or nan_n < 3: continue
        recs = []
        for feat in feat_cols:
            av = sub[sub['Group']=='ASD'][feat].dropna().values
            nv = sub[sub['Group']=='Non-ASD'][feat].dropna().values
            if len(av) < 3 or len(nv) < 3: continue
            _, p = stats.mannwhitneyu(av, nv, alternative='two-sided')
            recs.append({'feature': feat, 'cohens_d': cohen_d(av,nv),
                         'p_raw': p, 'label': lbl})
        if recs: label_mwu[lbl] = pd.DataFrame(recs)

    cons_recs = []; consistent_feats = []
    label_all = (pd.concat(label_mwu.values(), ignore_index=True)
                 if label_mwu else pd.DataFrame())
    for feat in sig_feats:
        if len(label_all) == 0: break
        sub = label_all[label_all['feature'] == feat]
        if len(sub) < 2: continue
        signs = np.sign(sub['cohens_d'].values)
        n_same = int((signs == signs[0]).sum()); passed = (n_same == len(sub))
        cons_recs.append({'feature': feat, 'n_labels_tested': len(sub),
                          'n_same_direction': n_same, 'consistent': passed})
        if passed: consistent_feats.append(feat)
    return pd.DataFrame(cons_recs), consistent_feats, label_mwu

# ── Label × Group interaction ────────────────────────────────────────
def run_label_group_interaction(clip_df, feat_cols, subset_label='full'):
    """Test whether Group effect DIFFERS by jump label — validity check."""
    records = []
    df_use = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    df_use, dummy_cols  = _add_label_dummies(df_use)
    for feat in feat_cols:
        keep = ['pid','Group_bin','age_mo_c','label_lower',feat] + dummy_cols
        sub  = df_use[[c for c in keep if c in df_use.columns]].dropna(
                   subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2 or sub['pid'].nunique() < 5: continue
        if not dummy_cols: continue
        lbl_x_grp = [f'Group_bin:{c}' for c in dummy_cols]
        formula = (f'{feat} ~ Group_bin + age_mo_c + ' + '+'.join(dummy_cols)
                   + ' + ' + '+'.join(lbl_x_grp))
        try:
            mdf = smf.mixedlm(formula, sub, groups=sub['pid']).fit(
                method=['lbfgs'], reml=True, maxiter=300)
            for lxg in lbl_x_grp:
                if lxg in mdf.pvalues.index:
                    records.append({
                        'feature': feat, 'interaction_term': lxg,
                        'coef': float(mdf.params.get(lxg, np.nan)),
                        'p_raw': float(mdf.pvalues.get(lxg, np.nan)),
                        'subset': subset_label,
                    })
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Spearman age correlations ────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# LME SUITE  (Part 6)
# ═══════════════════════════════════════════════════════════════════════

def _prep_lme_df(df, feat):
    cols = ['pid','Group','age_mo','label_lower',feat]
    tmp  = df[[c for c in cols if c in df.columns]].dropna().copy()
    if len(tmp) < 6: return None
    tmp['Group_bin'] = (tmp['Group'] == 'ASD').astype(float)
    tmp['age_mo_c']  = tmp['age_mo'] - tmp['age_mo'].mean()
    return tmp

def run_lme_feature(df, feat, formula_type='main', subset_label='combined'):
    tmp = _prep_lme_df(df, feat)
    if tmp is None or tmp['pid'].nunique() < 4: return None
    try:
        if formula_type == 'main':
            formula = f'Q("{feat}") ~ Group_bin + age_mo_c'; key_var = 'Group_bin'
        elif formula_type == 'age_strat':
            formula = f'Q("{feat}") ~ Group_bin'; key_var = 'Group_bin'
        elif formula_type == 'asd_traj':
            tmp = tmp[tmp['Group']=='ASD'].copy()
            if len(tmp) < 6 or tmp['pid'].nunique() < 3: return None
            formula = f'Q("{feat}") ~ age_mo_c'; key_var = 'age_mo_c'
        elif formula_type == 'interaction':
            formula = f'Q("{feat}") ~ Group_bin * age_mo_c'
            key_var = 'Group_bin:age_mo_c'
        else:
            return None
        safe = feat.replace('.','_'); tmp = tmp.rename(columns={feat: safe})
        formula = formula.replace(f'Q("{feat}")', safe)
        model  = smf.mixedlm(formula, tmp, groups=tmp['pid'])
        result = model.fit(reml=True, method='lbfgs')
        params = result.params; pvals = result.pvalues; ci_ = result.conf_int()
        if key_var not in params.index:
            kv_list = [k for k in params.index if 'Group_bin' in k and 'age' in k]
            if not kv_list: return None
            key_var = kv_list[0]
        return {
            'feature': feat, 'subset': subset_label, 'formula_type': formula_type,
            'n_obs': int(len(tmp)), 'n_subjects': int(tmp['pid'].nunique()),
            'coef': float(params[key_var]), 'ci_lo': float(ci_.loc[key_var,0]),
            'ci_hi': float(ci_.loc[key_var,1]), 'p_lme': float(pvals[key_var]),
            'converged': result.converged,
        }
    except: return None

def run_lme_suite(feat_df, child_df, primary_feats, output_dir, variant_name):
    hr(f"PART 6 (LME/GEE): {variant_name}")
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    def _fdr_flag(df_res, p_col):
        if len(df_res) > 1:
            _, p_fdr, _, _ = multipletests(df_res[p_col], method='fdr_bh')
            df_res['p_fdr'] = p_fdr; df_res['sig_fdr05'] = p_fdr < 0.05
        else:
            df_res['p_fdr'] = df_res[p_col]; df_res['sig_fdr05'] = df_res[p_col] < 0.05
        df_res['sig_raw05'] = df_res[p_col] < 0.05
        return df_res

    # 6a: main effect
    print("\n--- 6a: LME clip-level main effect ---")
    recs_6a = [r for feat in primary_feats
               for r in [run_lme_feature(feat_df, feat, 'main', 'combined_clip')]
               if r is not None]
    df_6a = None
    if recs_6a:
        df_6a = _fdr_flag(pd.DataFrame(recs_6a).sort_values('p_lme'), 'p_lme')
        df_6a.to_csv(os.path.join(output_dir, 'lme_6a_clip_main.csv'), index=False)
        print(f"  Tested: {len(df_6a)}  sig_raw: {df_6a['sig_raw05'].sum()}  FDR: {df_6a['sig_fdr05'].sum()}")

    # 6b: age-stratified
    print("\n--- 6b: LME age-stratified ---")
    recs_6b = []
    for band in STAT_BANDS:
        sub = feat_df[feat_df['age_band'] == band]
        if sub[sub['Group']=='ASD']['pid'].nunique() < 3: continue
        band_recs = [r for feat in primary_feats
                     for r in [run_lme_feature(sub.copy(), feat, 'age_strat', band)]
                     if r is not None]
        if band_recs:
            df_b = _fdr_flag(pd.DataFrame(band_recs), 'p_lme')
            df_b['age_band'] = band; recs_6b.append(df_b)
            print(f"  {band}: sig_raw={df_b['sig_raw05'].sum()} FDR={df_b['sig_fdr05'].sum()}")
    if recs_6b:
        pd.concat(recs_6b, ignore_index=True).to_csv(
            os.path.join(output_dir, 'lme_6b_age_stratified.csv'), index=False)

    # 6c: within-ASD trajectory
    print("\n--- 6c: LME within-ASD trajectory ---")
    asd_df = feat_df[feat_df['Group']=='ASD'].copy()
    recs_6c = [r for feat in primary_feats
               for r in [run_lme_feature(asd_df, feat, 'asd_traj', 'ASD_only')]
               if r is not None]
    if recs_6c:
        df_6c = _fdr_flag(pd.DataFrame(recs_6c).sort_values('p_lme'), 'p_lme')
        df_6c.to_csv(os.path.join(output_dir, 'lme_6c_asd_trajectory.csv'), index=False)
        print(f"  sig_raw: {df_6c['sig_raw05'].sum()} FDR: {df_6c['sig_fdr05'].sum()}")

    # 6d: group × age interaction
    print("\n--- 6d: LME group × age growth-curve ---")
    recs_6d = [r for feat in primary_feats
               for r in [run_lme_feature(feat_df, feat, 'interaction', 'combined_clip')]
               if r is not None]
    if recs_6d:
        df_6d = _fdr_flag(pd.DataFrame(recs_6d).sort_values('p_lme'), 'p_lme')
        df_6d.to_csv(os.path.join(output_dir, 'lme_6d_growth_curve.csv'), index=False)
        print(f"  sig_raw: {df_6d['sig_raw05'].sum()} FDR: {df_6d['sig_fdr05'].sum()}")

    # 6e: GEE robustness
    print("\n--- 6e: GEE robustness ---")
    gee_recs = []
    for feat in primary_feats:
        r = None
        try:
            tmp = _prep_lme_df(feat_df, feat)
            if tmp is None or tmp['pid'].nunique() < 4: continue
            pid_map = {p: i for i, p in enumerate(tmp['pid'].unique())}
            tmp['pid_int'] = tmp['pid'].map(pid_map)
            counts = tmp.groupby('pid_int').size()
            tmp = tmp[tmp['pid_int'].isin(counts[counts>=2].index)]
            if len(tmp) < 20: continue
            safe = feat.replace('.','_'); tmp2 = tmp.rename(columns={feat: safe})
            res = GEE.from_formula(f'{safe} ~ Group_bin + age_mo_c', 'pid_int',
                                   data=tmp2, family=Gaussian(),
                                   cov_struct=Exchangeable()).fit(maxiter=100)
            gee_recs.append({
                'feature': feat, 'n_obs': len(tmp),
                'n_subjects': tmp['pid'].nunique(),
                'coef_gee': float(res.params.get('Group_bin', np.nan)),
                'ci_lo_gee': float(res.conf_int().loc['Group_bin',0]),
                'ci_hi_gee': float(res.conf_int().loc['Group_bin',1]),
                'p_gee': float(res.pvalues.get('Group_bin', np.nan)),
            })
        except: pass
    if gee_recs:
        df_6e = _fdr_flag(pd.DataFrame(gee_recs).sort_values('p_gee'), 'p_gee')
        df_6e.to_csv(os.path.join(output_dir, 'lme_6e_gee_robustness.csv'), index=False)
        print(f"  GEE sig_raw: {df_6e['sig_raw05'].sum()} FDR: {df_6e['sig_fdr05'].sum()}")

    # Fig LME-1: forest plot
    if df_6a is not None and len(df_6a):
        fp = df_6a.copy()
        fp['label'] = fp['feature'].map(SHORT_LABELS).fillna(fp['feature'])
        fp = fp.sort_values('coef')
        fig, ax = plt.subplots(figsize=(10, max(5, len(fp)*0.45)))
        bar_colors = [ASD_COLOR if c > 0 else NONASD_COLOR for c in fp['coef']]
        ax.barh(fp['label'], fp['coef'], color=bar_colors, edgecolor='white',
                height=0.6, alpha=0.85)
        for j, (_, row) in enumerate(fp.iterrows()):
            ax.plot([row['ci_lo'], row['ci_hi']], [j, j], color='black', lw=1.5)
            ax.plot(row['ci_lo'], j, '|', color='black', markersize=6)
            ax.plot(row['ci_hi'], j, '|', color='black', markersize=6)
            if row.get('sig_fdr05'):
                ax.text(row['ci_hi']+0.002, j, '★', va='center', fontsize=10, color='gold')
            elif row.get('sig_raw05'):
                ax.text(row['ci_hi']+0.002, j, '●', va='center', fontsize=8)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('LME coefficient for Group (ASD vs Non-ASD)')
        ax.set_title(f'LME Forest Plot — {variant_name}  ★=FDR sig  ●=raw p<0.05',
                     fontweight='bold')
        ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                           mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
        plt.tight_layout()
        fig.savefig(os.path.join(figure_dir, 'fig_lme1_forest.png'))
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# BAYESIAN SUITE  (Part 7)  — now with prior sensitivity + PPC
# ═══════════════════════════════════════════════════════════════════════


def _build_bayes_df(df, feat, reference=LABEL_REFERENCE):
    cols = ['pid','Group','age_mo','label_lower',feat]
    tmp  = df[[c for c in cols if c in df.columns]].dropna().copy()
    if len(tmp) < 8 or tmp['pid'].nunique() < 4: return None
    tmp['Group_bin'] = (tmp['Group'] == 'ASD').astype(float)
    tmp['age_c']     = tmp['age_mo'] - tmp['age_mo'].mean()
    # label dummies
    if 'label_lower' in tmp.columns:
        labels = sorted(tmp['label_lower'].unique())
        non_ref = [lb for lb in labels if lb != reference]
        lbl_mat = np.column_stack(
            [(tmp['label_lower']==lb).astype(float).values for lb in non_ref]
        ) if non_ref else np.zeros((len(tmp), 0))
    else:
        lbl_mat = np.zeros((len(tmp), 0))
    y_z, ym, ys = _standardise(tmp[feat])
    pids, pid_idx = np.unique(tmp['pid'].values, return_inverse=True)
    return {
        'df': tmp, 'y_z': y_z.astype(float),
        'group_bin': tmp['Group_bin'].values.astype(float),
        'age_c':     tmp['age_c'].values.astype(float),
        'lbl_mat':   lbl_mat, 'n_lbl_dum': lbl_mat.shape[1],
        'pid_idx':   pid_idx, 'n_pids': len(pids),
        'y_mean': ym, 'y_std': ys, 'n_obs': len(tmp),
    }


def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                    chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha     = pm.Normal('alpha', 0, 1)
        b_group   = pm.Normal('b_group', 0, prior_sd)
        b_age     = pm.Normal('b_age', 0, 0.5)
        lbl_contrib = 0.0
        if bd['n_lbl_dum'] > 0:
            b_lbl = pm.Normal('b_lbl', 0, 0.5, shape=bd['n_lbl_dum'])
            lbl_contrib = pm.math.dot(bd['lbl_mat'], b_lbl)
        sigma_pid = pm.HalfNormal('sigma_pid', 1)
        sigma     = pm.HalfNormal('sigma', 1)
        alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu = (alpha + alpha_pid[bd['pid_idx']] + lbl_contrib
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
        'p_pos': float((b_post>0).mean()), 'bf10': bf10,
        'rhat': rhat, 'ess_bulk': ess, 'n_divergences': n_div,
        'converged': bool(rhat < 1.05 and ess > 400 and n_div == 0),
        'prior_sd': prior_sd,
    }

def prior_predictive_check(bd, feat, prior_sd=0.5):
    """Check whether prior covers observed data range."""
    with pm.Model():
        b_group = pm.Normal('b_group', 0, prior_sd)
        b_age   = pm.Normal('b_age', 0, 0.5)
        sigma   = pm.HalfNormal('sigma', 1)
        alpha   = pm.Normal('alpha', 0, 1)
        mu = alpha + b_group*bd['group_bin'] + b_age*bd['age_c']
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        ppc = pm.sample_prior_predictive(samples=200, random_seed=42)
    prior_ys    = ppc.prior_predictive['y_obs'].values.flatten()
    obs_range   = (bd['y_z'].min(), bd['y_z'].max())
    prior_range = (float(np.percentile(prior_ys,1)),
                   float(np.percentile(prior_ys,99)))
    return {
        'feature': feat,
        'obs_min': obs_range[0], 'obs_max': obs_range[1],
        'prior_p1': prior_range[0], 'prior_p99': prior_range[1],
        'plausible': (prior_range[0] <= obs_range[0] and
                      prior_range[1] >= obs_range[1]),
    }

def _fit_bayes_interaction(bd, prior_sd=0.5, draws=BAYES_DRAWS,
                           tune=BAYES_TUNE, chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha      = pm.Normal('alpha', 0, 1)
        b_group    = pm.Normal('b_group', 0, prior_sd)
        b_age      = pm.Normal('b_age', 0, 0.5)
        b_interact = pm.Normal('b_interact', 0, 0.5)
        sigma_pid  = pm.HalfNormal('sigma_pid', 1)
        sigma      = pm.HalfNormal('sigma', 1)
        alpha_pid  = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu = (alpha + alpha_pid[bd['pid_idx']]
              + b_group*bd['group_bin'] + b_age*bd['age_c']
              + b_interact*bd['group_bin']*bd['age_c'])
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    int_post = idata.posterior['b_interact'].values.flatten()
    hdi      = az.hdi(idata, var_names=['b_interact'], hdi_prob=0.94)['b_interact'].values
    diag     = az.summary(idata, var_names=['b_interact'], hdi_prob=0.94)
    rhat     = float(diag['r_hat'].values[0]); ess = float(diag['ess_bulk'].values[0])
    n_div    = int(idata.sample_stats['diverging'].values.sum())
    return idata, {
        'b_interact_mean': float(int_post.mean()), 'b_interact_sd': float(int_post.std()),
        'hdi94_lo': float(hdi[0]), 'hdi94_hi': float(hdi[1]),
        'p_pos': float((int_post>0).mean()),
        'bf10': _savage_dickey_bf(int_post, prior_sd),
        'rhat': rhat, 'ess_bulk': ess, 'n_divergences': n_div,
        'converged': bool(rhat<1.05 and ess>400 and n_div==0),
    }

def _fit_bayes_age_only(bd_asd, prior_sd=0.5, draws=BAYES_DRAWS,
                        tune=BAYES_TUNE, chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha     = pm.Normal('alpha', 0, 1)
        b_age     = pm.Normal('b_age', 0, prior_sd)
        sigma_pid = pm.HalfNormal('sigma_pid', 1)
        sigma     = pm.HalfNormal('sigma', 1)
        alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd_asd['n_pids'])
        mu = alpha + alpha_pid[bd_asd['pid_idx']] + b_age*bd_asd['age_c']
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd_asd['y_z'])
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    age_post = idata.posterior['b_age'].values.flatten()
    hdi      = az.hdi(idata, var_names=['b_age'], hdi_prob=0.94)['b_age'].values
    diag     = az.summary(idata, var_names=['b_age'], hdi_prob=0.94)
    rhat     = float(diag['r_hat'].values[0]); ess = float(diag['ess_bulk'].values[0])
    n_div    = int(idata.sample_stats['diverging'].values.sum())
    return idata, {
        'b_age_mean': float(age_post.mean()), 'b_age_sd': float(age_post.std()),
        'hdi94_lo': float(hdi[0]), 'hdi94_hi': float(hdi[1]),
        'p_pos': float((age_post>0).mean()),
        'bf10': _savage_dickey_bf(age_post, prior_sd),
        'rhat': rhat, 'ess_bulk': ess, 'n_divergences': n_div,
        'converged': bool(rhat<1.05 and ess>400 and n_div==0),
    }

def _bayes_forest_plot(df_res, coef_col, hdi_lo_col, hdi_hi_col,
                       title, save_path, p_pos_col=None):
    if df_res is None or len(df_res) == 0: return
    fp = df_res.copy()
    fp['label'] = fp['feature'].map(SHORT_LABELS).fillna(fp['feature'])
    fp = fp.sort_values(coef_col)
    fig, ax = plt.subplots(figsize=(11, max(5, len(fp)*0.5)))
    for j, (_, row) in enumerate(fp.iterrows()):
        c = ASD_COLOR if row[coef_col] > 0 else NONASD_COLOR
        ax.plot([row[hdi_lo_col], row[hdi_hi_col]], [j,j],
                color=c, lw=2.5, alpha=0.8)
        ax.scatter(row[coef_col], j, color=c, s=60, zorder=5)
        ax.plot(row[hdi_lo_col], j, '|', color=c, markersize=7)
        ax.plot(row[hdi_hi_col], j, '|', color=c, markersize=7)
        ann = ''
        if p_pos_col and p_pos_col in row.index:
            pp = row[p_pos_col]; eff_p = pp if pp >= 0.5 else 1-pp
            ann = f'P={eff_p:.2f}'
        if 'bf10' in row.index and not np.isnan(float(row['bf10'])):
            ann += f'  BF={float(row["bf10"]):.1f}'
        if ann:
            ax.text(row[hdi_hi_col]+0.01, j, ann, va='center', fontsize=7)
        if 'converged' in row.index and not row['converged']:
            ax.text(row[hdi_lo_col]-0.01, j, '⚠', va='center',
                    ha='right', fontsize=8, color='orange')
    ax.axvline(0, color='black', lw=1.2, ls='--')
    ax.set_yticks(range(len(fp))); ax.set_yticklabels(fp['label'], fontsize=9)
    ax.set_xlabel('Posterior mean  |  94% HDI  (standardised units)')
    ax.set_title(title, fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')],
              fontsize=9)
    plt.tight_layout(); fig.savefig(save_path); plt.close(fig)

def run_bayesian_suite(feat_df, primary_feats, output_dir, variant_name):
    hr(f"PART 7 (BAYESIAN): {variant_name}")
    if not RUN_BAYESIAN or not _PYMC_OK:
        print("  Bayesian skipped"); return
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    # choose top features by permutation p-value if available
    perm_path = os.path.join(output_dir, 'stats_child_permutation.csv')
    if os.path.isfile(perm_path):
        perm_res = pd.read_csv(perm_path)
        bayes_feats = perm_res.sort_values('p_raw').head(15)['feature'].tolist()
    else:
        bayes_feats = primary_feats[:10]

    print(f"\n  Fitting Bayesian models on {len(bayes_feats)} features")
    print(f"  {BAYES_CHAINS} chains × {BAYES_DRAWS} draws × {BAYES_TUNE} tune")

    # ── 7a: main effect with prior sensitivity ────────────────────
    print("\n--- 7a: Bayesian main effect + prior sensitivity ---")
    ppc_records      = []; sensitivity_records = []; bayes_records = []

    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        # prior predictive check
        try:
            ppc_rec = prior_predictive_check(bd, feat)
            ppc_records.append(ppc_rec)
            if not ppc_rec['plausible']:
                print(f"  ⚠ PPC narrow for {feat}")
        except Exception as e:
            print(f"  PPC failed for {feat}: {e}")
        # sensitivity across three prior widths
        bf_vals = {}
        for psd in PRIOR_SDS:
            try:
                _, summ = _fit_bayes_main(bd, prior_sd=psd)
                summ['feature'] = feat; summ['prior_sd'] = psd
                sensitivity_records.append(summ); bf_vals[psd] = summ['bf10']
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")
        # primary result (prior_sd=0.5) + robustness flag
        if 0.5 in bf_vals:
            match = [r for r in sensitivity_records
                     if r['feature']==feat and r['prior_sd']==0.5]
            if match:
                rec = match[-1].copy()
                bfs = [bf_vals[p] for p in PRIOR_SDS
                       if p in bf_vals and not np.isnan(bf_vals[p])]
                rec['bf_robust'] = bool(
                    len(bfs) >= 2 and all((b>1)==(bfs[0]>1) for b in bfs))
                bayes_records.append(rec)
                flag = '✓' if rec.get('converged') else '⚠'
                bf_str = ' | '.join(
                    [f"sd={p}:BF={bf_vals.get(p,np.nan):.2f}" for p in PRIOR_SDS])
                print(f"  {feat:<45} {bf_str} {flag}")

    if ppc_records:
        pd.DataFrame(ppc_records).to_csv(
            os.path.join(output_dir, 'bayes_ppc.csv'), index=False)
    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            os.path.join(output_dir, 'bayes_sensitivity.csv'), index=False)
    df_7a = None
    if bayes_records:
        df_7a = pd.DataFrame(bayes_records).sort_values('bf10', ascending=False)
        df_7a.to_csv(os.path.join(output_dir, 'bayes_7a_main_effect.csv'), index=False)
        print(f"\n  BF10>3:  {(df_7a['bf10']>3).sum()}/{len(df_7a)}")
        print(f"  BF10>10: {(df_7a['bf10']>10).sum()}/{len(df_7a)}")
        print(f"  BF robust: {df_7a['bf_robust'].sum()}/{len(df_7a)}")
        bad = df_7a[~df_7a['converged']]
        if len(bad):
            print(f"  ⚠ {len(bad)} convergence issues")
            for _, r in bad.iterrows():
                print(f"    {r['feature']}  rhat={r.get('rhat',np.nan):.3f}")
        _bayes_forest_plot(
            df_7a, 'b_group_mean', 'hdi94_lo', 'hdi94_hi',
            title=(f'Bayesian Forest — {variant_name}\n'
                   'model: y ~ Group + age + label + (1|pid)  β~N(0,0.5)  94% HDI'),
            save_path=os.path.join(figure_dir, 'fig_bayes1_forest_main.png'),
            p_pos_col='p_pos')
        print("  Saved fig_bayes1_forest_main.png")

    # ── Fig: prior sensitivity ────────────────────────────────────
    sens_path = os.path.join(output_dir, 'bayes_sensitivity.csv')
    if os.path.isfile(sens_path):
        sens = pd.read_csv(sens_path)
        if len(sens):
            feats_s = sens['feature'].unique()[:12]
            fig, axes = plt.subplots(int(np.ceil(len(feats_s)/3)), 3,
                                     figsize=(15, 4*int(np.ceil(len(feats_s)/3))))
            fig.suptitle('Prior Sensitivity — BF10 across prior widths', fontweight='bold')
            axes = axes.flatten()
            for i, feat in enumerate(feats_s):
                ax = axes[i]; sub = sens[sens['feature']==feat].sort_values('prior_sd')
                ax.plot(sub['prior_sd'], sub['bf10'], marker='o', color=ASD_COLOR, lw=2)
                ax.axhline(3,  color='green', lw=1, ls='--', label='BF=3')
                ax.axhline(1,  color='gray',  lw=0.8, ls=':')
                ax.set_xlabel('Prior SD'); ax.set_ylabel('BF10')
                ax.set_title(SHORT_LABELS.get(feat, feat)[:25], fontsize=9)
                ax.legend(fontsize=7)
            for j in range(len(feats_s), len(axes)): axes[j].set_visible(False)
            plt.tight_layout()
            fig.savefig(os.path.join(figure_dir, 'fig_bayes_prior_sensitivity.png'))
            plt.close(fig)
            print("  Saved fig_bayes_prior_sensitivity.png")

    # ── 7b: age-stratified ───────────────────────────────────────
    print("\n--- 7b: Bayesian age-stratified ---")
    recs_7b = []
    for band in STAT_BANDS:
        sub = feat_df[feat_df['age_band'] == band].copy()
        if sub[sub['Group']=='ASD']['pid'].nunique() < 3: continue
        for feat in bayes_feats:
            bd = _build_bayes_df(sub, feat)
            if bd is None: continue
            try:
                _, summ = _fit_bayes_main(bd)
                summ['feature'] = feat; summ['age_band'] = band
                recs_7b.append(summ)
            except: pass
    if recs_7b:
        df_7b = pd.DataFrame(recs_7b)
        df_7b.to_csv(os.path.join(output_dir, 'bayes_7b_age_stratified.csv'), index=False)
        for band in STAT_BANDS:
            sub_b = df_7b[df_7b['age_band']==band]
            if len(sub_b) == 0: continue
            _bayes_forest_plot(
                sub_b, 'b_group_mean', 'hdi94_lo', 'hdi94_hi',
                title=f'Bayesian Forest — {variant_name} — {band}',
                save_path=os.path.join(figure_dir,
                    f'fig_bayes_forest_{band.replace("-","_")}.png'),
                p_pos_col='p_pos')

    # ── 7c: within-ASD trajectory ────────────────────────────────
    print("\n--- 7c: Bayesian within-ASD trajectory ---")
    asd_sub = feat_df[feat_df['Group']=='ASD'].copy()
    recs_7c = []
    if asd_sub['pid'].nunique() >= 4:
        for feat in bayes_feats:
            bd = _build_bayes_df(asd_sub, feat)
            if bd is None: continue
            try:
                _, summ = _fit_bayes_age_only(bd)
                summ['feature'] = feat; recs_7c.append(summ)
            except: pass
    if recs_7c:
        df_7c = pd.DataFrame(recs_7c).sort_values('b_age_mean', key=abs, ascending=False)
        df_7c.to_csv(os.path.join(output_dir, 'bayes_7c_asd_trajectory.csv'), index=False)
        _bayes_forest_plot(
            df_7c.rename(columns={'b_age_mean':'b_coef',
                                  'hdi94_lo':'hdi_lo','hdi94_hi':'hdi_hi'}),
            'b_coef','hdi_lo','hdi_hi',
            title=f'Bayesian Within-ASD Trajectory — {variant_name}',
            save_path=os.path.join(figure_dir, 'fig_bayes_asd_trajectory.png'),
            p_pos_col='p_pos')

    # ── 7d: growth-curve interaction ─────────────────────────────
    print("\n--- 7d: Bayesian growth-curve interaction ---")
    recs_7d = []
    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        try:
            _, summ = _fit_bayes_interaction(bd)
            summ['feature'] = feat; recs_7d.append(summ)
        except: pass
    if recs_7d:
        df_7d = pd.DataFrame(recs_7d).sort_values('b_interact_mean',
                                                    key=abs, ascending=False)
        df_7d.to_csv(os.path.join(output_dir, 'bayes_7d_growth_curve.csv'), index=False)
        _bayes_forest_plot(
            df_7d.rename(columns={'b_interact_mean':'b_coef',
                                  'hdi94_lo':'hdi_lo','hdi94_hi':'hdi_hi'}),
            'b_coef','hdi_lo','hdi_hi',
            title=f'Bayesian Growth-Curve Interaction — {variant_name}',
            save_path=os.path.join(figure_dir, 'fig_bayes_growth_curve.png'),
            p_pos_col='p_pos')

    # ── 7e: diagnostics summary ───────────────────────────────────
    print("\n--- 7e: Convergence diagnostics ---")
    all_bayes = []
    for fname in ['bayes_7a_main_effect.csv','bayes_7b_age_stratified.csv',
                  'bayes_7c_asd_trajectory.csv','bayes_7d_growth_curve.csv']:
        fp = os.path.join(output_dir, fname)
        if os.path.isfile(fp):
            tmp = pd.read_csv(fp); tmp['source'] = fname; all_bayes.append(tmp)
    if all_bayes:
        diag = pd.concat(all_bayes, ignore_index=True)
        if 'rhat' in diag.columns:
            print(f"  Total models: {len(diag)}")
            print(f"  R-hat > 1.05: {(diag['rhat']>1.05).sum()}")
            print(f"  ESS < 400:    {(diag['ess_bulk']<400).sum()}")
        diag.to_csv(os.path.join(output_dir, 'bayes_diagnostics_all.csv'), index=False)


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def run_loso_cv(df, feat_cols, label='ASD', n_perm=500, seed=42,
                min_feats=2, min_per_class=4):
    df = df.copy(); df['y'] = (df['Group'] == label).astype(int)
    if df['y'].sum() < min_per_class or (1-df['y']).sum() < min_per_class:
        return None
    usable = [f for f in feat_cols
              if f in df.columns and df[f].notna().mean() > 0.5]
    if len(usable) < min_feats: return None
    df[usable] = df[usable].fillna(df[usable].median())
    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=1000, C=0.1,
                                                class_weight='balanced',
                                                random_state=seed))])
    y_true, y_score = [], []
    for pid in df['pid'].unique():
        test = df[df['pid']==pid]; train = df[df['pid']!=pid]
        if len(train['y'].unique()) < 2: continue
        try:
            pipe.fit(train[usable].values, train['y'].values)
            y_score.extend(pipe.predict_proba(test[usable].values)[:,1].tolist())
            y_true.extend(test['y'].values.tolist())
        except: continue
    if len(set(y_true)) < 2: return None
    auc = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    perm_aucs = [roc_auc_score(rng.permuted(np.array(y_true)), y_score)
                 for _ in range(n_perm)]
    p_perm = float((np.array(perm_aucs) >= auc).mean())
    print(f"    AUC={auc:.3f}  p_perm={p_perm:.4f}  n_feat={len(usable)}")
    return {'auc': auc, 'perm_p': p_perm, 'n_features': len(usable),
            'n_subjects': df['pid'].nunique(),
            'y_true': y_true, 'y_score': y_score, 'perm_aucs': perm_aucs}


# ═══════════════════════════════════════════════════════════════════════
# MAIN PER-VARIANT RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_variant(variant, jump_rmm_all):
    name        = variant['name']
    JUMP_LABELS = variant['jump_labels']
    OUTPUT_DIR  = variant['output_dir']
    FIGURE_DIR  = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(FIGURE_DIR, exist_ok=True)

    hr(f"VARIANT: {name}  |  labels: {JUMP_LABELS}")
    jump_rmm = jump_rmm_all[jump_rmm_all['label_lower'].isin(JUMP_LABELS)].copy()
    if len(jump_rmm) == 0:
        print(f"ERROR [{name}]: No clips found. Skipping."); return

    print(f"\nClips: {len(jump_rmm)} | ASD: {len(jump_rmm[jump_rmm['Group']=='ASD'])} "
          f"| Non-ASD: {len(jump_rmm[jump_rmm['Group']=='Non-ASD'])}")
    print(jump_rmm.groupby(['label_lower','Group']).size().reset_index(name='n')
          .to_string(index=False))

    # ── PART 1: FEATURE EXTRACTION ─────────────────────────────────
    hr(f"PART 1: FEATURE EXTRACTION  [{name}]")
    all_features = []; n_ok = n_fail_ts = n_fail_pose = n_fail_kp = 0
    for _, row in jump_rmm.iterrows():
        ts_str = str(row.get('matched_ts',''))
        segs   = parse_timestamps(ts_str)
        if not segs: n_fail_ts += 1; continue
        try:
            with open(row['hrnet_path'], 'r') as f: pose_data = json.load(f)
            pose_frames = pose_data.get('frames', {})
            ann_fps     = float(pose_data.get('ann_fps', FPS))
            if ann_fps != FPS: segs = parse_timestamps(ts_str, fps=ann_fps)
        except: n_fail_pose += 1; continue
        for seg_start, seg_end in segs:
            frame_idx = list(range(seg_start, seg_end+1))
            if len(frame_idx) < 5: continue
            feats = extract_jumping_features(pose_frames, frame_idx, ann_fps)
            if feats is None: n_fail_kp += 1; continue
            n_ok += 1
            feats.update({'pid': row['pid'], 'Group': row['Group'],
                          'age_mo': row['age_mo'], 'age_band': row['age_band'],
                          'original_label': row['matched_label'],
                          'label_lower': row['label_lower'],
                          'clip': row.get('clip_filename','')})
            all_features.append(feats)
    print(f"Extraction: ok={n_ok}  fail_ts={n_fail_ts}  "
          f"fail_pose={n_fail_pose}  fail_kp={n_fail_kp}")
    if n_ok == 0:
        print(f"ERROR [{name}]: No features extracted. Skipping."); return

    feat_df = pd.DataFrame(all_features)
    feat_df.to_csv(os.path.join(OUTPUT_DIR, 'clip_level_features.csv'), index=False)

    META_COLS = {'pid','Group','age_mo','age_band','original_label','label_lower',
                 'clip','n_valid_frames','n_total_frames','pct_valid',
                 'duration_sec','mean_conf','mean_scale'}
    FEAT_COLS = [c for c in feat_df.columns if c not in META_COLS]

    PRIMARY_FEATS = [f for f in FEAT_COLS if any(x in f for x in [
        'mean_hip_y_amplitude','mean_hip_dom_freq','mean_hip_spectral_entropy',
        'mean_hip_band_power','mean_hip_vel','mean_hip_acc',
        'vertical_excursion_hip','vertical_excursion_ankle',
        'hip_ankle_amp_ratio','hip_knee_amp_ratio',
        'hip_y_L_amplitude','hip_y_R_amplitude',
        'hip_y_L_vel_mean','hip_y_R_vel_mean',
        'hip_y_L_acc_mean','hip_y_R_acc_mean',
        'hip_y_L_dom_freq','hip_y_R_dom_freq',
        'hip_y_L_band_power','hip_y_R_band_power',
        'ankle_y_L_mean','ankle_y_R_mean',
        'ankle_y_L_vel_mean','ankle_y_R_vel_mean',
        'ankle_y_L_acc_mean','ankle_y_R_acc_mean',
        'shoulder_y_R_dom_freq','shoulder_y_R_band_power',
        'shoulder_y_L_dom_freq','shoulder_y_L_vel_mean',
        'shoulder_y_R_vel_mean',
        'bilateral_hip_y_corr','bilateral_hip_sym_index',
        'bilateral_hip_phase_lag','bilateral_hip_amp_diff',
        'bilateral_knee_y_corr','bilateral_knee_sym_index',
        'bilateral_ankle_y_corr','bilateral_ankle_sym_index',
        'bilateral_ankle_phase_lag',
    ])]
    PRIMARY_FEATS = [f for f in PRIMARY_FEATS if f in feat_df.columns]

    child_df = feat_df.groupby(['pid','Group'])[FEAT_COLS].mean().reset_index()
    child_df['n_clips']  = feat_df.groupby(['pid','Group']).size().values
    child_df['age_mo']   = feat_df.groupby(['pid','Group'])['age_mo'].first().values
    child_df['age_band'] = feat_df.groupby(['pid','Group'])['age_band'].first().values
    child_df['label_lower'] = feat_df.groupby(['pid','Group'])['label_lower'].agg(
        lambda x: x.mode()[0]).values
    child_df.to_csv(os.path.join(OUTPUT_DIR, 'child_level_features.csv'), index=False)
    CHILD_FEATS = [f for f in PRIMARY_FEATS if f in child_df.columns]

    # ── PART 2: FULL STATISTICAL BATTERY ───────────────────────────
    hr(f"PART 2: STATISTICAL BATTERY  [{name}]")

    # 2a: ICC
    print("\n--- 2a: ICC ---")
    icc_df = compute_icc(feat_df, PRIMARY_FEATS)
    icc_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_icc.csv'), index=False)
    print(icc_df.head(10).to_string(index=False))
    print(f"  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)} features")

    # 2b: MWU clip level
    print("\n--- 2b: MWU clip level ---")
    res_clip = run_mwu_comparison(feat_df, PRIMARY_FEATS, level='clip',
                                  subset_label='combined')
    res_clip.to_csv(os.path.join(OUTPUT_DIR, 'stats_combined_clip.csv'), index=False)
    if len(res_clip):
        print(f"  sig_raw: {res_clip['sig_raw05'].sum()}  FDR: {res_clip['sig_fdr05'].sum()}")

    # 2c: MWU child level
    print("\n--- 2c: MWU child level (pseudo-bulk) ---")
    res_child = run_mwu_comparison(child_df, CHILD_FEATS, level='child',
                                   subset_label='combined')
    res_child.to_csv(os.path.join(OUTPUT_DIR, 'stats_combined_child.csv'), index=False)
    if len(res_child):
        print(f"  sig_raw: {res_child['sig_raw05'].sum()}  FDR: {res_child['sig_fdr05'].sum()}")

    # 2d: LME + KR
    print("\n--- 2d: LME + Kenward-Roger ---")
    lme_all = run_lme_kr(feat_df, PRIMARY_FEATS, subset_label='full')
    if len(lme_all):
        lme_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_kr.csv'), index=False)
        print(f"  sig_raw: {lme_all['sig_raw05'].sum()}  FDR: {lme_all['sig_fdr05'].sum()}"
              f"  method: {lme_all['method'].mode()[0]}")

    # 2e: CR2
    print("\n--- 2e: CR2 (wildboottest) ---")
    cr2_all = run_cr2(feat_df, PRIMARY_FEATS, subset_label='full')
    if len(cr2_all):
        cr2_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_cr2.csv'), index=False)
        print(f"  sig_raw: {cr2_all['sig_raw05'].sum()}  FDR: {cr2_all['sig_fdr05'].sum()}")

    # 2f: GEE
    print("\n--- 2f: GEE ---")
    gee_all = run_gee(feat_df, PRIMARY_FEATS, subset_label='full')
    if len(gee_all):
        gee_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_gee.csv'), index=False)
        print(f"  sig_raw: {gee_all['sig_raw05'].sum()}  FDR: {gee_all['sig_fdr05'].sum()}")

    # 2g: Child permutation
    print("\n--- 2g: Child-level permutation ---")
    perm_all = run_child_permutation(child_df, CHILD_FEATS, n_perm=5000,
                                     subset_label='full')
    if len(perm_all):
        perm_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_child_permutation.csv'), index=False)
        print(f"  sig_raw: {perm_all['sig_raw05'].sum()}  FDR: {perm_all['sig_fdr05'].sum()}")

    # 2h: Wild cluster bootstrap
    print("\n--- 2h: Wild cluster bootstrap ---")
    boot_all = run_wild_bootstrap(child_df, CHILD_FEATS, n_boot=5000,
                                  subset_label='full')
    if len(boot_all):
        boot_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_wild_bootstrap.csv'), index=False)
        print(f"  sig_raw: {boot_all['sig_raw05'].sum()}  FDR: {boot_all['sig_fdr05'].sum()}")

    # 2i: Consensus
    print("\n--- 2i: Consensus across methods ---")
    all_results = {
        'LME_KR':   lme_all if len(lme_all) else None,
        'CR2':      cr2_all if len(cr2_all) else None,
        'GEE':      gee_all if len(gee_all) else None,
        'ChildPerm': perm_all if len(perm_all) else None,
        'WildBoot': boot_all if len(boot_all) else None,
        'MWU_child': res_child if len(res_child) else None,
    }
    consensus_all = make_consensus(
        {k: v for k, v in all_results.items() if v is not None},
        PRIMARY_FEATS)
    consensus_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_consensus.csv'), index=False)
    top_cons = consensus_all[consensus_all['n_methods_sig'] > 0].head(10)
    for _, r in top_cons.iterrows():
        print(f"  {r['feature']:<45} n_sig={r['n_methods_sig']}")

    # 2j: Consistency gate
    print("\n--- 2j: Consistency gate across jump labels ---")
    sig_feats = list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else []
    cons_df, consistent_feats, label_mwu_dict = run_consistency_gate(
        feat_df, PRIMARY_FEATS, sig_feats)
    if len(cons_df):
        cons_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_consistency_gate.csv'), index=False)
        print(f"  Passed: {len(consistent_feats)}/{len(sig_feats)}")
        for f in consistent_feats: print(f"    ✓ {f}")

    # 2k: Label × Group interaction
    print("\n--- 2k: Label × Group interaction ---")
    lgi_all = run_label_group_interaction(feat_df, PRIMARY_FEATS, 'full')
    if len(lgi_all):
        lgi_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_label_group_interaction.csv'),
                       index=False)
        n_sig_lgi = lgi_all['sig_raw05'].sum()
        if n_sig_lgi > 0:
            print(f"  ⚠ Label×Group interactions (n={n_sig_lgi}) — some Group effects "
                  f"differ by jump label:")
            for _, r in lgi_all[lgi_all['sig_raw05']].head(5).iterrows():
                print(f"    {r['feature']:<40} {r['interaction_term']}  p={r['p_raw']:.4f}")

    # 2l: Per-label MWU (behavior-stratified)
    print("\n--- 2l: Per-label MWU ---")
    label_mwu_results = {}
    for lbl in sorted(JUMP_LABELS):
        sub = feat_df[feat_df['label_lower'] == lbl]
        asd_n = sub[sub['Group']=='ASD']['pid'].nunique()
        nan_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        print(f"  {lbl}: ASD={asd_n}, Non-ASD={nan_n}", end='')
        if asd_n >= 3 and nan_n >= 3:
            r = run_mwu_comparison(sub, [f for f in PRIMARY_FEATS if f in sub.columns],
                                   level='clip', subset_label=lbl)
            label_mwu_results[lbl] = r
            print(f" → sig_raw: {r['sig_raw05'].sum()}  FDR: {r['sig_fdr05'].sum()}")
        else:
            print(" → too few")
    if label_mwu_results:
        pd.concat(label_mwu_results.values(), ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, 'stats_per_label_mwu.csv'), index=False)

    # 2m: Age-stratified MWU
    print("\n--- 2m: Age-stratified MWU ---")
    age_strat_results = []
    for band in AGE_BANDS.keys():
        sub_clip  = feat_df[feat_df['age_band'] == band]
        asd_n = sub_clip[sub_clip['Group']=='ASD']['pid'].nunique()
        nan_n = sub_clip[sub_clip['Group']=='Non-ASD']['pid'].nunique()
        print(f"  {band}: ASD={asd_n}, Non-ASD={nan_n}")
        if asd_n < 3 or nan_n < 3: print("    → descriptive only"); continue
        r = run_mwu_comparison(sub_clip, PRIMARY_FEATS, level='clip', subset_label=band)
        r['age_band'] = band; age_strat_results.append(r)
        print(f"    sig_raw: {r['sig_raw05'].sum()} FDR: {r['sig_fdr05'].sum()}")
    if age_strat_results:
        pd.concat(age_strat_results, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, 'stats_age_stratified.csv'), index=False)

    # 2n: Within-group trajectories (early vs late)
    print("\n--- 2n: Within-ASD / Within-Non-ASD age trajectory ---")
    for grp in GROUPS:
        early = feat_df[(feat_df['Group']==grp) & (feat_df['age_band']=='11-18mo')]
        late  = feat_df[(feat_df['Group']==grp) & (feat_df['age_band']=='32-38mo')]
        print(f"  {grp}: early={early['pid'].nunique()} late={late['pid'].nunique()}", end='')
        if len(early) >= 3 and len(late) >= 3:
            r = run_mwu_comparison(
                pd.concat([early, late]), PRIMARY_FEATS,
                group_col='age_band', group_a='11-18mo', group_b='32-38mo',
                level='clip', subset_label=f'{grp.replace("-","")}_traj')
            fname = f'stats_within_{grp.replace("-","")}_trajectory.csv'
            r.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
            print(f" → sig_raw: {r['sig_raw05'].sum()}")
        else:
            print(" → too few")

    # 2o: Spearman age correlations
    print("\n--- 2o: Spearman age correlations ---")
    sp_df = run_spearman_age(feat_df, PRIMARY_FEATS)
    if len(sp_df):
        sp_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_spearman_age.csv'), index=False)
        print(f"  Significant (p<0.05): {sp_df['sig_p05'].sum()}")

    # 2p: Kruskal-Wallis age × group
    print("\n--- 2p: Kruskal-Wallis age×group ---")
    feat_df['age_group_cell'] = (feat_df['Group'].str.replace('-','') + '_' +
                                  feat_df['age_band'].fillna('unknown'))
    cells = feat_df['age_group_cell'].value_counts()
    kw_records = []
    for feat in PRIMARY_FEATS:
        groups_data = [feat_df[feat_df['age_group_cell']==cell][feat].dropna().values
                       for cell in cells.index
                       if len(feat_df[feat_df['age_group_cell']==cell][feat].dropna()) >= 3]
        if len(groups_data) < 2: continue
        try:
            stat, p = stats.kruskal(*groups_data)
            kw_records.append({'feature': feat, 'kw_stat': stat, 'p_raw': p})
        except: pass
    if kw_records:
        kw_df = pd.DataFrame(kw_records).sort_values('p_raw')
        if len(kw_df) > 1:
            _, p_fdr, _, _ = multipletests(kw_df['p_raw'], method='fdr_bh')
            kw_df['p_fdr'] = p_fdr
        kw_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_age_group_kruskal.csv'), index=False)
        print(f"  sig_raw: {(kw_df['p_raw']<0.05).sum()}")

    # ── PART 3: CLASSIFICATION ─────────────────────────────────────
    hr(f"PART 3: CLASSIFICATION  [{name}]")
    clf_results = {}
    for subset_name, df_, fc in [
        ('combined_clip',  feat_df,  PRIMARY_FEATS),
        ('combined_child', child_df, CHILD_FEATS),
    ]:
        print(f"\n--- {subset_name} ---")
        r = run_loso_cv(df_, fc)
        if r: clf_results[subset_name] = r

    for band in AGE_BANDS.keys():
        sub = child_df[child_df['age_band']==band]
        asd_n = (sub['Group']=='ASD').sum(); nan_n = (sub['Group']=='Non-ASD').sum()
        print(f"\n--- {band} child (ASD={asd_n}, Non-ASD={nan_n}) ---")
        if asd_n >= 4 and nan_n >= 4:
            r = run_loso_cv(sub, CHILD_FEATS)
            if r: clf_results[f'{band}_child'] = r
        else: print("  → skipped")

    if clf_results:
        pd.DataFrame([{'subset': k, 'auc': v['auc'], 'perm_p': v['perm_p'],
                       'n_features': v['n_features'], 'n_subjects': v['n_subjects']}
                      for k, v in clf_results.items()]
                     ).to_csv(os.path.join(OUTPUT_DIR, 'classification_summary.csv'),
                              index=False)

    # ── PART 4: FIGURES ────────────────────────────────────────────
    hr(f"PART 4: FIGURES  [{name}]")

    # Fig 1: sample overview
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(f'Figure 1: Sample Overview — {name}', fontweight='bold')
    ax = axes[0]
    gc = child_df['Group'].value_counts()
    bars = ax.bar(GROUPS, [gc.get(g,0) for g in GROUPS],
                  color=[COLORS[g] for g in GROUPS], width=0.5, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                str(int(bar.get_height())), ha='center', fontweight='bold')
    ax.set_title('(a) Children per group'); ax.set_ylabel('N')
    ax = axes[1]
    for grp in GROUPS:
        ax.hist(child_df[child_df['Group']==grp]['n_clips'],
                bins=range(1, int(child_df['n_clips'].max())+2),
                alpha=0.6, color=COLORS[grp], label=grp, edgecolor='white')
    ax.set_title('(b) Clips per child'); ax.set_xlabel('N clips'); ax.legend()
    ax = axes[2]
    for grp in GROUPS:
        ax.hist(child_df[child_df['Group']==grp]['age_mo'],
                bins=10, alpha=0.6, color=COLORS[grp], label=grp, edgecolor='white')
    for (lo,hi), col, alpha in [((11,18),'#7B5EA7',0.12),
                                  ((19,31),'#4A9B6F',0.08),
                                  ((32,38),'#D47C2A',0.12)]:
        ax.axvspan(lo, hi, alpha=alpha, color=col)
    ax.set_title('(c) Age distribution'); ax.set_xlabel('Age (months)'); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig1_sample_overview.png')); plt.close(fig)

    # Fig 2: violin plots
    DISP_FEATS = [(f,l) for f,l in [
        ('mean_hip_y_amplitude',          'Hip Amplitude (mean)'),
        ('vertical_excursion_hip_max',    'Hip Excursion (max)'),
        ('mean_hip_dom_freq',             'Hip Dom. Freq (Hz)'),
        ('mean_hip_band_power_1_4hz',     'Hip Power 1-4 Hz'),
        ('mean_hip_spectral_entropy',     'Hip Spectral Entropy'),
        ('bilateral_hip_y_corr',          'Bilateral Hip Corr'),
        ('bilateral_hip_sym_index',       'Bilateral Hip Symmetry'),
        ('bilateral_knee_sym_index',      'Bilateral Knee Symmetry'),
        ('ankle_y_L_vel_mean',            'Ankle L Velocity'),
        ('shoulder_y_R_band_power_1_4hz', 'Shoulder R Power 1-4Hz'),
    ] if f in feat_df.columns]
    if DISP_FEATS and len(res_clip):
        ncols = 4; nrows = int(np.ceil(len(DISP_FEATS)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        fig.suptitle(f'Figure 2: Jumping Kinematics — ASD vs Non-ASD — {name}',
                     fontweight='bold')
        axes = axes.flatten()
        for i, (feat, label) in enumerate(DISP_FEATS):
            ax = axes[i]
            dg = [feat_df.loc[feat_df['Group']==g, feat].dropna().values for g in GROUPS]
            if any(len(d)==0 for d in dg): ax.set_visible(False); continue
            parts = ax.violinplot(dg, positions=[0,1], showmedians=True, showextrema=False)
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
            parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
            for j, vals in enumerate(dg):
                ax.scatter(j+np.random.uniform(-0.07,0.07,len(vals)), vals,
                           color=list(COLORS.values())[j], alpha=0.2, s=8, zorder=3)
            row = res_clip[res_clip['feature']==feat]
            if len(row):
                p_r = row['p_raw'].values[0]; p_f = row['p_fdr'].values[0]
                d   = row['cohens_d'].values[0]
                col = '#cc0000' if p_f<0.05 else ('#ff8800' if p_r<0.05 else 'gray')
                ax.text(0.5, 0.97, f'p={p_r:.3f}|FDR={p_f:.3f}|d={d:.2f}',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=7.5, color=col)
                ymax = max(np.percentile(d2,95) for d2 in dg if len(d2))
                yr   = ymax - min(np.percentile(d2,5) for d2 in dg if len(d2))
                add_sig_bar(ax, 0, 1, ymax+yr*0.05, p_r, h=yr*0.04)
            ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
            ax.set_title(label, fontsize=9, fontweight='bold')
        for j in range(len(DISP_FEATS), len(axes)): axes[j].set_visible(False)
        fig.legend(handles=[mpatches.Patch(color=COLORS[g],label=g) for g in GROUPS],
                   loc='upper right')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig2_violins.png')); plt.close(fig)
        print("  Saved fig2_violins.png")

    # Fig 3: effect size forest
    if len(res_clip):
        ht = res_clip.copy()
        ht['label'] = ht['feature'].map(SHORT_LABELS).fillna(ht['feature'])
        ht = ht.reindex(ht['cohens_d'].abs().sort_values(ascending=True).index)
        fig, ax = plt.subplots(figsize=(11, max(6, len(ht)*0.38)))
        ax.barh(ht['label'], ht['cohens_d'],
                color=[ASD_COLOR if d>0 else NONASD_COLOR for d in ht['cohens_d']],
                edgecolor='white', height=0.6)
        if 'd_ci_lo' in ht.columns:
            ax.errorbar(ht['cohens_d'], range(len(ht)),
                        xerr=[ht['cohens_d']-ht['d_ci_lo'], ht['d_ci_hi']-ht['cohens_d']],
                        fmt='none', color='black', lw=1.2, capsize=3, alpha=0.7)
        ax.axvline(0, color='black', lw=0.8)
        for t, ls in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
            for sign in [1,-1]: ax.axvline(sign*t, color='gray', lw=0.7, ls=ls, alpha=0.5)
        for j, (_, row) in enumerate(ht.iterrows()):
            if row.get('sig_fdr05'):
                ci_hi = row.get('d_ci_hi', row['cohens_d'])
                ax.text(ci_hi+0.02, j, '★', va='center', fontsize=11, color='gold')
            elif row.get('sig_raw05'):
                ci_hi = row.get('d_ci_hi', row['cohens_d'])
                ax.text(ci_hi+0.02, j, '●', va='center', fontsize=9)
        ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)")
        ax.set_title(f"Figure 3: Effect Sizes — {name}\n"
                     "★=FDR q<0.05  ●=raw p<0.05  bars=95% bootstrap CI",
                     fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig3_effect_sizes.png')); plt.close(fig)
        print("  Saved fig3_effect_sizes.png")

    # Fig 4: consensus heatmap (NEW)
    if len(consensus_all) > 0:
        p_cols = [c for c in consensus_all.columns if c.startswith('p_')]
        heat_data = consensus_all.set_index('feature')[p_cols].head(20)
        heat_log  = -np.log10(heat_data.clip(lower=1e-5, upper=1.0).astype(float))
        fig, ax   = plt.subplots(figsize=(len(p_cols)*2+2, max(6, len(heat_data)*0.4)))
        im = ax.imshow(heat_log.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=4)
        ax.set_xticks(range(len(p_cols)))
        ax.set_xticklabels([c.replace('p_','') for c in p_cols], rotation=30, ha='right')
        ax.set_yticks(range(len(heat_data)))
        ax.set_yticklabels([SHORT_LABELS.get(f,f) for f in heat_data.index], fontsize=9)
        for i in range(heat_log.shape[0]):
            for j in range(heat_log.shape[1]):
                raw_p = heat_data.values[i,j]
                ax.text(j, i, f'{raw_p:.3f}{"*" if raw_p<0.05 else ""}',
                        ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, label='-log10(p)')
        ax.set_title(f'Figure 4: Consensus p-values — {name} (top 20 features)',
                     fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig4_consensus_heatmap.png')); plt.close(fig)
        print("  Saved fig4_consensus_heatmap.png")

    # Fig 5: ICC bar (NEW)
    if len(icc_df) > 0:
        top_icc = icc_df.head(20)
        fig, ax  = plt.subplots(figsize=(10, max(5, len(top_icc)*0.4)))
        colors_icc = ['#2ecc71' if v>0.10 else '#e74c3c' for v in top_icc['ICC']]
        ax.barh(top_icc['feature'].map(SHORT_LABELS).fillna(top_icc['feature']),
                top_icc['ICC'], color=colors_icc, edgecolor='white', height=0.65)
        ax.axvline(0.10, color='orange', lw=1.5, ls='--', label='ICC=0.10 threshold')
        ax.set_xlabel('ICC')
        ax.set_title(f'Figure 5: Intraclass Correlation — {name}\n'
                     'Green = significant within-child clustering', fontweight='bold')
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig5_icc.png')); plt.close(fig)
        print("  Saved fig5_icc.png")

    # Fig 6: consistency gate (NEW)
    if len(cons_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(cons_df)*0.45)))
        cols_cg = [ASD_COLOR if v else NONASD_COLOR for v in cons_df['consistent']]
        ax.barh(
            cons_df['feature'].map(SHORT_LABELS).fillna(cons_df['feature']),
            cons_df['n_same_direction'] / cons_df['n_labels_tested'],
            color=cols_cg, edgecolor='white', height=0.6)
        ax.axvline(1.0, color='green', lw=1.5, ls='--', label='All consistent')
        ax.axvline(0.5, color='orange', lw=1, ls=':', label='50%')
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Fraction of jump labels with same effect direction')
        ax.set_title(f'Figure 6: Consistency Gate — {name}\n'
                     'Red = failed (effect reverses across labels)', fontweight='bold')
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig6_consistency_gate.png')); plt.close(fig)
        print("  Saved fig6_consistency_gate.png")

    # Fig 7: developmental trajectories
    TRAJ_FEATS = [f for f in ['mean_hip_y_amplitude','mean_hip_dom_freq',
                               'bilateral_hip_y_corr','mean_hip_band_power_1_4hz']
                  if f in feat_df.columns]
    if TRAJ_FEATS:
        ncols = min(len(TRAJ_FEATS), 2)
        nrows = int(np.ceil(len(TRAJ_FEATS)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
        fig.suptitle(f'Figure 7: Developmental Trajectories — {name}', fontweight='bold')
        axes = np.array(axes).flatten()
        for i, feat in enumerate(TRAJ_FEATS):
            ax = axes[i]
            for grp in GROUPS:
                sub = feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
                if len(sub) < 3: continue
                ax.scatter(sub['age_mo'], sub[feat], color=COLORS[grp], alpha=0.3, s=15)
                if len(sub) >= 5:
                    m, b, r, p, _ = stats.linregress(sub['age_mo'], sub[feat])
                    xr = np.linspace(sub['age_mo'].min(), sub['age_mo'].max(), 100)
                    ax.plot(xr, m*xr+b, color=COLORS[grp], lw=2.5,
                            label=f'{grp}  r={r:.2f}  p={p:.3f}')
            for band, (lo,hi) in AGE_BANDS.items():
                ax.axvspan(lo, hi, alpha=0.07, color=BAND_COLORS[band])
            ax.set_xlabel('Age (months)')
            ax.set_ylabel(SHORT_LABELS.get(feat, feat)[:20], fontsize=9)
            ax.set_title(SHORT_LABELS.get(feat, feat), fontsize=9, fontweight='bold')
            ax.legend(fontsize=8)
        for j in range(len(TRAJ_FEATS), len(axes)): axes[j].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig7_trajectories.png')); plt.close(fig)
        print("  Saved fig7_trajectories.png")

    # Fig 8: bilateral coordination
    BILAT_FEATS = [f for f in ['bilateral_hip_y_corr','bilateral_hip_sym_index',
                                'bilateral_hip_phase_lag','bilateral_knee_y_corr',
                                'bilateral_knee_sym_index','bilateral_ankle_y_corr',
                                'bilateral_ankle_sym_index']
                   if f in feat_df.columns]
    if BILAT_FEATS:
        ncols = min(4, len(BILAT_FEATS))
        nrows = int(np.ceil(len(BILAT_FEATS)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        fig.suptitle(f'Figure 8: Bilateral Coordination — {name}', fontweight='bold')
        axes = np.array(axes).flatten()
        for i, feat in enumerate(BILAT_FEATS):
            ax = axes[i]
            dg = [feat_df.loc[feat_df['Group']==g, feat].dropna().values for g in GROUPS]
            if any(len(d)==0 for d in dg): ax.set_visible(False); continue
            parts = ax.violinplot(dg, positions=[0,1], showmedians=True, showextrema=False)
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
            parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
            for j, vals in enumerate(dg):
                ax.scatter(j+np.random.uniform(-0.07,0.07,len(vals)), vals,
                           color=list(COLORS.values())[j], alpha=0.3, s=8, zorder=3)
            if len(dg[0])>=3 and len(dg[1])>=3:
                _, p = stats.mannwhitneyu(dg[0], dg[1], alternative='two-sided')
                ymax = max(np.percentile(d,95) for d in dg if len(d))
                yr   = abs(ymax - min(np.percentile(d,5) for d in dg if len(d)))
                add_sig_bar(ax, 0, 1, ymax+yr*0.05, p, h=yr*0.04)
            ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
            ax.set_title(SHORT_LABELS.get(feat, feat.replace('_',' ')), fontsize=9)
        for j in range(len(BILAT_FEATS), len(axes)): axes[j].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig8_bilateral.png')); plt.close(fig)
        print("  Saved fig8_bilateral.png")

    # Fig 9: child boxplots
    CHILD_BOX = [f for f in ['mean_hip_y_amplitude','mean_hip_dom_freq',
                              'bilateral_hip_y_corr','ankle_y_L_vel_mean']
                 if f in child_df.columns]
    if CHILD_BOX:
        fig, axes = plt.subplots(1, len(CHILD_BOX), figsize=(4*len(CHILD_BOX), 5))
        if len(CHILD_BOX) == 1: axes = [axes]
        fig.suptitle(f'Figure 9: Child-Level Averages — {name}', fontweight='bold')
        for i, feat in enumerate(CHILD_BOX):
            ax = axes[i]
            for j, grp in enumerate(GROUPS):
                vals = child_df[child_df['Group']==grp][feat].dropna().values
                if len(vals) == 0: continue
                bp = ax.boxplot(vals, positions=[j], widths=0.45, patch_artist=True,
                                showfliers=False,
                                medianprops={'color':'black','linewidth':2})
                bp['boxes'][0].set_facecolor(COLORS_LIGHT[grp])
                bp['boxes'][0].set_edgecolor(COLORS[grp])
                bp['boxes'][0].set_linewidth(1.5)
                ax.scatter(j+np.random.uniform(-0.12,0.12,len(vals)), vals,
                           color=COLORS[grp], alpha=0.65, s=28, zorder=4)
            da = child_df[child_df['Group']=='ASD'][feat].dropna().values
            dn = child_df[child_df['Group']=='Non-ASD'][feat].dropna().values
            if len(da)>=3 and len(dn)>=3:
                _, p = stats.mannwhitneyu(da, dn, alternative='two-sided')
                ymax = np.percentile(child_df[feat].dropna().values, 97)
                add_sig_bar(ax, 0, 1, ymax, p, h=abs(ymax)*0.04)
            ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
            ax.set_title(SHORT_LABELS.get(feat, feat.replace('_',' ')),
                         fontsize=9, fontweight='bold')
            ax.text(0.5,-0.12, f'n={len(da)}/{len(dn)}',
                    transform=ax.transAxes, ha='center', fontsize=8, color='gray')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig9_child_boxplots.png')); plt.close(fig)
        print("  Saved fig9_child_boxplots.png")

    # Fig 10: ROC
    if clf_results:
        n = len(clf_results)
        ncols = min(n, 4); nrows = int(np.ceil(n/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        if nrows*ncols == 1: axes = np.array([[axes]])
        elif nrows == 1:     axes = axes.reshape(1,-1)
        fig.suptitle(f'Figure 10: Classification ROC — {name}', fontweight='bold')
        for i, (cname, r) in enumerate(clf_results.items()):
            ax = axes[i//ncols][i%ncols]
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_score'])
            ax.plot(fpr, tpr, color=ASD_COLOR, lw=2,
                    label=f"AUC={r['auc']:.3f}  p={r['perm_p']:.3f}")
            ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.12, color=ASD_COLOR)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'{cname}', fontsize=8); ax.legend(fontsize=8)
            if r.get('perm_aucs'):
                axins = ax.inset_axes([0.55,0.05,0.4,0.28])
                axins.hist(r['perm_aucs'], bins=20, color='gray', alpha=0.7)
                axins.axvline(r['auc'], color=ASD_COLOR, lw=2)
                axins.set_title('Null', fontsize=6); axins.tick_params(labelsize=5)
        for i in range(len(clf_results), nrows*ncols):
            axes[i//ncols][i%ncols].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'fig10_roc.png')); plt.close(fig)
        print("  Saved fig10_roc.png")

    # ── PART 5: SUMMARY ────────────────────────────────────────────
    hr(f"PART 5: SUMMARY  [{name}]")
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print("\n--- CSVs ---")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith('.csv'):
            try:
                tmp = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
                print(f"  {fname:<65}  {tmp.shape[0]:>5} rows × {tmp.shape[1]:>3} cols")
            except: print(f"  {fname}")
    print("\n--- Figures ---")
    for fname in sorted(os.listdir(FIGURE_DIR)):
        if fname.endswith('.png'):
            sz = os.path.getsize(os.path.join(FIGURE_DIR,fname))/1024
            print(f"  {fname:<55} {sz:.0f} KB")
    print("\n--- KEY RESULTS ---")
    for name_, res in [('LME_KR', lme_all), ('ChildPerm', perm_all),
                       ('WildBoot', boot_all), ('MWU_child', res_child)]:
        if len(res):
            print(f"  {name_}: sig_raw={res['sig_raw05'].sum()}  "
                  f"FDR={res['sig_fdr05'].sum()}")
    print(f"\n  Consistency gate: {len(consistent_feats)}/{len(sig_feats)} passed")
    for f in consistent_feats: print(f"    ✓ {f}")
    if clf_results:
        print("\n  Classification (LOSO):")
        for k, v in clf_results.items():
            print(f"    {k:<40} AUC={v['auc']:.3f}  p_perm={v['perm_p']:.4f}")

    # ── PART 6: LME SUITE ─────────────────────────────────────────
    run_lme_suite(feat_df, child_df, PRIMARY_FEATS, OUTPUT_DIR, name)

    # ── PART 7: BAYESIAN SUITE ─────────────────────────────────────
    run_bayesian_suite(feat_df, PRIMARY_FEATS, OUTPUT_DIR, name)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
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

rmm = pd.read_csv(RMM_CSV)
rmm['pid'] = rmm['csv_bids_processed'].apply(extract_pid)
rmm = rmm.merge(pid_info, on='pid', how='left')
rmm_labeled = rmm[rmm['Group'].isin(['ASD','Non-ASD'])].copy()
rmm_labeled['label_lower'] = rmm_labeled['matched_label'].str.strip().str.lower()

jump_rmm_all = rmm_labeled[rmm_labeled['label_lower'].isin(ALL_JUMP_LABELS)].copy()
jump_rmm_all['hrnet_path'] = jump_rmm_all['csv_bids_processed'].map(video_to_hrnet)
jump_rmm_all = jump_rmm_all[
    jump_rmm_all['hrnet_path'].apply(
        lambda p: isinstance(p,str) and os.path.isfile(p))
].copy()
jump_rmm_all['age_band'] = jump_rmm_all['age_mo'].apply(assign_age_band)

print(f"\nTotal jumping clips with pose file: {len(jump_rmm_all)}")
print(f"  ASD:     {len(jump_rmm_all[jump_rmm_all['Group']=='ASD'])}")
print(f"  Non-ASD: {len(jump_rmm_all[jump_rmm_all['Group']=='Non-ASD'])}")
print(jump_rmm_all.groupby(['label_lower','Group']).size()
      .reset_index(name='n').to_string(index=False))

for variant in VARIANTS:
    run_variant(variant, jump_rmm_all)

hr("ALL VARIANTS COMPLETE")
print(f"\nOutputs written under: {BASE_DIR}")