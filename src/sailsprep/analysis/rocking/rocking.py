#!/usr/bin/env python3
"""
rocking analysis
"""

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
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Optional dependencies ─────────────────────────────────────────
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
# CONFIG
# ═══════════════════════════════════════════════════════════════════
MAIN_CSV   = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
RMM_CSV    = "/home/aparnabg/orcd/scratch/all_project_files/phase_2_analyais/clip_to_csv_matching.csv"
OUTPUT_DIR = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/analysis/rocking/v3"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

FPS      = 15.0
MIN_CONF = 0.3

ROCKING_LABELS = {'rocking'}

AGE_BANDS = {
    '11-18mo': (11, 18),
    '19-31mo': (19, 31),
    '32-38mo': (32, 38),
}
AGE_BINS   = [(11, 18), (19, 25), (26, 31), (32, 38)]
STAT_BANDS = ['11-18mo', '32-38mo']

# Bayesian config
BAYES_DRAWS  = 4000
BAYES_TUNE   = 2000
BAYES_CHAINS = 4
RUN_BAYESIAN = True
PRIOR_SDS    = [0.3, 0.5, 1.0]   # prior sensitivity sweep

KP = {
    'left_shoulder':  'kp_005', 'right_shoulder': 'kp_006',
    'left_elbow':     'kp_007', 'right_elbow':    'kp_008',
    'left_wrist':     'kp_009', 'right_wrist':    'kp_010',
    'left_hip':       'kp_011', 'right_hip':      'kp_012',
    'left_knee':      'kp_013', 'right_knee':     'kp_014',
    'nose':           'kp_000',
}

ASD_COLOR    = '#E05C5C'; NONASD_COLOR = '#5B8DB8'
ASD_LIGHT    = '#F2AEAE'; NONASD_LIGHT = '#A8C8E8'
COLORS       = {'ASD': ASD_COLOR, 'Non-ASD': NONASD_COLOR}
COLORS_LIGHT = {'ASD': ASD_LIGHT, 'Non-ASD': NONASD_LIGHT}
BAND_COLORS  = {'11-18mo': '#7B5EA7', '19-31mo': '#4A9B6F', '32-38mo': '#D47C2A'}
GROUPS       = ['ASD', 'Non-ASD']

SHORT_LABELS = {
    'mean_hip_x_amplitude':            'Hip X Amp',
    'mean_hip_x_vel_mean':             'Hip X Vel',
    'mean_hip_x_vel_max':              'Hip X Vel Max',
    'mean_hip_x_acc_mean':             'Hip X Acc',
    'mean_hip_x_dom_freq':             'Hip X Dom Freq',
    'mean_hip_x_spectral_entropy':     'Hip X Entropy',
    'mean_hip_x_band_power_0p3_2hz':   'Hip X 0.3-2Hz',
    'trunk_tilt_amplitude':            'Trunk Tilt Amp',
    'trunk_tilt_std':                  'Trunk Tilt Std',
    'trunk_tilt_band_power_0p3_2hz':   'Trunk Tilt Power',
    'trunk_tilt_spectral_entropy':     'Trunk Tilt Entropy',
    'nose_x_amplitude':                'Nose X Amp',
    'nose_x_dom_freq':                 'Nose X Freq',
    'nose_x_spectral_entropy':         'Nose X Entropy',
    'hip_2d_amplitude_max':            'Hip 2D Max',
    'bilateral_hip_x_corr':            'Bilat Hip Corr',
    'bilateral_hip_x_sym':             'Bilat Hip Sym',
    'hip_x_y_ratio':                   'Hip X/Y Ratio',
    'sh_x_L_amplitude':                'Shoulder L X Amp',
    'sh_x_R_amplitude':                'Shoulder R X Amp',
    'hip_x_L_amplitude':               'Hip L X Amp',
    'hip_x_R_amplitude':               'Hip R X Amp',
    'hip_y_L_band_power_0p3_2hz':      'Hip L Y Power',
    'hip_y_R_band_power_0p3_2hz':      'Hip R Y Power',
    'hip_x_L_band_power_0p3_2hz':      'Hip L X Power',
    'hip_x_R_band_power_0p3_2hz':      'Hip R X Power',
    'mean_sh_x_amplitude':             'Shoulder X Amp',
    'mean_sh_x_spectral_entropy':      'Shoulder X Entropy',
}

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

def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURE_DIR, name)); plt.close(fig)
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
            if e > s: segs.append((int(s*fps), int(e*fps)))
    return segs

def get_kp(fd, key, min_conf=MIN_CONF):
    if key not in fd: return None
    kp = fd[key]
    if not isinstance(kp, dict): return None
    if kp.get('confidence', 0) < min_conf: return None
    return kp

def butter_lp(data, cutoff=4.0, fs=15.0, order=2):
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
    sx = (ls['x']+rs['x'])/2; sy = (ls['y']+rs['y'])/2
    hx = (lh['x']+rh['x'])/2; hy = (lh['y']+rh['y'])/2
    d  = np.sqrt((sx-hx)**2 + (sy-hy)**2)
    return d if d > 5 else None

def get_scale(fd):
    tl = torso_length(fd)
    if tl: return tl
    lh = get_kp(fd, KP['left_hip'],  min_conf=0.1)
    rh = get_kp(fd, KP['right_hip'], min_conf=0.1)
    if lh and rh:
        d = np.sqrt((lh['x']-rh['x'])**2+(lh['y']-rh['y'])**2)
        if d > 5: return d
    ls = get_kp(fd, KP['left_shoulder'],  min_conf=0.1)
    rs = get_kp(fd, KP['right_shoulder'], min_conf=0.1)
    if ls and rs:
        d = np.sqrt((ls['x']-rs['x'])**2+(ls['y']-rs['y'])**2)
        if d > 5: return d
    return None

def spectral_features(arr, fps, lo=0.3, hi=2.0):
    if len(arr) < 16: return np.nan, np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq  = freqs[np.argmax(psd)]
        psd_n     = psd / (psd.sum()+1e-12)
        entropy   = -np.sum(psd_n[psd_n>0]*np.log2(psd_n[psd_n>0]))
        band_mask = (freqs >= lo) & (freqs <= hi)
        band_pwr  = psd[band_mask].sum() / (psd.sum()+1e-12)
        return float(dom_freq), float(entropy), float(band_pwr)
    except:
        return np.nan, np.nan, np.nan

def cohen_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1)+np.var(b, ddof=1))/2)
    return (np.mean(a)-np.mean(b))/pooled if pooled > 0 else 0.0

def bootstrap_ci_d(a, b, n_boot=500, seed=42):
    rng  = np.random.default_rng(seed)
    boot = [cohen_d(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def fdr_annotate(df_res, p_col):
    df_res = df_res.copy()
    valid  = df_res[p_col].fillna(1)
    if len(df_res) > 1:
        _, p_fdr, _, _ = multipletests(valid, method='fdr_bh')
        df_res['p_fdr'] = p_fdr
    else:
        df_res['p_fdr'] = valid
    df_res['sig_fdr05'] = df_res['p_fdr'] < 0.05
    df_res['sig_raw05'] = df_res[p_col] < 0.05
    return df_res

def add_sig_bar(ax, x1, x2, y, p, h=0.02):
    label = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col   = '#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], lw=1.2, color='black')
    ax.text((x1+x2)/2, y+h*1.05, label, ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')

def assign_age_band(age_mo):
    for band, (lo, hi) in AGE_BANDS.items():
        if lo <= age_mo <= hi: return band
    return None

def _standardise(series):
    m, s = series.mean(), series.std()
    s = s if s > 1e-10 else 1.0
    return ((series - m) / s).values, m, s

def _savage_dickey_bf(posterior_samples, prior_sd=0.5):
    prior_at_0 = spnorm.pdf(0, 0, prior_sd)
    try:
        post_at_0 = gaussian_kde(posterior_samples)(0)[0]
        return float(prior_at_0 / post_at_0) if post_at_0 > 0 else np.nan
    except:
        return np.nan


# ═══════════════════════════════════════════════════════════════════
# PART 0: LOAD DATA
# ═══════════════════════════════════════════════════════════════════
hr("PART 0: LOAD DATA")

df_main = pd.read_csv(MAIN_CSV)
df_main['pid']    = df_main['video_path'].apply(extract_pid)
df_main['age_mo'] = df_main['Age'] * 12
df_main = df_main[df_main['pid'].notna() &
                  df_main['Group'].isin(['ASD', 'Non-ASD'])].copy()

video_to_hrnet = dict(zip(df_main['video_path'], df_main['hrnet_full_path']))
pid_info = (df_main.dropna(subset=['pid', 'Group'])
            .groupby('pid')
            .agg(Group=('Group', 'first'), age_mo=('age_mo', 'mean'))
            .reset_index())

rmm = pd.read_csv(RMM_CSV)
rmm['pid'] = rmm['csv_bids_processed'].apply(extract_pid)
rmm = rmm.merge(pid_info, on='pid', how='left')
rmm_labeled = rmm[rmm['Group'].isin(['ASD', 'Non-ASD'])].copy()
rmm_labeled['label_lower'] = rmm_labeled['matched_label'].str.strip().str.lower()

rock_rmm = rmm_labeled[rmm_labeled['label_lower'].isin(ROCKING_LABELS)].copy()
rock_rmm['hrnet_path'] = rock_rmm['csv_bids_processed'].map(video_to_hrnet)
rock_rmm = rock_rmm[
    rock_rmm['hrnet_path'].apply(lambda p: isinstance(p, str) and os.path.isfile(p))
].copy()
rock_rmm['age_band'] = rock_rmm['age_mo'].apply(assign_age_band)

print(f"Rocking clips with pose: {len(rock_rmm)}")
print(f"  ASD: {len(rock_rmm[rock_rmm['Group']=='ASD'])}  "
      f"Non-ASD: {len(rock_rmm[rock_rmm['Group']=='Non-ASD'])}")
print(f"\nAge band × Group:")
print(rock_rmm.groupby(['age_band', 'Group']).size()
      .reset_index(name='n').to_string(index=False))
print(f"\nChildren: ASD={rock_rmm[rock_rmm['Group']=='ASD']['pid'].nunique()}  "
      f"Non-ASD={rock_rmm[rock_rmm['Group']=='Non-ASD']['pid'].nunique()}")
print("\n⚠ AGE CONFOUND: ASD-only at 26-31mo and 32-38mo. All models include age covariate.")


# ═══════════════════════════════════════════════════════════════════
# PART 1: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
hr("PART 1: KINEMATIC FEATURE EXTRACTION")

def _jfeats(arr, rec, name, fps):
    a = np.array(arr, dtype=float)
    if len(a) < 5: return
    rec[f'{name}_amplitude'] = float(np.ptp(a))
    rec[f'{name}_std']       = float(np.std(a))
    rec[f'{name}_mean']      = float(np.mean(a))
    rec[f'{name}_iqr']       = float(np.percentile(a,75)-np.percentile(a,25))
    if len(a) >= 8:
        try:
            sm  = butter_lp(a, fs=fps); vel = np.diff(sm)*fps
            rec[f'{name}_vel_mean'] = float(np.mean(np.abs(vel)))
            rec[f'{name}_vel_std']  = float(np.std(vel))
            rec[f'{name}_vel_max']  = float(np.max(np.abs(vel)))
            if len(vel) >= 4:
                acc = np.diff(vel)*fps
                rec[f'{name}_acc_mean'] = float(np.mean(np.abs(acc)))
                rec[f'{name}_acc_max']  = float(np.max(np.abs(acc)))
        except: pass
    df_f, se, bp = spectral_features(a, fps)
    rec[f'{name}_dom_freq']           = df_f
    rec[f'{name}_spectral_entropy']   = se
    rec[f'{name}_band_power_0p3_2hz'] = bp

def extract_rocking_features(pose_frames, frame_indices, ann_fps=FPS):
    hip_x_L, hip_x_R = [], []
    hip_y_L, hip_y_R = [], []
    sh_x_L,  sh_x_R  = [], []
    sh_y_L,  sh_y_R  = [], []
    knee_x_L, knee_x_R = [], []
    nose_x_arr = []
    conf_vals  = []
    n_valid    = 0

    for fi in frame_indices:
        fk = str(fi)
        if fk not in pose_frames: continue
        fd    = pose_frames[fk]
        scale = get_scale(fd)
        if scale is None: continue
        lh = get_kp(fd, KP['left_hip']);       rh = get_kp(fd, KP['right_hip'])
        ls = get_kp(fd, KP['left_shoulder']);  rs = get_kp(fd, KP['right_shoulder'])
        lk = get_kp(fd, KP['left_knee']);      rk = get_kp(fd, KP['right_knee'])
        ns = get_kp(fd, KP['nose'])
        if lh is None and rh is None and ls is None and rs is None: continue
        n_valid += 1
        if lh: hip_x_L.append(lh['x']/scale); hip_y_L.append(lh['y']/scale); conf_vals.append(lh['confidence'])
        if rh: hip_x_R.append(rh['x']/scale); hip_y_R.append(rh['y']/scale); conf_vals.append(rh['confidence'])
        if ls: sh_x_L.append(ls['x']/scale);  sh_y_L.append(ls['y']/scale)
        if rs: sh_x_R.append(rs['x']/scale);  sh_y_R.append(rs['y']/scale)
        if lk: knee_x_L.append(lk['x']/scale)
        if rk: knee_x_R.append(rk['x']/scale)
        if ns: nose_x_arr.append(ns['x']/scale)

    if n_valid < 5: return None

    rec = {
        'n_valid_frames': n_valid,
        'n_total_frames': len(frame_indices),
        'pct_valid':      n_valid/len(frame_indices),
        'duration_sec':   len(frame_indices)/ann_fps,
        'mean_conf':      float(np.mean(conf_vals)) if conf_vals else np.nan,
    }

    for arr, name in [
        (hip_x_L,'hip_x_L'), (hip_x_R,'hip_x_R'),
        (hip_y_L,'hip_y_L'), (hip_y_R,'hip_y_R'),
        (sh_x_L,'sh_x_L'),   (sh_x_R,'sh_x_R'),
        (sh_y_L,'sh_y_L'),   (sh_y_R,'sh_y_R'),
        (knee_x_L,'knee_x_L'), (knee_x_R,'knee_x_R'),
        (nose_x_arr,'nose_x'),
    ]:
        _jfeats(arr, rec, name, ann_fps)

    # Mean hip midpoint X (primary rocking signal)
    if hip_x_L and hip_x_R:
        ml  = min(len(hip_x_L), len(hip_x_R))
        mhx = (np.array(hip_x_L[:ml])+np.array(hip_x_R[:ml]))/2
        _jfeats(mhx, rec, 'mean_hip_x', ann_fps)
    elif hip_x_L:
        rec['mean_hip_x_amplitude'] = float(np.ptp(hip_x_L))
    elif hip_x_R:
        rec['mean_hip_x_amplitude'] = float(np.ptp(hip_x_R))

    # Mean shoulder midpoint X
    if sh_x_L and sh_x_R:
        ml  = min(len(sh_x_L), len(sh_x_R))
        msx = (np.array(sh_x_L[:ml])+np.array(sh_x_R[:ml]))/2
        _jfeats(msx, rec, 'mean_sh_x', ann_fps)

    # Trunk tilt: shoulder midpoint X minus hip midpoint X
    if sh_x_L and sh_x_R and hip_x_L and hip_x_R:
        ml   = min(len(sh_x_L), len(sh_x_R), len(hip_x_L), len(hip_x_R))
        msh  = (np.array(sh_x_L[:ml])+np.array(sh_x_R[:ml]))/2
        mhip = (np.array(hip_x_L[:ml])+np.array(hip_x_R[:ml]))/2
        tilt = msh - mhip
        _jfeats(tilt, rec, 'trunk_tilt', ann_fps)

    # Bilateral hip X coordination
    if len(hip_x_L) >= 5 and len(hip_x_R) >= 5:
        ml = min(len(hip_x_L), len(hip_x_R))
        xl = np.array(hip_x_L[:ml]); xr = np.array(hip_x_R[:ml])
        rec['bilateral_hip_x_corr']     = float(np.corrcoef(xl,xr)[0,1])
        rec['bilateral_hip_x_amp_diff'] = float(abs(np.ptp(xl)-np.ptp(xr)))
        rec['bilateral_hip_x_sym']      = float(
            1 - abs(np.ptp(xl)-np.ptp(xr))/(np.ptp(xl)+np.ptp(xr)+1e-8))

    hip_x_amp = max(float(np.ptp(hip_x_L)) if hip_x_L else 0.0,
                    float(np.ptp(hip_x_R)) if hip_x_R else 0.0)
    hip_y_amp = max(float(np.ptp(hip_y_L)) if hip_y_L else 0.0,
                    float(np.ptp(hip_y_R)) if hip_y_R else 0.0)
    rec['hip_2d_amplitude_max'] = float(max(hip_x_amp, hip_y_amp))
    rec['hip_x_y_ratio']        = float(hip_x_amp/(hip_y_amp+1e-8))
    return rec

# ── Run extraction ────────────────────────────────────────────────
all_features = []
n_ok = n_fail_ts = n_fail_pose = n_fail_kp = 0

for _, row in rock_rmm.iterrows():
    ts_str = str(row.get('matched_ts', ''))
    segs   = parse_timestamps(ts_str)
    if not segs: n_fail_ts += 1; continue
    try:
        with open(row['hrnet_path'], 'r') as f:
            pose_data = json.load(f)
        frames  = pose_data.get('frames', {})
        ann_fps = float(pose_data.get('ann_fps', FPS))
        if ann_fps != FPS: segs = parse_timestamps(ts_str, fps=ann_fps)
    except: n_fail_pose += 1; continue
    for s, e in segs:
        fidx  = list(range(s, e+1))
        if len(fidx) < 5: continue
        feats = extract_rocking_features(frames, fidx, ann_fps)
        if feats is None: n_fail_kp += 1; continue
        n_ok += 1
        feats.update({'pid': row['pid'], 'Group': row['Group'],
                      'age_mo': row['age_mo'], 'age_band': row['age_band'],
                      'label': row['matched_label'],
                      'clip': row.get('clip_filename', '')})
        all_features.append(feats)

print(f"Extraction: ok={n_ok}  fail_ts={n_fail_ts}  fail_pose={n_fail_pose}  fail_kp={n_fail_kp}")
if n_ok == 0:
    print("ERROR: No features extracted."); import sys; sys.exit(1)

feat_df = pd.DataFrame(all_features)
feat_df.to_csv(os.path.join(OUTPUT_DIR, 'clip_level_features.csv'), index=False)

META_COLS = {'pid','Group','age_mo','age_band','label','clip',
             'n_valid_frames','n_total_frames','pct_valid','duration_sec','mean_conf'}
FEAT_COLS = [c for c in feat_df.columns if c not in META_COLS]

PRIMARY_FEATS = [f for f in FEAT_COLS if any(x in f for x in [
    'mean_hip_x', 'mean_sh_x', 'trunk_tilt', 'nose_x',
    'hip_x_L', 'hip_x_R', 'hip_y_L', 'hip_y_R',
    'sh_x_L', 'sh_x_R', 'sh_y_L', 'sh_y_R',
    'bilateral_hip_x', 'hip_2d', 'hip_x_y',
])]
PRIMARY_FEATS = [f for f in PRIMARY_FEATS if f in feat_df.columns]

child_df = feat_df.groupby(['pid','Group'])[FEAT_COLS].mean().reset_index()
child_df['n_clips']  = feat_df.groupby(['pid','Group']).size().values
child_df['age_mo']   = feat_df.groupby(['pid','Group'])['age_mo'].first().values
child_df['age_band'] = feat_df.groupby(['pid','Group'])['age_band'].first().values
child_df.to_csv(os.path.join(OUTPUT_DIR, 'child_level_features.csv'), index=False)

CHILD_FEATS = [f for f in PRIMARY_FEATS if f in child_df.columns]

print(f"\nclip_level_features.csv : {len(feat_df)} rows")
print(f"child_level_features.csv: {len(child_df)} children")
print(f"PRIMARY_FEATS: {len(PRIMARY_FEATS)}")


# ═══════════════════════════════════════════════════════════════════
# PART 2: STATISTICAL ANALYSIS — FULL BATTERY
# ═══════════════════════════════════════════════════════════════════
hr("PART 2: STATISTICAL ANALYSIS")

# ── Step 2a: Combined MWU ─────────────────────────────────────────
def run_mwu(df, feat_cols, group_col='Group',
            ga='ASD', gb='Non-ASD', level='clip', subset='ALL'):
    recs = []
    dfa  = df[df[group_col]==ga]; dfb = df[df[group_col]==gb]
    for feat in feat_cols:
        av = dfa[feat].dropna().values; bv = dfb[feat].dropna().values
        if len(av)<3 or len(bv)<3: continue
        stat, p = stats.mannwhitneyu(av, bv, alternative='two-sided')
        d = cohen_d(av, bv); ci_lo, ci_hi = bootstrap_ci_d(av, bv, n_boot=500)
        recs.append({
            'feature': feat, 'subset': subset, 'level': level,
            f'{ga}_n': len(av), f'{gb}_n': len(bv),
            f'{ga}_median': float(np.median(av)), f'{gb}_median': float(np.median(bv)),
            f'{ga}_mean': float(np.mean(av)),     f'{gb}_mean': float(np.mean(bv)),
            'mw_stat': stat, 'p_raw': float(p),
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
        })
    if not recs: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(recs), 'p_raw').sort_values('p_raw')

print("\n--- 2a: Combined ASD vs Non-ASD — clip level ---")
r2a = run_mwu(feat_df, PRIMARY_FEATS, level='clip', subset='combined')
r2a.to_csv(os.path.join(OUTPUT_DIR, 'stats_combined_clip.csv'), index=False)
if len(r2a):
    print(f"  Features: {len(r2a)}  sig_raw: {r2a['sig_raw05'].sum()}  FDR: {r2a['sig_fdr05'].sum()}")
    for _, row in r2a.head(10).iterrows():
        sig = '★FDR' if row['sig_fdr05'] else ('*' if row['sig_raw05'] else '')
        print(f"    {row['feature']:<42} ASD={row['ASD_median']:.3f} "
              f"NASD={row['Non-ASD_median']:.3f} p={row['p_raw']:.4f} "
              f"d={row['cohens_d']:.2f} {sig}")

print("\n--- 2a(ii): Combined — child level ---")
r2a_child = run_mwu(child_df, CHILD_FEATS, level='child', subset='combined')
r2a_child.to_csv(os.path.join(OUTPUT_DIR, 'stats_combined_child.csv'), index=False)
if len(r2a_child):
    print(f"  sig_raw: {r2a_child['sig_raw05'].sum()}  FDR: {r2a_child['sig_fdr05'].sum()}")

# ── Step 2b: Age-stratified MWU ───────────────────────────────────
print("\n--- 2b: Age-stratified ASD vs Non-ASD ---")
age_strat_results = []
for band in AGE_BANDS:
    sub_c = feat_df[feat_df['age_band']==band]
    sub_k = child_df[child_df['age_band']==band]
    asd_n = sub_k[sub_k['Group']=='ASD']['pid'].nunique()
    nan_n = sub_k[sub_k['Group']=='Non-ASD']['pid'].nunique()
    print(f"  {band}: ASD={asd_n}, Non-ASD={nan_n}", end='')
    if asd_n >= 3 and nan_n >= 3:
        r = run_mwu(sub_c, PRIMARY_FEATS, level='clip', subset=band)
        r['age_band'] = band; age_strat_results.append(r)
        print(f"  sig_raw={r['sig_raw05'].sum()}  FDR={r['sig_fdr05'].sum()}")
    else:
        print(" → descriptive only")
if age_strat_results:
    pd.concat(age_strat_results).to_csv(
        os.path.join(OUTPUT_DIR, 'stats_age_stratified.csv'), index=False)

# ── Step 2c: Within-ASD trajectory ───────────────────────────────
print("\n--- 2c: Within-ASD trajectory ---")
asd_early = feat_df[(feat_df['Group']=='ASD') & (feat_df['age_band']=='19-31mo')]
asd_late  = feat_df[(feat_df['Group']=='ASD') & (feat_df['age_band']=='32-38mo')]
print(f"  ASD 19-31mo: n={asd_early['pid'].nunique()}  ASD 32-38mo: n={asd_late['pid'].nunique()}")
if len(asd_early)>=5 and len(asd_late)>=5:
    r2c = run_mwu(pd.concat([asd_early,asd_late]), PRIMARY_FEATS,
                  group_col='age_band', ga='19-31mo', gb='32-38mo',
                  level='clip', subset='ASD_19_31_vs_32_38')
    r2c.to_csv(os.path.join(OUTPUT_DIR, 'stats_within_ASD_trajectory.csv'), index=False)
    print(f"  sig_raw={r2c['sig_raw05'].sum()}  FDR={r2c['sig_fdr05'].sum()}")

# ── Step 2d: Within-Non-ASD trajectory ───────────────────────────
print("\n--- 2d: Within-Non-ASD trajectory ---")
nasd_early = feat_df[(feat_df['Group']=='Non-ASD') & (feat_df['age_band']=='11-18mo')]
nasd_mid   = feat_df[(feat_df['Group']=='Non-ASD') & (feat_df['age_band']=='19-31mo')]
if len(nasd_early)>=3 and len(nasd_mid)>=3:
    r2d = run_mwu(pd.concat([nasd_early,nasd_mid]), PRIMARY_FEATS,
                  group_col='age_band', ga='11-18mo', gb='19-31mo',
                  level='clip', subset='NonASD_11_18_vs_19_31')
    r2d.to_csv(os.path.join(OUTPUT_DIR, 'stats_within_NonASD_trajectory.csv'), index=False)
    print(f"  sig_raw={r2d['sig_raw05'].sum()}")

# ── Step 2e: Kruskal-Wallis Age × Group ──────────────────────────
print("\n--- 2e: Kruskal-Wallis Age × Group interaction ---")
feat_df['age_grp_cell'] = (feat_df['Group'].str.replace('-','') +
                            '_' + feat_df['age_band'].fillna('unk'))
kw_recs = []
for feat in PRIMARY_FEATS:
    cells = feat_df['age_grp_cell'].value_counts()
    grps  = [feat_df[feat_df['age_grp_cell']==c][feat].dropna().values
              for c in cells.index
              if len(feat_df[feat_df['age_grp_cell']==c][feat].dropna()) >= 3]
    if len(grps) < 2: continue
    try:
        stat, p = stats.kruskal(*grps)
        kw_recs.append({'feature': feat, 'kw_stat': stat, 'p_raw': p})
    except: pass
if kw_recs:
    kw_df = fdr_annotate(pd.DataFrame(kw_recs).sort_values('p_raw'), 'p_raw')
    kw_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_age_group_kruskal.csv'), index=False)
    print(f"  sig_raw={kw_df['sig_raw05'].sum()}  FDR={kw_df['sig_fdr05'].sum()}")

# ── Step 2f: Spearman age correlations ───────────────────────────
print("\n--- 2f: Spearman age correlations ---")
sp_recs = []
for grp in GROUPS:
    sub = feat_df[feat_df['Group']==grp]
    for feat in PRIMARY_FEATS:
        vals = sub[['age_mo',feat]].dropna()
        if len(vals)<5: continue
        r, p = stats.spearmanr(vals['age_mo'], vals[feat])
        sp_recs.append({'Group': grp, 'feature': feat,
                        'spearman_r': r, 'p_raw': p, 'n': len(vals)})
if sp_recs:
    sp_df = fdr_annotate(pd.DataFrame(sp_recs), 'p_raw')
    sp_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_age_spearman.csv'), index=False)
    print(f"  Sig age correlations: {sp_df['sig_raw05'].sum()}")

# ── Step 2g: ICC ─────────────────────────────────────────────────
print("\n--- 2g: ICC (within-child clustering) ---")
def compute_icc(clip_df, feat_cols):
    records = []
    for feat in feat_cols:
        sub = clip_df[['pid', feat]].dropna()
        if len(sub) < 10: continue
        groups = [g[feat].values for _, g in sub.groupby('pid') if len(g) >= 2]
        if len(groups) < 5: continue
        n_total = sum(len(g) for g in groups); k = len(groups)
        n0 = (n_total - sum(len(g)**2/n_total for g in groups))/(k-1)
        grand = np.concatenate(groups)
        ms_between = np.sum([len(g)*(np.mean(g)-np.mean(grand))**2
                             for g in groups])/(k-1)
        ms_within  = np.sum([np.sum((g-np.mean(g))**2)
                             for g in groups])/(n_total-k)
        icc = max(0.0, (ms_between-ms_within)/(ms_between+(n0-1)*ms_within))
        records.append({'feature': feat, 'ICC': round(icc,4)})
    return pd.DataFrame(records).sort_values('ICC', ascending=False)

icc_df = compute_icc(feat_df, PRIMARY_FEATS)
icc_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_icc.csv'), index=False)
print(f"  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)} features")
print(icc_df.head(8).to_string(index=False))

# ── Step 2h: LME + Kenward-Roger ─────────────────────────────────
print("\n--- 2h: LME + Kenward-Roger (clip level) ---")
def run_lme_kr(clip_df, feat_cols, subset_label='combined'):
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo_c',feat]].dropna(
              subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique()<2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min()<3: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        d  = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        p_val = np.nan; coef = np.nan; se = np.nan
        method_used = 'none'; converged = False
        if _RPY2_OK:
            try:
                safe  = re.sub(r'[^A-Za-z0-9_]','_',feat)
                sub2  = sub.rename(columns={feat: safe})
                formula = f'{safe} ~ Group_bin + age_mo_c + (1|pid)'
                r_df  = pandas2ri.py2rpy(sub2); ro.globalenv['r_df'] = r_df
                ro.r(f'fit <- lmerTest::lmer({formula}, data=r_df, REML=TRUE)')
                summ  = pandas2ri.rpy2py(
                    ro.r('as.data.frame(coef(summary(fit,ddf="Kenward-Roger")))'))
                if 'Group_bin' in summ.index:
                    coef = float(summ.loc['Group_bin','Estimate'])
                    se   = float(summ.loc['Group_bin','Std. Error'])
                    p_val = float(summ.loc['Group_bin','Pr(>|t|)'])
                    method_used = 'LME_KR'; converged = True
            except: pass
        if method_used == 'none':
            try:
                mdf = smf.mixedlm(f'{feat} ~ Group_bin + age_mo_c', sub,
                                  groups=sub['pid']).fit(method=['lbfgs'],
                                                         reml=True, maxiter=300)
                coef = float(mdf.params.get('Group_bin', np.nan))
                se   = float(mdf.bse.get('Group_bin', np.nan))
                p_val = float(mdf.pvalues.get('Group_bin', np.nan))
                method_used = 'LME_noKR'; converged = bool(mdf.converged)
            except: pass
        records.append({'feature': feat, 'subset': subset_label,
                        'method': method_used, 'coef_ASD': coef, 'se': se,
                        'p_raw': p_val, 'cohens_d': d,
                        'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
                        'converged': converged,
                        'n_asd': sub[sub['Group_bin']==1]['pid'].nunique(),
                        'n_nasd': sub[sub['Group_bin']==0]['pid'].nunique(),
                        'n_clips': len(sub)})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

lme_kr_all = run_lme_kr(feat_df, PRIMARY_FEATS, 'combined')
if len(lme_kr_all):
    lme_kr_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_kr.csv'), index=False)
    print(f"  sig_raw={lme_kr_all['sig_raw05'].sum()}  FDR={lme_kr_all['sig_fdr05'].sum()}")
    method_name = lme_kr_all['method'].mode()[0]
    print(f"  Method used: {method_name}")

# Age-stratified LME
print("\n  LME per age band:")
lme_band_results = {}
for band in STAT_BANDS:
    sub   = feat_df[feat_df['age_band']==band]
    asd_n = sub[sub['Group']=='ASD']['pid'].nunique()
    nan_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
    print(f"    {band}: ASD={asd_n}, Non-ASD={nan_n}", end='')
    if asd_n >= 3 and nan_n >= 3:
        r = run_lme_kr(sub, PRIMARY_FEATS, band)
        if len(r):
            r.to_csv(os.path.join(OUTPUT_DIR, f'stats_lme_{band.replace("-","_")}.csv'), index=False)
            lme_band_results[band] = r
            print(f"  sig_raw={r['sig_raw05'].sum()}")
    else: print(" → skipped")

# Within-ASD LME (age covariate only)
print("\n  Within-ASD age trajectory (LME):")
asd_sub = feat_df[feat_df['Group']=='ASD'].copy()
if asd_sub['pid'].nunique() >= 4:
    asd_sub['Group_bin'] = 0.0  # dummy, replaced by age_mo_c below
    asd_traj_recs = []
    for feat in PRIMARY_FEATS:
        sub2 = asd_sub[['pid','age_mo',feat]].dropna()
        if len(sub2)<6 or sub2['pid'].nunique()<3: continue
        sub2['age_mo_c'] = sub2['age_mo'] - sub2['age_mo'].mean()
        try:
            mdf = smf.mixedlm(f'{feat} ~ age_mo_c', sub2, groups=sub2['pid']).fit(
                method=['lbfgs'], reml=True, maxiter=300)
            asd_traj_recs.append({'feature': feat,
                                   'coef_age': float(mdf.params.get('age_mo_c', np.nan)),
                                   'p_raw': float(mdf.pvalues.get('age_mo_c', np.nan)),
                                   'n_subjects': sub2['pid'].nunique()})
        except: pass
    if asd_traj_recs:
        at_df = fdr_annotate(pd.DataFrame(asd_traj_recs).sort_values('p_raw'), 'p_raw')
        at_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_asd_trajectory.csv'), index=False)
        print(f"  sig_raw={at_df['sig_raw05'].sum()}  FDR={at_df['sig_fdr05'].sum()}")

# Growth-curve interaction (Group × Age)
print("\n  Growth-curve interaction (Group × Age):")
gci_recs = []
for feat in PRIMARY_FEATS:
    sub = feat_df[['pid','Group','age_mo',feat]].dropna().copy()
    if len(sub)<10 or sub['pid'].nunique()<4: continue
    sub['Group_bin'] = (sub['Group']=='ASD').astype(float)
    sub['age_mo_c']  = sub['age_mo'] - sub['age_mo'].mean()
    if sub['Group_bin'].nunique() < 2: continue
    try:
        mdf = smf.mixedlm(f'{feat} ~ Group_bin * age_mo_c', sub,
                           groups=sub['pid']).fit(method=['lbfgs'], reml=True, maxiter=300)
        ix_key = next((k for k in mdf.pvalues.index
                       if 'Group_bin' in k and 'age' in k), None)
        if ix_key:
            gci_recs.append({'feature': feat,
                              'coef_interaction': float(mdf.params.get(ix_key, np.nan)),
                              'p_raw': float(mdf.pvalues.get(ix_key, np.nan))})
    except: pass
if gci_recs:
    gci_df = fdr_annotate(pd.DataFrame(gci_recs).sort_values('p_raw'), 'p_raw')
    gci_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_growth_curve.csv'), index=False)
    print(f"  Interaction sig_raw={gci_df['sig_raw05'].sum()}  FDR={gci_df['sig_fdr05'].sum()}")

# Age-only confound check
print("\n  Age-only confound check (all subjects, no Group term):")
age_only_recs = []
for feat in PRIMARY_FEATS:
    sub = feat_df[['pid','age_mo',feat]].dropna()
    if len(sub)<6 or sub['pid'].nunique()<3: continue
    sub = sub.copy(); sub['age_mo_c'] = sub['age_mo'] - sub['age_mo'].mean()
    try:
        mdf = smf.mixedlm(f'{feat} ~ age_mo_c', sub, groups=sub['pid']).fit(
            method=['lbfgs'], reml=True, maxiter=300)
        age_only_recs.append({'feature': feat,
                               'coef_age': float(mdf.params.get('age_mo_c', np.nan)),
                               'p_raw': float(mdf.pvalues.get('age_mo_c', np.nan)),
                               'n_subjects': sub['pid'].nunique()})
    except: pass
if age_only_recs:
    ao_df = fdr_annotate(pd.DataFrame(age_only_recs).sort_values('p_raw'), 'p_raw')
    ao_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_lme_age_confound.csv'), index=False)
    print(f"  Features with sig age effect: {ao_df['sig_raw05'].sum()}/{len(ao_df)}")

# ── Step 2i: GEE robustness check ────────────────────────────────
print("\n--- 2i: GEE robustness check ---")
def run_gee(clip_df, feat_cols, subset_label='combined'):
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    pid_map = {p:i for i,p in enumerate(df_use['pid'].unique())}
    df_use['pid_int'] = df_use['pid'].map(pid_map)
    for feat in feat_cols:
        sub = df_use[['pid_int','Group_bin','age_mo_c',feat]].dropna(
              subset=['pid_int','Group_bin',feat])
        if sub['Group_bin'].nunique()<2 or len(sub)<20: continue
        counts = sub.groupby('pid_int').size()
        sub    = sub[sub['pid_int'].isin(counts[counts>=2].index)]
        if len(sub)<20: continue
        try:
            safe  = re.sub(r'[^A-Za-z0-9_]','_',feat)
            sub2  = sub.rename(columns={feat: safe})
            res   = GEE.from_formula(f'{safe} ~ Group_bin + age_mo_c',
                                      'pid_int', data=sub2,
                                      family=Gaussian(),
                                      cov_struct=Exchangeable()).fit(maxiter=100)
            av = sub[sub['Group_bin']==1][feat].values
            nv = sub[sub['Group_bin']==0][feat].values
            records.append({'feature': feat, 'subset': subset_label,
                            'coef_ASD': float(res.params.get('Group_bin', np.nan)),
                            'p_raw': float(res.pvalues.get('Group_bin', np.nan)),
                            'cohens_d': cohen_d(av, nv), 'n_clips': len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

gee_all = run_gee(feat_df, PRIMARY_FEATS)
if len(gee_all):
    gee_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_gee.csv'), index=False)
    print(f"  sig_raw={gee_all['sig_raw05'].sum()}  FDR={gee_all['sig_fdr05'].sum()}")

# ── Step 2j: Child-level permutation ─────────────────────────────
print("\n--- 2j: Child-level permutation test ---")
def run_child_permutation(cdf, feat_cols, n_perm=5000, subset_label='combined'):
    rng = np.random.default_rng(42); records = []
    for feat in feat_cols:
        sub = cdf[['pid','Group',feat]].dropna()
        if sub['Group'].nunique()<2: continue
        av  = sub[sub['Group']=='ASD'][feat].values
        nv  = sub[sub['Group']=='Non-ASD'][feat].values
        if len(av)<3 or len(nv)<3: continue
        obs_stat = abs(np.mean(av)-np.mean(nv))
        n_asd    = len(av); n_total = len(sub)
        vals_arr = sub[feat].values
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            sl  = rng.permutation(['ASD']*n_asd+['Non-ASD']*(n_total-n_asd))
            a_v = vals_arr[np.array(sl)=='ASD']
            n_v = vals_arr[np.array(sl)=='Non-ASD']
            a_v = a_v[~np.isnan(a_v)]; n_v = n_v[~np.isnan(n_v)]
            perm_stats[i] = abs(np.mean(a_v)-np.mean(n_v)) if len(a_v)>0 and len(n_v)>0 else 0
        p_perm = max(float(np.mean(perm_stats>=obs_stat)), 1.0/n_perm)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({'feature': feat, 'subset': subset_label,
                        'method': 'ChildPerm', 'obs_stat': float(obs_stat),
                        'p_raw': p_perm, 'cohens_d': d,
                        'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
                        'n_asd': len(av), 'n_nasd': len(nv)})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

perm_all = run_child_permutation(child_df, CHILD_FEATS)
if len(perm_all):
    perm_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_child_permutation.csv'), index=False)
    print(f"  sig_raw={perm_all['sig_raw05'].sum()}  FDR={perm_all['sig_fdr05'].sum()}")

# ── Step 2k: Wild cluster bootstrap ──────────────────────────────
print("\n--- 2k: Wild cluster bootstrap (child level) ---")
def run_wild_bootstrap(cdf, feat_cols, n_boot=5000, subset_label='combined'):
    rng = np.random.default_rng(99); records = []
    df_use = cdf.copy().dropna(subset=['age_mo'])
    df_use['Group_bin'] = (df_use['Group']=='ASD').astype(float)
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo',feat]].dropna()
        if sub['Group_bin'].nunique()<2: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        if len(av)<3 or len(nv)<3: continue
        n = len(sub); y = sub[feat].values.astype(float)
        X = np.column_stack([np.ones(n), sub['Group_bin'].values, sub['age_mo'].values])
        try:
            beta,_,_,_ = np.linalg.lstsq(X, y, rcond=None)
        except: continue
        resid = y - X@beta; t_obs = beta[1]/(np.std(resid)/np.sqrt(n)+1e-10)
        X0    = X[:,[0,2]]
        try:
            beta0,_,_,_ = np.linalg.lstsq(X0, y, rcond=None)
        except: continue
        resid0 = y - X0@beta0; pids = sub['pid'].values; u_pids = np.unique(pids)
        t_boot = np.zeros(n_boot)
        for b in range(n_boot):
            w_map = {p: rng.choice([-1.0,1.0]) for p in u_pids}
            w     = np.array([w_map[p] for p in pids])
            y_b   = X0@beta0 + resid0*w
            try:
                beta_b,_,_,_ = np.linalg.lstsq(X, y_b, rcond=None)
                resid_b = y_b - X@beta_b
                t_boot[b] = beta_b[1]/(np.std(resid_b)/np.sqrt(n)+1e-10)
            except: t_boot[b] = 0.0
        p_wb = max(float(np.mean(np.abs(t_boot)>=abs(t_obs))), 1.0/n_boot)
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({'feature': feat, 'subset': subset_label,
                        'method': 'WildBoot', 'coef_ASD': float(beta[1]),
                        't_obs': float(t_obs), 'p_raw': p_wb,
                        'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
                        'n_asd': int(len(av)), 'n_nasd': int(len(nv))})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

boot_all = run_wild_bootstrap(child_df, CHILD_FEATS)
if len(boot_all):
    boot_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_wild_bootstrap.csv'), index=False)
    print(f"  sig_raw={boot_all['sig_raw05'].sum()}  FDR={boot_all['sig_fdr05'].sum()}")

# ── Step 2i(ii): CR2 bias-reduced linearization ──────────────────
print("\n--- 2i(ii): CR2 bias-reduced linearization ---")
def run_cr2(clip_df, feat_cols, subset_label='combined'):
    if not _WBT_OK: print("  [CR2] skipped — wildboottest not available"); return pd.DataFrame()
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo_c',feat]].dropna(
              subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique()<2 or len(sub)<10: continue
        X_cols = ['Group_bin','age_mo_c']
        X      = sub[X_cols].values.astype(float)
        y      = sub[feat].values.astype(float)
        clusters = sub['pid'].values
        try:
            wbt = WildboottestHC(X=X, y=y, cluster=clusters,
                                 R=np.eye(len(X_cols))[[0],:],
                                 B=999, bootstrap_type='WCR11')
            wbt.get_wildboottest()
            records.append({'feature': feat, 'subset': subset_label,
                            'method': 'CR2', 'p_raw': float(wbt.pvalue),
                            'n_clips': len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

cr2_all = run_cr2(feat_df, PRIMARY_FEATS)
if len(cr2_all):
    cr2_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_cr2.csv'), index=False)
    print(f"  sig_raw={cr2_all['sig_raw05'].sum()}  FDR={cr2_all['sig_fdr05'].sum()}")

# ── Step 2l: Consensus table ──────────────────────────────────────
print("\n--- 2l: Consensus table across all methods ---")
def make_consensus(results_dict, feat_cols, threshold=0.05):
    rows = []
    for feat in feat_cols:
        row = {'feature': feat}; n_sig = 0
        for mname, res_df in results_dict.items():
            if res_df is None or len(res_df)==0:
                row[f'p_{mname}'] = np.nan; continue
            match = res_df[res_df['feature']==feat]
            if len(match)==0:
                row[f'p_{mname}'] = np.nan
            else:
                p = match['p_raw'].values[0]
                row[f'p_{mname}'] = round(p,4)
                if p < threshold: n_sig += 1
        row['n_methods_sig'] = n_sig; rows.append(row)
    cons = pd.DataFrame(rows)
    # Pull Cohen's d from LME or permutation
    for src_name in ['LME_KR','ChildPerm','MWU']:
        src = results_dict.get(src_name)
        if src is not None and len(src) and 'cohens_d' in src.columns:
            cons['cohens_d_ref'] = cons['feature'].map(
                src.set_index('feature')['cohens_d'].to_dict())
            if 'd_ci_lo' in src.columns:
                cons['d_ci_lo'] = cons['feature'].map(
                    src.set_index('feature')['d_ci_lo'].to_dict())
                cons['d_ci_hi'] = cons['feature'].map(
                    src.set_index('feature')['d_ci_hi'].to_dict())
            break
    return cons.sort_values('n_methods_sig', ascending=False)

consensus_results = {
    'LME_KR':   lme_kr_all if len(lme_kr_all)>0 else pd.DataFrame(),
    'CR2':      cr2_all,
    'GEE':      gee_all,
    'ChildPerm': perm_all,
    'WildBoot': boot_all,
    'MWU':      r2a,
}
consensus_all = make_consensus(consensus_results, PRIMARY_FEATS)
consensus_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_consensus_all.csv'), index=False)
print(f"  Features sig on ≥3 methods: {(consensus_all['n_methods_sig']>=3).sum()}")
print(f"  Features sig on ≥2 methods: {(consensus_all['n_methods_sig']>=2).sum()}")
print(consensus_all[consensus_all['n_methods_sig']>0].head(10)[
    ['feature','n_methods_sig']].to_string(index=False))

# ── Step 2m: Consistency gate ─────────────────────────────────────
# For rocking there is only one behavior, so we use age-band as the
# stratification dimension instead of behavior.
print("\n--- 2m: Consistency gate (same-direction check across age bands) ---")
sig_feats = list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else \
            list(r2a[r2a['sig_raw05']]['feature'])[:20]

band_mwu_dict = {}
for band in AGE_BANDS:
    sub   = feat_df[feat_df['age_band']==band]
    asd_n = sub[sub['Group']=='ASD']['pid'].nunique()
    nan_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
    if asd_n<3 or nan_n<3: continue
    recs = []
    for feat in PRIMARY_FEATS:
        av = sub[sub['Group']=='ASD'][feat].dropna().values
        nv = sub[sub['Group']=='Non-ASD'][feat].dropna().values
        if len(av)<3 or len(nv)<3: continue
        _, p = stats.mannwhitneyu(av, nv, alternative='two-sided')
        recs.append({'feature': feat, 'cohens_d': cohen_d(av,nv),
                     'p_raw': p, 'age_band': band})
    if recs: band_mwu_dict[band] = pd.DataFrame(recs)

cons_recs = []; consistent_feats = []
band_all  = pd.concat(band_mwu_dict.values(), ignore_index=True) if band_mwu_dict else pd.DataFrame()
for feat in sig_feats:
    if len(band_all)==0: break
    sub = band_all[band_all['feature']==feat]
    if len(sub)<2: continue
    signs  = np.sign(sub['cohens_d'].values)
    n_same = int((signs==signs[0]).sum())
    passed = (n_same==len(sub))
    cons_recs.append({'feature': feat, 'n_bands_tested': len(sub),
                      'n_same_direction': n_same, 'consistent': passed})
    if passed: consistent_feats.append(feat)

if cons_recs:
    cons_gate_df = pd.DataFrame(cons_recs)
    cons_gate_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_consistency_gate.csv'), index=False)
    print(f"  Consistency gate: {len(consistent_feats)}/{len(sig_feats)} passed")
    for f in consistent_feats: print(f"    ✓ {f}")

# ── Step 2n: Age-band × Group interaction term ───────────────────
# (Adapted from RMM behavior×Group interaction)
print("\n--- 2n: Age-band × Group interaction (LME) ---")
feat_df['age_band_clean'] = feat_df['age_band'].str.replace('-','_').str.replace('mo','')
beh_dummy_cols = []
for band in ['19_31', '32_38']:  # '11_18' is reference
    col = f'band_{band}'
    feat_df[col] = (feat_df['age_band_clean']==band).astype(float)
    beh_dummy_cols.append(col)

bgi_recs = []
for feat in PRIMARY_FEATS:
    sub = feat_df[['pid','Group','age_mo',feat]+beh_dummy_cols].dropna(
          subset=['pid','Group',feat])
    if sub['Group'].nunique()<2 or sub['pid'].nunique()<5: continue
    sub = sub.copy()
    sub['Group_bin'] = (sub['Group']=='ASD').astype(float)
    sub['age_mo_c']  = sub['age_mo'] - sub['age_mo'].mean()
    beh_x_grp = [f'Group_bin:{c}' for c in beh_dummy_cols]
    formula   = (f'{feat} ~ Group_bin + age_mo_c + '
                 + '+'.join(beh_dummy_cols) + ' + ' + '+'.join(beh_x_grp))
    try:
        mdf = smf.mixedlm(formula, sub, groups=sub['pid']).fit(
            method=['lbfgs'], reml=True, maxiter=300)
        for bxg in beh_x_grp:
            if bxg in mdf.pvalues.index:
                bgi_recs.append({'feature': feat, 'interaction_term': bxg,
                                 'coef': float(mdf.params.get(bxg, np.nan)),
                                 'p_raw': float(mdf.pvalues.get(bxg, np.nan))})
    except: continue
if bgi_recs:
    bgi_df = fdr_annotate(pd.DataFrame(bgi_recs), 'p_raw').sort_values('p_raw')
    bgi_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_band_group_interaction.csv'), index=False)
    print(f"  Band×Group interactions sig (raw): {bgi_df['sig_raw05'].sum()}")
    if bgi_df['sig_raw05'].sum()>0:
        print("  ⚠ Group effect differs across age bands for these features:")
        for _, r in bgi_df[bgi_df['sig_raw05']].head(5).iterrows():
            print(f"    {r['feature']:<42} {r['interaction_term']}  p={r['p_raw']:.4f}")

# Print full method summary
print("\n=== METHOD SUMMARY ===")
for name, res in [('LME_KR', lme_kr_all), ('CR2', cr2_all), ('GEE', gee_all),
                  ('ChildPerm', perm_all), ('WildBoot', boot_all), ('MWU', r2a)]:
    if len(res)>0:
        print(f"  {name:<12}: sig_raw={res['sig_raw05'].sum():>3}  "
              f"FDR={res['sig_fdr05'].sum():>3}")


# ═══════════════════════════════════════════════════════════════════
# PART 3: BAYESIAN HIERARCHICAL LMM
# ═══════════════════════════════════════════════════════════════════
hr("PART 3: BAYESIAN HIERARCHICAL LMM")

bayes_main_results = {}

def _build_bayes_df(df, feat):
    tmp = df[['pid','Group','age_mo',feat]].dropna().copy()
    if len(tmp)<8 or tmp['pid'].nunique()<4: return None
    tmp['Group_bin'] = (tmp['Group']=='ASD').astype(float)
    tmp['age_c']     = tmp['age_mo'] - tmp['age_mo'].mean()
    y_z, ym, ys     = _standardise(tmp[feat])
    pids, pid_idx   = np.unique(tmp['pid'].values, return_inverse=True)
    return {'df': tmp, 'y_z': y_z.astype(float),
            'group_bin': tmp['Group_bin'].values.astype(float),
            'age_c': tmp['age_c'].values.astype(float),
            'pid_idx': pid_idx, 'n_pids': len(pids),
            'y_mean': ym, 'y_std': ys, 'n_obs': len(tmp)}

def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS,
                    tune=BAYES_TUNE, chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha     = pm.Normal('alpha', 0, 1)
        b_group   = pm.Normal('b_group', 0, prior_sd)
        b_age     = pm.Normal('b_age', 0, 0.5)
        sigma_pid = pm.HalfNormal('sigma_pid', 1)
        sigma     = pm.HalfNormal('sigma', 1)
        alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu        = (alpha + alpha_pid[bd['pid_idx']]
                     + b_group*bd['group_bin'] + b_age*bd['age_c'])
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
        'converged': bool(rhat<1.05 and ess>400 and n_div==0),
        'prior_sd': prior_sd,
    }

def _fit_bayes_age_only(bd, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                        chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha     = pm.Normal('alpha', 0, 1)
        b_age     = pm.Normal('b_age', 0, 0.5)
        sigma_pid = pm.HalfNormal('sigma_pid', 1)
        sigma     = pm.HalfNormal('sigma', 1)
        alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu        = alpha + alpha_pid[bd['pid_idx']] + b_age*bd['age_c']
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    age_post = idata.posterior['b_age'].values.flatten()
    hdi      = az.hdi(idata, var_names=['b_age'], hdi_prob=0.94)['b_age'].values
    diag     = az.summary(idata, var_names=['b_age'], hdi_prob=0.94)
    return idata, {
        'b_age_mean': float(age_post.mean()), 'b_age_sd': float(age_post.std()),
        'hdi94_lo': float(hdi[0]), 'hdi94_hi': float(hdi[1]),
        'p_pos': float((age_post>0).mean()),
        'bf10': _savage_dickey_bf(age_post),
        'rhat': float(diag['r_hat'].values[0]),
        'ess_bulk': float(diag['ess_bulk'].values[0]),
        'n_divergences': int(idata.sample_stats['diverging'].values.sum()),
    }

def _fit_bayes_interaction(bd, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                           chains=BAYES_CHAINS, seed=42):
    with pm.Model():
        alpha      = pm.Normal('alpha', 0, 1)
        b_group    = pm.Normal('b_group', 0, 0.5)
        b_age      = pm.Normal('b_age', 0, 0.5)
        b_interact = pm.Normal('b_interact', 0, 0.5)
        sigma_pid  = pm.HalfNormal('sigma_pid', 1)
        sigma      = pm.HalfNormal('sigma', 1)
        alpha_pid  = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
        mu         = (alpha + alpha_pid[bd['pid_idx']]
                      + b_group*bd['group_bin'] + b_age*bd['age_c']
                      + b_interact*bd['group_bin']*bd['age_c'])
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, return_inferencedata=True)
    int_post = idata.posterior['b_interact'].values.flatten()
    hdi      = az.hdi(idata, var_names=['b_interact'], hdi_prob=0.94)['b_interact'].values
    diag     = az.summary(idata, var_names=['b_interact'], hdi_prob=0.94)
    return idata, {
        'b_interact_mean': float(int_post.mean()), 'b_interact_sd': float(int_post.std()),
        'hdi94_lo': float(hdi[0]), 'hdi94_hi': float(hdi[1]),
        'p_pos': float((int_post>0).mean()),
        'bf10': _savage_dickey_bf(int_post),
        'rhat': float(diag['r_hat'].values[0]),
        'ess_bulk': float(diag['ess_bulk'].values[0]),
        'n_divergences': int(idata.sample_stats['diverging'].values.sum()),
    }

def prior_predictive_check(bd, feat, prior_sd=0.5):
    with pm.Model():
        b_group = pm.Normal('b_group', 0, prior_sd)
        b_age   = pm.Normal('b_age', 0, 0.5)
        sigma   = pm.HalfNormal('sigma', 1)
        alpha   = pm.Normal('alpha', 0, 1)
        mu      = alpha + b_group*bd['group_bin'] + b_age*bd['age_c']
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
        ppc = pm.sample_prior_predictive(samples=200, random_seed=42)
    prior_ys    = ppc.prior_predictive['y_obs'].values.flatten()
    obs_range   = (bd['y_z'].min(), bd['y_z'].max())
    prior_range = (float(np.percentile(prior_ys,1)), float(np.percentile(prior_ys,99)))
    return {'feature': feat, 'obs_min': obs_range[0], 'obs_max': obs_range[1],
            'prior_p1': prior_range[0], 'prior_p99': prior_range[1],
            'plausible': prior_range[0]<=obs_range[0] and prior_range[1]>=obs_range[1]}

if not RUN_BAYESIAN or not _PYMC_OK:
    print("  Bayesian skipped")
else:
    # Top features by permutation or MWU
    bayes_feats = (perm_all.sort_values('p_raw').head(15)['feature'].tolist()
                   if len(perm_all)>0 else
                   r2a.sort_values('p_raw').head(10)['feature'].tolist())
    print(f"\nRunning Bayesian models on {len(bayes_feats)} features ...")
    print(f"  Prior widths tested: {PRIOR_SDS}")

    ppc_records = []; sensitivity_records = []; bayes_records = []
    idata_main  = {}

    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        # Prior predictive check
        try:
            ppc_rec = prior_predictive_check(bd, feat)
            ppc_records.append(ppc_rec)
            if not ppc_rec['plausible']:
                print(f"  ⚠ PPC narrow for {feat}")
        except: pass
        # Prior sensitivity sweep
        bf_vals = {}
        for psd in PRIOR_SDS:
            try:
                idata, summ = _fit_bayes_main(bd, prior_sd=psd)
                summ['feature'] = feat; summ['prior_sd'] = psd
                sensitivity_records.append(summ)
                bf_vals[psd] = summ['bf10']
                if psd == 0.5:
                    idata_main[feat] = idata
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")
        # Main result at prior_sd=0.5 + robustness flag
        if 0.5 in bf_vals:
            match = [r for r in sensitivity_records
                     if r['feature']==feat and r['prior_sd']==0.5]
            if match:
                rec = match[-1].copy()
                bfs = [bf_vals[p] for p in PRIOR_SDS
                       if p in bf_vals and not np.isnan(bf_vals.get(p, np.nan))]
                rec['bf_robust'] = bool(len(bfs)>=2 and
                                        all((b>1)==(bfs[0]>1) for b in bfs))
                bayes_records.append(rec)
                flag   = '✓' if rec.get('converged') else '⚠'
                bf_str = ' | '.join([f"sd={p}:BF={bf_vals.get(p,np.nan):.2f}"
                                     for p in PRIOR_SDS])
                print(f"  {feat:<42} {bf_str} {flag}")

    # --- Age-only confound model (Rocking-specific) ---
    print("\n  Bayesian age-only confound models ...")
    bayes_age_recs = []
    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        try:
            _, summ = _fit_bayes_age_only(bd)
            summ['feature'] = feat; bayes_age_recs.append(summ)
        except: pass
    if bayes_age_recs:
        pd.DataFrame(bayes_age_recs).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_age_confound.csv'), index=False)

    # --- Growth-curve interaction ---
    print("\n  Bayesian growth-curve interactions ...")
    bayes_int_recs = []
    for feat in bayes_feats:
        bd = _build_bayes_df(feat_df, feat)
        if bd is None: continue
        try:
            _, summ = _fit_bayes_interaction(bd)
            summ['feature'] = feat; bayes_int_recs.append(summ)
        except: pass
    if bayes_int_recs:
        pd.DataFrame(bayes_int_recs).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_growth_curve.csv'), index=False)

    # --- Age-stratified Bayesian ---
    print("\n  Bayesian age-stratified ...")
    bayes_strat_recs = []
    for band in STAT_BANDS:
        sub   = feat_df[feat_df['age_band']==band]
        asd_n = sub[sub['Group']=='ASD']['pid'].nunique()
        nan_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        print(f"  {band}: ASD={asd_n}, Non-ASD={nan_n}", end='')
        if asd_n<3 or nan_n<3: print(" → skipped"); continue
        cnt = 0
        for feat in bayes_feats:
            bd = _build_bayes_df(sub, feat)
            if bd is None: continue
            try:
                _, summ = _fit_bayes_main(bd)
                summ['feature'] = feat; summ['age_band'] = band
                bayes_strat_recs.append(summ); cnt += 1
            except: pass
        print(f" → fitted {cnt} models")
    if bayes_strat_recs:
        pd.DataFrame(bayes_strat_recs).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_age_stratified.csv'), index=False)

    # --- Save outputs ---
    if ppc_records:
        pd.DataFrame(ppc_records).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_ppc.csv'), index=False)
    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_sensitivity.csv'), index=False)
    if bayes_records:
        bayes_df = pd.DataFrame(bayes_records).sort_values('bf10', ascending=False)
        bayes_df.to_csv(os.path.join(OUTPUT_DIR, 'bayes_main.csv'), index=False)
        bayes_main_results['combined'] = bayes_df
        print(f"\n  BF10>3 : {(bayes_df['bf10']>3).sum()}/{len(bayes_df)}")
        print(f"  BF10>10: {(bayes_df['bf10']>10).sum()}/{len(bayes_df)}")
        print(f"  BF robust: {bayes_df['bf_robust'].sum()}/{len(bayes_df)}")
        bad = bayes_df[~bayes_df['converged']]
        if len(bad):
            print(f"  ⚠ {len(bad)} convergence issues:")
            for _, r in bad.iterrows():
                print(f"    {r['feature']} rhat={r.get('rhat',np.nan):.3f} "
                      f"ess={r.get('ess_bulk',np.nan):.0f}")

    # --- Diagnostics summary ---
    all_bayes_csvs = ['bayes_main.csv','bayes_age_stratified.csv',
                      'bayes_age_confound.csv','bayes_growth_curve.csv']
    diag_frames = []
    for fn in all_bayes_csvs:
        fp = os.path.join(OUTPUT_DIR, fn)
        if os.path.isfile(fp):
            tmp = pd.read_csv(fp); tmp['source'] = fn; diag_frames.append(tmp)
    if diag_frames:
        diag_all = pd.concat(diag_frames, ignore_index=True)
        diag_all.to_csv(os.path.join(OUTPUT_DIR, 'bayes_diagnostics_all.csv'), index=False)
        if 'rhat' in diag_all.columns:
            print(f"\n  Convergence summary across all Bayesian models:")
            print(f"    Total models: {len(diag_all)}")
            print(f"    R-hat>1.05  : {(diag_all['rhat']>1.05).sum()}")
            if 'ess_bulk' in diag_all.columns:
                print(f"    ESS<400     : {(diag_all['ess_bulk']<400).sum()}")


# ═══════════════════════════════════════════════════════════════════
# PART 4: CLASSIFICATION — LOSO (LR + RF)
# ═══════════════════════════════════════════════════════════════════
hr("PART 4: CLASSIFICATION — LOSO (LR + RF)")

def run_loso(df, feat_cols, clf_name='LR', n_perm=500, seed=42):
    df = df.copy(); df['y'] = (df['Group']=='ASD').astype(int)
    if df['y'].sum()<4 or (1-df['y']).sum()<4: return None
    usable = [f for f in feat_cols if f in df.columns and df[f].notna().mean()>0.5]
    if len(usable)<2: return None
    df[usable] = df[usable].fillna(df[usable].median())
    clf = (LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced',
                              random_state=seed) if clf_name=='LR' else
           RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=seed, n_jobs=-1))
    pipe  = Pipeline([('sc', StandardScaler()), ('clf', clf)])
    y_true, y_score = [], []
    for pid in df['pid'].unique():
        test  = df[df['pid']==pid]; train = df[df['pid']!=pid]
        if len(train['y'].unique())<2: continue
        try:
            pipe.fit(train[usable].values, train['y'].values)
            y_score.extend(pipe.predict_proba(test[usable].values)[:,1].tolist())
            y_true.extend(test['y'].values.tolist())
        except: continue
    if len(set(y_true))<2: return None
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    perm = [roc_auc_score(rng.permuted(np.array(y_true)), y_score)
            for _ in range(n_perm)]
    p_perm = float((np.array(perm)>=auc).mean())
    cm = confusion_matrix(y_true, (np.array(y_score)>=0.5).astype(int))
    print(f"  [{clf_name}] AUC={auc:.3f}  AP={ap:.3f}  p_perm={p_perm:.4f}  "
          f"n_feat={len(usable)}")
    return {'auc': auc, 'ap': ap, 'perm_p': p_perm,
            'n_features': len(usable), 'n_subjects': df['pid'].nunique(),
            'y_true': y_true, 'y_score': y_score, 'perm_aucs': perm,
            'confusion_matrix': cm, 'clf': clf_name}

clf_results = {}

print("\n--- Combined child level ---")
for cname in ['LR', 'RF']:
    r = run_loso(child_df, CHILD_FEATS, clf_name=cname)
    if r: clf_results[f'combined_{cname}'] = r

print("\n--- Age-band stratified (child level) ---")
for band in AGE_BANDS:
    sub   = child_df[child_df['age_band']==band]
    asd_n = (sub['Group']=='ASD').sum()
    nan_n = (sub['Group']=='Non-ASD').sum()
    print(f"  {band}: ASD={asd_n}, Non-ASD={nan_n}", end=' ')
    if asd_n>=4 and nan_n>=4:
        r = run_loso(sub, CHILD_FEATS, clf_name='LR')
        if r: clf_results[f'{band}_LR'] = r
    else: print("→ skipped")

# RF feature importances (full dataset)
feat_importance_df = pd.DataFrame()
try:
    tmp = child_df.copy(); tmp['y'] = (tmp['Group']=='ASD').astype(int)
    usable = [f for f in CHILD_FEATS if tmp[f].notna().mean()>0.5]
    tmp[usable] = tmp[usable].fillna(tmp[usable].median())
    sc  = StandardScaler(); X = sc.fit_transform(tmp[usable].values)
    rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    rf.fit(X, tmp['y'].values)
    feat_importance_df = pd.DataFrame(
        {'feature': usable, 'importance': rf.feature_importances_}
    ).sort_values('importance', ascending=False)
    feat_importance_df['label'] = feat_importance_df['feature'].map(
        SHORT_LABELS).fillna(feat_importance_df['feature'])
    feat_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, 'rf_feature_importances.csv'), index=False)
    print("\nTop 10 RF importances:")
    for _, r in feat_importance_df.head(10).iterrows():
        print(f"  {r['feature']:<45} {r['importance']:.4f}")
except Exception as e:
    print(f"RF importance failed: {e}")

if clf_results:
    pd.DataFrame([{'subset': k, 'clf': v.get('clf',''), 'auc': v['auc'],
                   'ap': v['ap'], 'perm_p': v['perm_p'],
                   'n_features': v['n_features'], 'n_subjects': v['n_subjects']}
                  for k, v in clf_results.items()]).to_csv(
        os.path.join(OUTPUT_DIR, 'classification_summary.csv'), index=False)


# ═══════════════════════════════════════════════════════════════════
# PART 5: FIGURES
# ═══════════════════════════════════════════════════════════════════
hr("PART 5: FIGURES")

# Fig 1: Sample overview
print("  Fig 1: Sample overview...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Rocking Sample Overview', fontweight='bold')
ax = axes[0]
gc  = child_df['Group'].value_counts()
bars = ax.bar(GROUPS, [gc.get(g,0) for g in GROUPS],
              color=[COLORS[g] for g in GROUPS], width=0.5, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            str(int(bar.get_height())), ha='center', fontweight='bold')
ax.set_title('(a) Children per group'); ax.set_ylabel('N')
ax = axes[1]
beh_counts = feat_df.groupby(['age_band','Group']).size().unstack(fill_value=0)
x = np.arange(len(beh_counts)); w = 0.35
for i, grp in enumerate(GROUPS):
    if grp in beh_counts.columns:
        ax.bar(x+i*w, beh_counts[grp], w, color=COLORS[grp],
               label=grp, alpha=0.85, edgecolor='white')
ax.set_xticks(x+w/2)
ax.set_xticklabels([b.replace('mo','\nmo') for b in beh_counts.index], fontsize=8)
ax.set_title('(b) Clips per age band'); ax.set_ylabel('N'); ax.legend(fontsize=8)
ax = axes[2]
for grp in GROUPS:
    ax.hist(child_df[child_df['Group']==grp]['age_mo'],
            bins=10, alpha=0.6, color=COLORS[grp], label=grp, edgecolor='white')
for band, (lo,hi) in AGE_BANDS.items():
    ax.axvspan(lo, hi, alpha=0.1, color=BAND_COLORS[band], label=band)
ax.set_title('(c) Age distribution\n⚠ ASD-only at 26-38mo', fontsize=9)
ax.set_xlabel('Age (months)'); ax.legend(fontsize=7)
plt.tight_layout(); save_fig(fig, 'fig01_sample_overview.png')

# Fig 2: Violin plots — primary features
print("  Fig 2: Violin plots...")
DISP_FEATS = [f for f in ['mean_hip_x_amplitude','mean_hip_x_vel_mean',
    'mean_hip_x_band_power_0p3_2hz','mean_hip_x_spectral_entropy',
    'trunk_tilt_amplitude','trunk_tilt_band_power_0p3_2hz',
    'nose_x_amplitude','nose_x_spectral_entropy',
    'hip_2d_amplitude_max','bilateral_hip_x_corr',
    'mean_sh_x_amplitude','hip_x_y_ratio'] if f in feat_df.columns]
if DISP_FEATS and len(r2a):
    ncols = 4; nrows = int(np.ceil(len(DISP_FEATS)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    fig.suptitle('Rocking Kinematics — ASD vs Non-ASD', fontweight='bold')
    axes = axes.flatten()
    for i, feat in enumerate(DISP_FEATS):
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
        row = r2a[r2a['feature']==feat]
        if len(row):
            p_r = row['p_raw'].values[0]; p_f = row['p_fdr'].values[0]
            d   = row['cohens_d'].values[0]
            col = '#cc0000' if p_f<0.05 else ('#ff8800' if p_r<0.05 else 'gray')
            ax.text(0.5, 0.97, f'p={p_r:.3f}|FDR={p_f:.3f}|d={d:.2f}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=7.5, color=col)
            ymax = max(np.percentile(d2,95) for d2 in dg if len(d2))
            yr   = ymax - min(np.percentile(d2,5) for d2 in dg if len(d2))
            add_sig_bar(ax, 0, 1, ymax+yr*0.05, p_r, h=max(yr*0.04, 1e-6))
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_title(SHORT_LABELS.get(feat, feat), fontsize=9, fontweight='bold')
    for j in range(len(DISP_FEATS), len(axes)): axes[j].set_visible(False)
    fig.legend(handles=[mpatches.Patch(color=COLORS[g],label=g) for g in GROUPS],
               loc='upper right')
    plt.tight_layout(); save_fig(fig, 'fig02_violins.png')

# Fig 3: Effect sizes forest — bootstrapped CI
print("  Fig 3: Effect sizes forest...")
if len(r2a):
    ht = r2a.copy()
    ht['label'] = ht['feature'].map(SHORT_LABELS).fillna(ht['feature'])
    ht = ht.reindex(ht['cohens_d'].abs().sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(11, max(6, len(ht)*0.38)))
    bar_colors = [ASD_COLOR if d>0 else NONASD_COLOR for d in ht['cohens_d']]
    ax.barh(ht['label'], ht['cohens_d'], color=bar_colors,
            edgecolor='white', height=0.6, alpha=0.85)
    ax.errorbar(ht['cohens_d'], range(len(ht)),
                xerr=[ht['cohens_d']-ht['d_ci_lo'], ht['d_ci_hi']-ht['cohens_d']],
                fmt='none', color='black', lw=1.2, capsize=3, alpha=0.7)
    ax.axvline(0, color='black', lw=0.8)
    for t, ls in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
        ax.axvline(t, color='gray', lw=0.7, ls=ls, alpha=0.5)
        ax.axvline(-t, color='gray', lw=0.7, ls=ls, alpha=0.5)
    for j, (_, row) in enumerate(ht.iterrows()):
        if row.get('sig_fdr05'):
            ax.text(row['d_ci_hi']+0.02, j, '★', va='center', fontsize=11, color='gold')
        elif row.get('sig_raw05'):
            ax.text(row['d_ci_hi']+0.02, j, '●', va='center', fontsize=9)
    ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)  |  95% bootstrap CI")
    ax.set_title("Effect Sizes — Rocking  ★=FDR q<0.05  ●=raw p<0.05",
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
    plt.tight_layout(); save_fig(fig, 'fig03_effect_sizes.png')

# Fig 4: Age band bar charts
print("  Fig 4: Age band bars...")
KEY_FEATS = [f for f in ['mean_hip_x_amplitude','mean_hip_x_band_power_0p3_2hz',
                          'trunk_tilt_amplitude','nose_x_amplitude',
                          'mean_hip_x_spectral_entropy'] if f in feat_df.columns]
if KEY_FEATS:
    band_list = list(AGE_BANDS.keys())
    fig, axes = plt.subplots(len(KEY_FEATS), len(band_list),
                             figsize=(4.5*len(band_list), 4*len(KEY_FEATS)), sharey='row')
    fig.suptitle('Key Features by Age Band — ASD vs Non-ASD\n'
                 '⚠ 26-31mo and 32-38mo are ASD-only', fontweight='bold')
    for ri, feat in enumerate(KEY_FEATS):
        for ci, band in enumerate(band_list):
            ax  = axes[ri][ci]
            sub = feat_df[feat_df['age_band']==band]
            da  = sub[sub['Group']=='ASD'][feat].dropna().values
            dn  = sub[sub['Group']=='Non-ASD'][feat].dropna().values
            means = [da.mean() if len(da) else 0, dn.mean() if len(dn) else 0]
            sems  = [stats.sem(da) if len(da)>1 else 0,
                     stats.sem(dn) if len(dn)>1 else 0]
            ax.bar([0,1], means, yerr=sems,
                   color=[COLORS['ASD'],COLORS['Non-ASD']],
                   capsize=5, width=0.5, edgecolor='white', alpha=0.85)
            for j, (vals, xp) in enumerate([(da,0),(dn,1)]):
                if len(vals):
                    ax.scatter(xp+np.random.uniform(-0.1,0.1,len(vals)), vals,
                               color=list(COLORS.values())[j], alpha=0.4, s=10)
            if len(da)>=3 and len(dn)>=3:
                _, p = stats.mannwhitneyu(da, dn, alternative='two-sided')
                ymax = max(means)+max(sems)+abs(max(means))*0.05
                add_sig_bar(ax, 0, 1, ymax, p, h=max(abs(ymax)*0.04, 0.001))
            if ri==0: ax.set_title(band, fontsize=9)
            if ci==0: ax.set_ylabel(SHORT_LABELS.get(feat,feat)[:20], fontsize=7)
            ax.set_xticks([0,1]); ax.set_xticklabels(['ASD','NASD'], fontsize=8)
            ax.text(0.5,-0.22,f'n={len(da)}/{len(dn)}',
                    transform=ax.transAxes, ha='center', fontsize=7.5, color='gray')
    plt.tight_layout(); save_fig(fig, 'fig04_age_bands.png')

# Fig 5: Developmental trajectories
print("  Fig 5: Trajectories...")
TRAJ_FEATS = [f for f in ['mean_hip_x_amplitude','mean_hip_x_band_power_0p3_2hz',
                            'trunk_tilt_amplitude','nose_x_amplitude',
                            'mean_hip_x_spectral_entropy','trunk_tilt_std']
              if f in feat_df.columns]
if TRAJ_FEATS:
    ncols = 3; nrows = int(np.ceil(len(TRAJ_FEATS)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    fig.suptitle('Developmental Trajectories — Rocking\n⚠ ASD-only at 26-38mo',
                 fontweight='bold')
    axes = axes.flatten()
    for i, feat in enumerate(TRAJ_FEATS):
        ax = axes[i]
        for grp in GROUPS:
            sub = feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
            if len(sub)<3: continue
            ax.scatter(sub['age_mo'], sub[feat], color=COLORS[grp], alpha=0.3, s=15)
            if len(sub)>=5:
                m, b, r, p, _ = stats.linregress(sub['age_mo'], sub[feat])
                xr = np.linspace(sub['age_mo'].min(), sub['age_mo'].max(), 100)
                ax.plot(xr, m*xr+b, color=COLORS[grp], lw=2.5,
                        label=f'{grp} r={r:.2f} p={p:.3f}')
        for band, (lo,hi) in AGE_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.07, color=BAND_COLORS[band])
        ax.set_xlabel('Age (months)')
        ax.set_ylabel(SHORT_LABELS.get(feat, feat)[:20], fontsize=9)
        ax.set_title(SHORT_LABELS.get(feat, feat), fontsize=9, fontweight='bold')
        ax.legend(fontsize=8)
    for j in range(len(TRAJ_FEATS), len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); save_fig(fig, 'fig05_trajectories.png')

# Fig 6: Child-level boxplots
print("  Fig 6: Child-level boxplots...")
CL_FEATS = [f for f in ['mean_hip_x_amplitude','mean_hip_x_band_power_0p3_2hz',
                          'trunk_tilt_amplitude','nose_x_amplitude',
                          'mean_hip_x_spectral_entropy'] if f in child_df.columns]
if CL_FEATS:
    fig, axes = plt.subplots(1, len(CL_FEATS), figsize=(4*len(CL_FEATS), 5))
    if len(CL_FEATS)==1: axes = [axes]
    fig.suptitle('Child-Level Averages (each dot = 1 child)', fontweight='bold')
    for i, feat in enumerate(CL_FEATS):
        ax = axes[i]
        for j, grp in enumerate(GROUPS):
            vals = child_df[child_df['Group']==grp][feat].dropna().values
            if len(vals)==0: continue
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
            ymax = child_df[feat].dropna().quantile(0.97)
            add_sig_bar(ax, 0, 1, ymax, p, h=abs(ymax)*0.04+1e-6)
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_title(SHORT_LABELS.get(feat, feat), fontsize=9, fontweight='bold')
        ax.text(0.5,-0.12,f'n={len(da)}/{len(dn)}',
                transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    plt.tight_layout(); save_fig(fig, 'fig06_child_boxplots.png')

# Fig 7: Consensus heatmap
print("  Fig 7: Consensus heatmap...")
if len(consensus_all)>0:
    p_cols    = [c for c in consensus_all.columns if c.startswith('p_')]
    heat_data = consensus_all.set_index('feature')[p_cols].head(20)
    heat_log  = -np.log10(heat_data.clip(lower=1e-5, upper=1.0).astype(float))
    fig, ax   = plt.subplots(figsize=(len(p_cols)*2+2, max(6,len(heat_data)*0.4)))
    im = ax.imshow(heat_log.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=4)
    ax.set_xticks(range(len(p_cols)))
    ax.set_xticklabels([c.replace('p_','') for c in p_cols], rotation=30, ha='right')
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels([SHORT_LABELS.get(f,f) for f in heat_data.index], fontsize=9)
    for i in range(heat_log.shape[0]):
        for j in range(heat_log.shape[1]):
            raw_p = heat_data.values[i,j]
            if not np.isnan(raw_p):
                ax.text(j, i, f'{raw_p:.3f}{"*" if raw_p<0.05 else ""}',
                        ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='-log10(p)')
    ax.set_title('Consensus p-values across methods (top 20 features)',
                 fontweight='bold')
    plt.tight_layout(); save_fig(fig, 'fig07_consensus_heatmap.png')

# Fig 8: ICC bar chart
print("  Fig 8: ICC...")
if len(icc_df)>0:
    top_icc = icc_df.head(20).copy()
    top_icc['label'] = top_icc['feature'].map(SHORT_LABELS).fillna(top_icc['feature'])
    fig, ax  = plt.subplots(figsize=(10, max(5,len(top_icc)*0.4)))
    cols_icc = ['#2ecc71' if v>0.1 else '#e74c3c' for v in top_icc['ICC']]
    ax.barh(top_icc['label'], top_icc['ICC'],
            color=cols_icc, edgecolor='white', height=0.65)
    ax.axvline(0.1, color='orange', lw=1.5, ls='--', label='ICC=0.10 threshold')
    ax.set_xlabel('ICC')
    ax.set_title('Intraclass Correlation — Within-Child Clustering\n'
                 'Green=clustering significant (ICC>0.10)', fontweight='bold')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig08_icc.png')

# Fig 9: Consistency gate
print("  Fig 9: Consistency gate...")
if 'cons_gate_df' in dir() and len(cons_gate_df)>0:
    fig, ax = plt.subplots(figsize=(10, max(4,len(cons_gate_df)*0.45)))
    cols_cg = [ASD_COLOR if v else NONASD_COLOR for v in cons_gate_df['consistent']]
    ax.barh(
        cons_gate_df['feature'].map(SHORT_LABELS).fillna(cons_gate_df['feature']),
        cons_gate_df['n_same_direction']/cons_gate_df['n_bands_tested'],
        color=cols_cg, edgecolor='white', height=0.6)
    ax.axvline(1.0, color='green', lw=1.5, ls='--', label='All bands consistent')
    ax.axvline(0.5, color='orange', lw=1, ls=':', label='50%')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Fraction of age bands with same direction')
    ax.set_title('Consistency Gate — Red=failed (age-band direction inconsistent)',
                 fontweight='bold')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig09_consistency_gate.png')

# Fig 10: Bayesian forest
print("  Fig 10: Bayesian forest...")
if 'combined' in bayes_main_results and len(bayes_main_results['combined'])>0:
    bdf = bayes_main_results['combined'].copy()
    bdf['label'] = bdf['feature'].map(SHORT_LABELS).fillna(bdf['feature'])
    bdf = bdf.sort_values('b_group_mean')
    fig, ax = plt.subplots(figsize=(13, max(5, len(bdf)*0.5)))
    for j, (_, row) in enumerate(bdf.iterrows()):
        col = ASD_COLOR if row['b_group_mean']>0 else NONASD_COLOR
        ax.plot([row['hdi94_lo'],row['hdi94_hi']], [j,j],
                color=col, lw=2.5, alpha=0.8)
        ax.scatter(row['b_group_mean'], j, color=col, s=70, zorder=5)
        ax.plot(row['hdi94_lo'], j, '|', color=col, markersize=8)
        ax.plot(row['hdi94_hi'], j, '|', color=col, markersize=8)
        bf  = float(row['bf10']) if not np.isnan(float(row['bf10'])) else 0
        lbl = f"BF={bf:.1f}"
        if not row.get('converged', True): lbl += ' ⚠'
        if not row.get('bf_robust', True): lbl += ' [prior-sens]'
        ax.text(row['hdi94_hi']+0.01, j, lbl, va='center', fontsize=7)
    ax.axvline(0, color='black', lw=1.2, ls='--')
    ax.set_yticks(range(len(bdf))); ax.set_yticklabels(bdf['label'], fontsize=9)
    ax.set_xlabel('Posterior mean  |  94% HDI  (standardised)')
    ax.set_title('Bayesian Hierarchical LMM — Rocking\n'
                 '⚠=convergence  [prior-sens]=BF changed across priors',
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
    plt.tight_layout(); save_fig(fig, 'fig10_bayes_forest.png')

# Fig 11: Prior sensitivity
print("  Fig 11: Prior sensitivity...")
sens_path = os.path.join(OUTPUT_DIR, 'bayes_sensitivity.csv')
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
            ax.axhline(3,  color='green', lw=1, ls='--', label='BF=3')
            ax.axhline(10, color='green', lw=1, ls='-',  label='BF=10')
            ax.axhline(1,  color='gray',  lw=0.8, ls=':')
            ax.set_xlabel('Prior SD'); ax.set_ylabel('BF10')
            ax.set_title(SHORT_LABELS.get(feat,feat)[:25], fontsize=9)
            ax.legend(fontsize=7)
        for j in range(len(feats_s), len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); save_fig(fig, 'fig11_prior_sensitivity.png')

# Fig 12: Classification ROC
print("  Fig 12: ROC curves...")
if clf_results:
    keys  = list(clf_results.keys()); n = len(keys)
    ncols = min(n, 4); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    if nrows*ncols==1: axes = np.array([[axes]])
    elif nrows==1: axes = axes.reshape(1,-1)
    fig.suptitle('Classification ROC — Child-Level LOSO', fontweight='bold')
    for i, key in enumerate(keys):
        r  = clf_results[key]; ax = axes[i//ncols][i%ncols]
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
    plt.tight_layout(); save_fig(fig, 'fig12_roc.png')

# Fig 13: RF feature importances
print("  Fig 13: RF importances...")
if len(feat_importance_df)>0:
    top20 = feat_importance_df.head(20)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top20['label'], top20['importance'],
            color=ASD_COLOR, edgecolor='white', height=0.65, alpha=0.85)
    ax.set_xlabel('Mean decrease in impurity')
    ax.set_title('RF Feature Importances (child level)', fontweight='bold')
    ax.axvline(top20['importance'].mean(), color='gray', lw=1, ls='--', label='Mean')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig13_rf_importances.png')

# Fig 14: Age confound cross-plot (Bayesian)
print("  Fig 14: Age confound check...")
main_path = os.path.join(OUTPUT_DIR, 'bayes_main.csv')
conf_path = os.path.join(OUTPUT_DIR, 'bayes_age_confound.csv')
if os.path.isfile(main_path) and os.path.isfile(conf_path):
    bm   = pd.read_csv(main_path)[['feature','b_group_mean','bf10']].rename(
           columns={'b_group_mean':'b_group','bf10':'bf10_group'})
    ba   = pd.read_csv(conf_path)[['feature','b_age_mean','bf10']].rename(
           columns={'b_age_mean':'b_age','bf10':'bf10_age'})
    cmp2 = bm.merge(ba, on='feature')
    if len(cmp2)>=3:
        fig, ax = plt.subplots(figsize=(9,7))
        ax.scatter(
            np.log10(cmp2['bf10_age'].clip(lower=0.01)),
            np.log10(cmp2['bf10_group'].clip(lower=0.01)),
            c=[ASD_COLOR if g>0 else NONASD_COLOR for g in cmp2['b_group']],
            s=70, alpha=0.85, edgecolors='gray', lw=0.5)
        for thresh, ls_, lbl in [(np.log10(3),'--','BF=3'),(np.log10(10),'-','BF=10')]:
            ax.axhline(thresh, color='green',  lw=0.9, ls=ls_,
                       alpha=0.6, label=f'Group {lbl}')
            ax.axvline(thresh, color='orange', lw=0.9, ls=ls_,
                       alpha=0.6, label=f'Age {lbl}')
        ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
        for _, row in cmp2.iterrows():
            ax.annotate(SHORT_LABELS.get(row['feature'],
                                         row['feature'].replace('_',' ')[:18]),
                        (np.log10(max(float(row['bf10_age']),0.01)),
                         np.log10(max(float(row['bf10_group']),0.01))),
                        fontsize=6.5, alpha=0.8, xytext=(3,3),
                        textcoords='offset points')
        ax.set_xlabel('log₁₀(BF₁₀) — Age-only model\n(large = age drives the feature)')
        ax.set_ylabel('log₁₀(BF₁₀) — Group model (age-adjusted)\n'
                      '(large = Group drives the feature beyond age)')
        ax.set_title('Age Confound Check — Rocking\n'
                     'Top-right: BOTH age and group → interpret carefully\n'
                     'Top-left: Group effect NOT driven by age → most trustworthy',
                     fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout(); save_fig(fig, 'fig14_age_confound_check.png')

# Fig 15: Feature × Group × Age band heatmap
print("  Fig 15: Feature heatmap...")
HEAT_F = [f for f in ['mean_hip_x_amplitude','mean_hip_x_vel_mean',
                        'mean_hip_x_band_power_0p3_2hz','mean_hip_x_spectral_entropy',
                        'trunk_tilt_amplitude','trunk_tilt_band_power_0p3_2hz',
                        'nose_x_amplitude','bilateral_hip_x_corr']
          if f in feat_df.columns]
if HEAT_F:
    cells = []; cell_labels = []
    for grp in GROUPS:
        for band in AGE_BANDS:
            sub = feat_df[(feat_df['Group']==grp)&(feat_df['age_band']==band)]
            cells.append(sub[HEAT_F].median() if len(sub) else
                         pd.Series([np.nan]*len(HEAT_F), index=HEAT_F))
            cell_labels.append(f'{grp}\n{band}')
    heat_raw = pd.DataFrame(cells, index=cell_labels, columns=HEAT_F)
    heat_z   = (heat_raw - heat_raw.mean()) / (heat_raw.std()+1e-8)
    fig, ax  = plt.subplots(figsize=(len(HEAT_F)*1.1+1, len(cell_labels)*0.75+1))
    im = ax.imshow(heat_z.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(HEAT_F)))
    ax.set_xticklabels([SHORT_LABELS.get(f,f) for f in HEAT_F],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(cell_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label='Z-score', fraction=0.03)
    for i in range(len(cell_labels)):
        for j in range(len(HEAT_F)):
            v = heat_raw.values[i,j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='black')
    ax.set_title('Feature × Group × Age Band Heatmap (median, z-scored)',
                 fontweight='bold')
    plt.tight_layout(); save_fig(fig, 'fig15_feature_heatmap.png')


# ═══════════════════════════════════════════════════════════════════
# PART 6: FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
hr("FINAL SUMMARY")
print(f"\nOutput : {OUTPUT_DIR}")
print(f"Figures: {FIGURE_DIR}\n")
print("--- CSVs ---")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith('.csv'):
        try:
            tmp = pd.read_csv(os.path.join(OUTPUT_DIR,fname))
            print(f"  {fname:<65} {tmp.shape[0]:>5}r × {tmp.shape[1]:>3}c")
        except: print(f"  {fname}")
print("\n--- Figures ---")
for fname in sorted(os.listdir(FIGURE_DIR)):
    if fname.endswith('.png'):
        sz = os.path.getsize(os.path.join(FIGURE_DIR,fname))/1024
        print(f"  {fname:<55} {sz:.0f} KB")
print("\n--- KEY RESULTS ---")
for name, res in [('LME_KR', lme_kr_all), ('CR2', cr2_all), ('GEE', gee_all),
                  ('ChildPerm', perm_all), ('WildBoot', boot_all), ('MWU', r2a)]:
    if len(res)>0:
        print(f"  {name:<12}: sig_raw={res['sig_raw05'].sum():>3}  "
              f"FDR={res['sig_fdr05'].sum():>3}")
if 'cons_gate_df' in dir() and len(cons_gate_df)>0:
    print(f"\nConsistency gate: {len(consistent_feats)}/{len(sig_feats)} passed")
    for f in consistent_feats: print(f"  ✓ {f}")
if clf_results:
    print("\nClassification (child-level LOSO):")
    for k, v in clf_results.items():
        print(f"  {k:<40} AUC={v['auc']:.3f}  p_perm={v['perm_p']:.4f}")

hr("ROCKING ANALYSIS COMPLETE (Parts 1-6)")
