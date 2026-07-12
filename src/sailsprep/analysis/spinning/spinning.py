#!/usr/bin/env python3
"""
spinning analysis .py
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
import json
import os
import re
import traceback
import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.signal import butter, filtfilt, welch
from scipy.stats import gaussian_kde
from scipy.stats import norm as spnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests

from sailsprep.analysis.common.banners import hr_v2 as hr
from sailsprep.analysis.common.parsing import extract_pid, parse_timestamps_v2 as parse_timestamps
from sailsprep.analysis.common.keypoints import get_kp, assign_age_band
from sailsprep.analysis.common.effect_size import cohen_d_v3 as cohen_d
from sailsprep.analysis.common.signal_processing import butter_lp_v2 as butter_lp
from sailsprep.analysis.common.misc import run_spearman_age
from sailsprep.analysis.common.handflapping_spinning_stats import bootstrap_ci_d

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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
OUTPUT_DIR = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/analysis/spinning/v3"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

FPS            = 15.0
MIN_CONF       = 0.3
SPINNING_LABELS = {'spinning'}

AGE_BANDS = {
    '11-18mo': (11, 18),
    '19-31mo': (19, 31),
    '32-38mo': (32, 38),
}
# Bands with enough subjects for stratified stats
STAT_BANDS = ['11-18mo', '32-38mo']

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
    'nose':           'kp_000',
}

ASD_COLOR    = '#E05C5C'; NONASD_COLOR = '#5B8DB8'
ASD_LIGHT    = '#F2AEAE'; NONASD_LIGHT = '#A8C8E8'
COLORS       = {'ASD': ASD_COLOR, 'Non-ASD': NONASD_COLOR}
COLORS_LIGHT = {'ASD': ASD_LIGHT, 'Non-ASD': NONASD_LIGHT}
BAND_COLORS  = {'11-18mo': '#7B5EA7', '19-31mo': '#4A9B6F', '32-38mo': '#D47C2A'}
GROUPS       = ['ASD', 'Non-ASD']

SPINNING_SHORT = {
    'sw_amplitude':               'Shoulder Width Amp',
    'sw_cv':                      'Shoulder Width CV',
    'sw_dom_freq':                'SW Dom Freq',
    'sw_spectral_entropy':        'SW Entropy',
    'sw_band_power_0p5_2p5hz':    'SW 0.5-2.5 Hz',
    'sw_vel_mean':                'SW Vel Mean',
    'sw_vel_max':                 'SW Vel Max',
    'ls_x_amplitude':             'L Shoulder X Amp',
    'ls_x_spectral_entropy':      'L Shoulder X Entropy',
    'ls_x_band_power_0p5_2p5hz':  'L Shoulder X 0.5-2.5 Hz',
    'ls_x_dom_freq':              'L Shoulder X Freq',
    'rs_x_amplitude':             'R Shoulder X Amp',
    'rs_x_spectral_entropy':      'R Shoulder X Entropy',
    'rs_x_band_power_0p5_2p5hz':  'R Shoulder X 0.5-2.5 Hz',
    'ls_y_amplitude':             'L Shoulder Y Amp',
    'ls_y_spectral_entropy':      'L Shoulder Y Entropy',
    'ls_y_band_power_0p5_2p5hz':  'L Shoulder Y 0.5-2.5 Hz',
    'rs_y_amplitude':             'R Shoulder Y Amp',
    'rs_y_spectral_entropy':      'R Shoulder Y Entropy',
    'rs_y_band_power_0p5_2p5hz':  'R Shoulder Y 0.5-2.5 Hz',
    'lw_x_amplitude':             'L Wrist X Amp',
    'lw_x_band_power_0p5_2p5hz':  'L Wrist X 0.5-2.5 Hz',
    'lw_x_spectral_entropy':      'L Wrist X Entropy',
    'rw_x_amplitude':             'R Wrist X Amp',
    'rw_x_band_power_0p5_2p5hz':  'R Wrist X 0.5-2.5 Hz',
    'rw_x_spectral_entropy':      'R Wrist X Entropy',
    'lw_y_band_power_0p5_2p5hz':  'L Wrist Y 0.5-2.5 Hz',
    'rw_y_band_power_0p5_2p5hz':  'R Wrist Y 0.5-2.5 Hz',
    'nose_x_amplitude':           'Nose X Amp',
    'nose_x_spectral_entropy':    'Nose X Entropy',
    'nose_x_dom_freq':            'Nose X Freq',
    'sh_x_LR_corr':               'Shoulder X L-R Corr',
    'sh_x_LR_amp_diff':           'Shoulder X LR Diff',
    'sh_x_LR_phase_lag':          'Shoulder Phase Lag',
    'wrist_x_LR_corr':            'Wrist X L-R Corr',
    'wrist_x_LR_amp_diff':        'Wrist X LR Diff',
    'shoulder_to_hip_x_ratio':    'Sh/Hip X Ratio',
    'shoulder_x_amp':             'Shoulder X Amp (mid)',
    'spin_intensity_mean':        'Spin Intensity Mean',
    'spin_intensity_max':         'Spin Intensity Max',
    'ls_x_vel_max':               'L Shoulder X Vel Max',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

# ═══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURE_DIR, name)); plt.close(fig)
    print(f"  Saved {name}")





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

def spectral_features(arr, fps, lo=0.5, hi=2.5):
    if len(arr) < 16: return np.nan, np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq   = freqs[np.argmax(psd)]
        psd_n      = psd / (psd.sum() + 1e-12)
        entropy    = -np.sum(psd_n[psd_n > 0] * np.log2(psd_n[psd_n > 0]))
        band_mask  = (freqs >= lo) & (freqs <= hi)
        band_pwr   = psd[band_mask].sum() / (psd.sum() + 1e-12)
        return float(dom_freq), float(entropy), float(band_pwr)
    except:
        return np.nan, np.nan, np.nan



def fdr_annotate(df_res, p_col):
    df_res = df_res.copy()
    if len(df_res) > 1:
        _, p_fdr, _, _ = multipletests(df_res[p_col].fillna(1), method='fdr_bh')
        df_res['p_fdr'] = p_fdr
    else:
        df_res['p_fdr'] = df_res[p_col]
    df_res['sig_fdr05'] = df_res['p_fdr'] < 0.05
    df_res['sig_raw05'] = df_res[p_col] < 0.05
    return df_res


def add_sig_bar(ax, x1, x2, y, p, h=0.02):
    label = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col   = '#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color='black')
    ax.text((x1+x2)/2, y+h*1.05, label, ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════
# PART 0: LOAD METADATA
# ═══════════════════════════════════════════════════════════════════
hr("PART 0: LOAD METADATA")

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

spin_rmm = rmm_labeled[rmm_labeled['label_lower'].isin(SPINNING_LABELS)].copy()
spin_rmm['hrnet_path'] = spin_rmm['csv_bids_processed'].map(video_to_hrnet)
spin_rmm = spin_rmm[
    spin_rmm['hrnet_path'].apply(lambda p: isinstance(p, str) and os.path.isfile(p))
].copy()
spin_rmm['age_band'] = spin_rmm['age_mo'].apply(assign_age_band)

print(f"Spinning clips with pose: {len(spin_rmm)}")
print(spin_rmm.groupby(['age_band', 'Group']).size().reset_index(name='n').to_string(index=False))
print(f"\nChildren: ASD={spin_rmm[spin_rmm['Group']=='ASD']['pid'].nunique()}  "
      f"Non-ASD={spin_rmm[spin_rmm['Group']=='Non-ASD']['pid'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# PART 1: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
hr("PART 1: KINEMATIC FEATURE EXTRACTION")

def extract_spinning_features(pose_frames, frame_indices, ann_fps=FPS):
    ls_x, ls_y = [], []
    rs_x, rs_y = [], []
    lw_x, lw_y = [], []
    rw_x, rw_y = [], []
    le_x, re_x = [], []
    lh_x, rh_x = [], []
    nose_x      = []
    sw_series   = []
    conf_vals   = []
    n_valid = 0

    for fi in frame_indices:
        fk = str(fi)
        if fk not in pose_frames: continue
        fd    = pose_frames[fk]
        scale = get_scale(fd)
        if scale is None: continue
        ls  = get_kp(fd, KP['left_shoulder'])
        rs  = get_kp(fd, KP['right_shoulder'])
        lw  = get_kp(fd, KP['left_wrist'])
        rw  = get_kp(fd, KP['right_wrist'])
        le  = get_kp(fd, KP['left_elbow'])
        re_ = get_kp(fd, KP['right_elbow'])
        lh  = get_kp(fd, KP['left_hip'])
        rh  = get_kp(fd, KP['right_hip'])
        ns  = get_kp(fd, KP['nose'])
        if ls is None and rs is None and lw is None and rw is None: continue
        n_valid += 1
        if ls: ls_x.append(ls['x']/scale); ls_y.append(ls['y']/scale); conf_vals.append(ls['confidence'])
        if rs: rs_x.append(rs['x']/scale); rs_y.append(rs['y']/scale); conf_vals.append(rs['confidence'])
        if lw: lw_x.append(lw['x']/scale); lw_y.append(lw['y']/scale)
        if rw: rw_x.append(rw['x']/scale); rw_y.append(rw['y']/scale)
        if le: le_x.append(le['x']/scale)
        if re_: re_x.append(re_['x']/scale)
        if lh: lh_x.append(lh['x']/scale)
        if rh: rh_x.append(rh['x']/scale)
        if ns: nose_x.append(ns['x']/scale)
        if ls and rs:
            sw_series.append(np.sqrt((ls['x']-rs['x'])**2+(ls['y']-rs['y'])**2)/scale)

    if n_valid < 5: return None

    rec = {
        'n_valid_frames': n_valid,
        'n_total_frames': len(frame_indices),
        'pct_valid':      n_valid/len(frame_indices),
        'duration_sec':   len(frame_indices)/ann_fps,
        'mean_conf':      float(np.mean(conf_vals)) if conf_vals else np.nan,
    }

    def jfeats(arr, name):
        a = np.array(arr)
        if len(a) < 5: return
        rec[f'{name}_amplitude'] = float(np.ptp(a))
        rec[f'{name}_std']       = float(np.std(a))
        rec[f'{name}_mean']      = float(np.mean(a))
        if len(a) >= 8:
            try:
                sm  = butter_lp(a, fs=ann_fps)
                vel = np.diff(sm) * ann_fps
                rec[f'{name}_vel_mean'] = float(np.mean(np.abs(vel)))
                rec[f'{name}_vel_max']  = float(np.max(np.abs(vel)))
                if len(vel) >= 4:
                    acc = np.diff(vel) * ann_fps
                    rec[f'{name}_acc_mean'] = float(np.mean(np.abs(acc)))
            except: pass
        df_f, se, bp = spectral_features(a, ann_fps)
        rec[f'{name}_dom_freq']             = df_f
        rec[f'{name}_spectral_entropy']     = se
        rec[f'{name}_band_power_0p5_2p5hz'] = bp

    for arr, name in [
        (ls_x,'ls_x'), (ls_y,'ls_y'),
        (rs_x,'rs_x'), (rs_y,'rs_y'),
        (lw_x,'lw_x'), (lw_y,'lw_y'),
        (rw_x,'rw_x'), (rw_y,'rw_y'),
        (nose_x,'nose_x'),
    ]:
        jfeats(arr, name)

    if len(sw_series) >= 5:
        sw = np.array(sw_series)
        rec['sw_mean']      = float(np.mean(sw))
        rec['sw_amplitude'] = float(np.ptp(sw))
        rec['sw_std']       = float(np.std(sw))
        rec['sw_cv']        = float(np.std(sw) / (np.mean(sw) + 1e-8))
        df_f, se, bp = spectral_features(sw, ann_fps)
        rec['sw_dom_freq']             = df_f
        rec['sw_spectral_entropy']     = se
        rec['sw_band_power_0p5_2p5hz'] = bp
        if len(sw) >= 8:
            try:
                sm  = butter_lp(sw, fs=ann_fps)
                vel = np.diff(sm) * ann_fps
                rec['sw_vel_mean'] = float(np.mean(np.abs(vel)))
                rec['sw_vel_max']  = float(np.max(np.abs(vel)))
            except: pass

    if len(ls_x) >= 5 and len(rs_x) >= 5:
        ml = min(len(ls_x), len(rs_x))
        xl = np.array(ls_x[:ml]); xr = np.array(rs_x[:ml])
        rec['sh_x_LR_corr']     = float(np.corrcoef(xl, xr)[0, 1])
        rec['sh_x_LR_amp_diff'] = float(abs(np.ptp(xl) - np.ptp(xr)))
        try:
            xcorr = np.correlate(xl-xl.mean(), xr-xr.mean(), mode='full')
            lags  = np.arange(-(ml-1), ml)
            rec['sh_x_LR_phase_lag'] = float(abs(lags[np.argmax(xcorr)]) / ann_fps)
        except: pass

    if len(lw_x) >= 5 and len(rw_x) >= 5:
        ml = min(len(lw_x), len(rw_x))
        xl = np.array(lw_x[:ml]); xr = np.array(rw_x[:ml])
        rec['wrist_x_LR_corr']     = float(np.corrcoef(xl, xr)[0, 1])
        rec['wrist_x_LR_amp_diff'] = float(abs(np.ptp(xl) - np.ptp(xr)))

    if lh_x and rh_x and ls_x and rs_x:
        ml_h = min(len(lh_x), len(rh_x))
        ml_s = min(len(ls_x), len(rs_x))
        hip_mid_x_amp = float(np.ptp((np.array(lh_x[:ml_h])+np.array(rh_x[:ml_h]))/2))
        sh_mid_x_amp  = float(np.ptp((np.array(ls_x[:ml_s])+np.array(rs_x[:ml_s]))/2))
        rec['hip_x_amp']               = hip_mid_x_amp
        rec['shoulder_x_amp']          = sh_mid_x_amp
        rec['shoulder_to_hip_x_ratio'] = float(sh_mid_x_amp / (hip_mid_x_amp + 1e-8))

    ls_combo = float(np.sqrt(np.ptp(ls_x)**2+np.ptp(ls_y)**2)) if ls_x and ls_y else np.nan
    rs_combo = float(np.sqrt(np.ptp(rs_x)**2+np.ptp(rs_y)**2)) if rs_x and rs_y else np.nan
    lw_combo = float(np.sqrt(np.ptp(lw_x)**2+np.ptp(lw_y)**2)) if lw_x and lw_y else np.nan
    rw_combo = float(np.sqrt(np.ptp(rw_x)**2+np.ptp(rw_y)**2)) if rw_x and rw_y else np.nan
    vals_combo = [v for v in [ls_combo, rs_combo, lw_combo, rw_combo] if not np.isnan(v)]
    if vals_combo:
        rec['spin_intensity_mean'] = float(np.mean(vals_combo))
        rec['spin_intensity_max']  = float(np.max(vals_combo))

    return rec


all_features = []
n_ok = n_fail_ts = n_fail_pose = n_fail_kp = 0

for _, row in spin_rmm.iterrows():
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
        feats = extract_spinning_features(frames, fidx, ann_fps)
        if feats is None: n_fail_kp += 1; continue
        n_ok += 1
        feats.update({
            'pid': row['pid'], 'Group': row['Group'],
            'age_mo': row['age_mo'], 'age_band': row['age_band'],
            'clip': row.get('clip_filename', ''),
        })
        all_features.append(feats)

print(f"\nExtraction: ok={n_ok}  fail_ts={n_fail_ts}  "
      f"fail_pose={n_fail_pose}  fail_kp={n_fail_kp}")
if n_ok == 0:
    print("ERROR: No features extracted."); import sys; sys.exit(1)

feat_df = pd.DataFrame(all_features)
feat_df.to_csv(os.path.join(OUTPUT_DIR, 'clip_level_features.csv'), index=False)

META_COLS = {'pid','Group','age_mo','age_band','clip',
             'n_valid_frames','n_total_frames','pct_valid','duration_sec','mean_conf'}
FEAT_COLS = [c for c in feat_df.columns if c not in META_COLS]

PRIMARY_FEATS = [f for f in FEAT_COLS if any(x in f for x in [
    'sw_amplitude','sw_cv','sw_dom_freq','sw_spectral_entropy','sw_band_power',
    'sw_vel_mean','sw_vel_max',
    'ls_x_amplitude','ls_x_spectral_entropy','ls_x_band_power',
    'rs_x_amplitude','rs_x_spectral_entropy','rs_x_band_power',
    'ls_y_amplitude','ls_y_spectral_entropy','ls_y_band_power',
    'rs_y_amplitude','rs_y_spectral_entropy','rs_y_band_power',
    'lw_x_amplitude','lw_x_band_power','lw_x_spectral_entropy',
    'rw_x_amplitude','rw_x_band_power','rw_x_spectral_entropy',
    'lw_y_band_power','rw_y_band_power',
    'nose_x_amplitude','nose_x_spectral_entropy','nose_x_dom_freq',
    'sh_x_LR_corr','wrist_x_LR_corr',
    'sh_x_LR_amp_diff','sh_x_LR_phase_lag',
    'shoulder_to_hip_x_ratio','shoulder_x_amp',
    'spin_intensity_mean','spin_intensity_max',
    'ls_x_vel_max','ls_x_dom_freq',
])]
PRIMARY_FEATS = [f for f in PRIMARY_FEATS if f in feat_df.columns]

def make_child_df(clip_df):
    fc = [f for f in PRIMARY_FEATS if f in clip_df.columns]
    agg = clip_df.groupby(['pid', 'Group'])[fc].mean().reset_index()
    agg['n_clips']  = clip_df.groupby(['pid', 'Group']).size().values
    agg['age_mo']   = clip_df.groupby(['pid', 'Group'])['age_mo'].first().values
    agg['age_band'] = clip_df.groupby(['pid', 'Group'])['age_band'].first().values
    return agg

child_df = make_child_df(feat_df)
child_df.to_csv(os.path.join(OUTPUT_DIR, 'child_level_features.csv'), index=False)
print(f"\nclip_level_features.csv: {len(feat_df)} rows")
print(f"child_level_features.csv: {len(child_df)} children")
print(feat_df.groupby(['Group', 'age_band']).size().reset_index(name='n').to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# PART 2: STATISTICAL ANALYSIS — FULL BATTERY
# ═══════════════════════════════════════════════════════════════════
hr("PART 2: STATISTICAL ANALYSIS")

# ── Step 0: ICC (Intraclass Correlation) ─────────────────────────
def compute_icc(clip_df, feat_cols):
    """Measures within-child consistency: does ICC > 0.10 justify LME?"""
    records = []
    for feat in feat_cols:
        sub = clip_df[['pid', feat]].dropna()
        if len(sub) < 10: continue
        groups = [g[feat].values for _, g in sub.groupby('pid') if len(g) >= 2]
        if len(groups) < 5: continue
        try:
            n_total = sum(len(g) for g in groups); k = len(groups)
            n0 = (n_total - sum(len(g)**2/n_total for g in groups)) / (k-1)
            grand = np.concatenate(groups)
            ms_between = np.sum([len(g)*(np.mean(g)-np.mean(grand))**2
                                 for g in groups]) / (k-1)
            ms_within  = np.sum([np.sum((g-np.mean(g))**2)
                                  for g in groups]) / (n_total-k)
            icc = max(0.0, (ms_between - ms_within) /
                      (ms_between + (n0-1)*ms_within))
            f_stat, _ = stats.f_oneway(*groups)
            records.append({'feature': feat, 'ICC': round(icc, 4),
                            'f_stat': round(f_stat, 3)})
        except: pass
    return pd.DataFrame(records).sort_values('ICC', ascending=False)

print("\n--- Step 0: ICC ---")
icc_df = compute_icc(feat_df, PRIMARY_FEATS)
icc_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_icc.csv'), index=False)
print(icc_df.head(10).to_string(index=False))
print(f"  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)} features — clustering matters")

# ── Step 1: LME + Kenward-Roger ──────────────────────────────────
def run_lme_kr(clip_df, feat_cols, subset_label='combined'):
    """
    Clip-level LME: feature ~ Group_bin + age_mo_c + (1|pid)
    Uses Kenward-Roger correction (rpy2/lmerTest) if available,
    falls back to statsmodels MixedLM.
    """
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo_c',feat]].dropna(
                subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min() < 3: continue
        av = sub[sub['Group_bin']==1][feat].values
        nv = sub[sub['Group_bin']==0][feat].values
        d  = cohen_d(av, nv)
        ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        p_val = np.nan; coef = np.nan; se = np.nan
        method_used = 'none'; converged = False
        if _RPY2_OK:
            try:
                safe  = re.sub(r'[^A-Za-z0-9_]', '_', feat)
                sub2  = sub.rename(columns={feat: safe})
                formula = f'{safe} ~ Group_bin + age_mo_c + (1|pid)'
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
                mdf = smf.mixedlm(f'{feat} ~ Group_bin + age_mo_c', sub,
                                   groups=sub['pid']).fit(
                                   method=['lbfgs'], reml=True, maxiter=300)
                coef = float(mdf.params.get('Group_bin', np.nan))
                se   = float(mdf.bse.get('Group_bin', np.nan))
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

# ── Step 2: CR2 (bias-reduced linearization) ─────────────────────
def run_cr2(clip_df, feat_cols, subset_label='combined'):
    if not _WBT_OK: print("  [CR2] skipped"); return pd.DataFrame()
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    for feat in feat_cols:
        sub = df_use[['pid','Group_bin','age_mo_c',feat]].dropna(
                subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 10: continue
        X_cols = ['Group_bin','age_mo_c']
        X  = sub[X_cols].values.astype(float)
        y  = sub[feat].values.astype(float)
        clusters = sub['pid'].values
        try:
            wbt = WildboottestHC(X=X, y=y, cluster=clusters,
                                 R=np.eye(len(X_cols))[[0], :],
                                 B=999, bootstrap_type='WCR11')
            wbt.get_wildboottest()
            records.append({'feature': feat, 'subset': subset_label,
                            'method': 'CR2', 'p_raw': float(wbt.pvalue),
                            'n_clips': len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Step 3: GEE (robustness check) ───────────────────────────────
def run_gee(clip_df, feat_cols, subset_label='combined'):
    records = []
    df_use  = clip_df.copy()
    df_use['Group_bin'] = (df_use['Group'] == 'ASD').astype(float)
    df_use['age_mo_c']  = df_use['age_mo'] - df_use['age_mo'].mean()
    pid_map = {p: i for i, p in enumerate(df_use['pid'].unique())}
    df_use['pid_int'] = df_use['pid'].map(pid_map)
    for feat in feat_cols:
        sub = df_use[['pid_int','Group_bin','age_mo_c',feat]].dropna(
                subset=['pid_int','Group_bin',feat])
        if sub['Group_bin'].nunique() < 2 or len(sub) < 20: continue
        counts = sub.groupby('pid_int').size()
        sub = sub[sub['pid_int'].isin(counts[counts >= 2].index)]
        if len(sub) < 20: continue
        try:
            safe = re.sub(r'[^A-Za-z0-9_]', '_', feat)
            sub2 = sub.rename(columns={feat: safe})
            res  = GEE.from_formula(f'{safe} ~ Group_bin + age_mo_c',
                                    'pid_int', data=sub2,
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

# ── Step 4: Child-level permutation ──────────────────────────────
def run_child_permutation(child_df, feat_cols, n_perm=5000, subset_label='combined'):
    rng = np.random.default_rng(42); records = []
    for feat in feat_cols:
        sub = child_df[['pid','Group',feat]].dropna()
        if sub['Group'].nunique() < 2: continue
        av = sub[sub['Group']=='ASD'][feat].values
        nv = sub[sub['Group']=='Non-ASD'][feat].values
        if len(av) < 3 or len(nv) < 3: continue
        obs_stat = abs(np.mean(av) - np.mean(nv))
        n_asd    = len(av)
        vals_arr = sub[feat].values
        n_total  = len(sub)
        perm_stats = np.zeros(n_perm)
        for i in range(n_perm):
            sl  = rng.permutation(['ASD']*n_asd + ['Non-ASD']*(n_total-n_asd))
            a_v = vals_arr[np.array(sl)=='ASD']
            n_v = vals_arr[np.array(sl)=='Non-ASD']
            a_v = a_v[~np.isnan(a_v)]; n_v = n_v[~np.isnan(n_v)]
            perm_stats[i] = abs(np.mean(a_v)-np.mean(n_v)) if len(a_v)>0 and len(n_v)>0 else 0
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
def run_wild_bootstrap(child_df, feat_cols, n_boot=5000, subset_label='combined'):
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
        resid  = y - X@beta; t_obs = beta[1] / (np.std(resid)/np.sqrt(n) + 1e-10)
        X0     = X[:, [0, 2]]
        try: beta0, _, _, _ = np.linalg.lstsq(X0, y, rcond=None)
        except: continue
        resid0 = y - X0@beta0; pids = sub['pid'].values; u_pids = np.unique(pids)
        t_boot = np.zeros(n_boot)
        for b in range(n_boot):
            w_map = {p: rng.choice([-1.0, 1.0]) for p in u_pids}
            w     = np.array([w_map[p] for p in pids])
            y_b   = X0@beta0 + resid0*w
            try:
                beta_b, _, _, _ = np.linalg.lstsq(X, y_b, rcond=None)
                resid_b = y_b - X@beta_b
                t_boot[b] = beta_b[1] / (np.std(resid_b)/np.sqrt(n) + 1e-10)
            except: t_boot[b] = 0.0
        p_wb  = max(float(np.mean(np.abs(t_boot) >= abs(t_obs))), 1.0/n_boot)
        d     = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'WildBoot',
            'coef_ASD': float(beta[1]), 't_obs': float(t_obs), 'p_raw': p_wb,
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': int(len(av)), 'n_nasd': int(len(nv)),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Step 6: Pseudo-bulk MWU ──────────────────────────────────────
def run_mwu(child_df, feat_cols, subset_label='combined'):
    records = []
    for feat in feat_cols:
        av = child_df[child_df['Group']=='ASD'][feat].dropna().values
        nv = child_df[child_df['Group']=='Non-ASD'][feat].dropna().values
        if len(av) < 3 or len(nv) < 3: continue
        stat, p = stats.mannwhitneyu(av, nv, alternative='two-sided')
        d = cohen_d(av, nv); ci_lo, ci_hi = bootstrap_ci_d(av, nv, n_boot=500)
        records.append({
            'feature': feat, 'subset': subset_label, 'method': 'PseudobulkMW',
            'asd_median':  float(np.median(av)),
            'nasd_median': float(np.median(nv)),
            'mw_stat': float(stat), 'p_raw': float(p),
            'cohens_d': d, 'd_ci_lo': ci_lo, 'd_ci_hi': ci_hi,
            'n_asd': len(av), 'n_nasd': len(nv),
        })
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records), 'p_raw').sort_values('p_raw')

# ── Step 7: Consensus aggregation ────────────────────────────────
def make_consensus(results_dict, feat_cols, threshold=0.05):
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
        row['n_methods_sig'] = n_sig
        rows.append(row)
    cons = pd.DataFrame(rows)
    # Attach Cohen's d from LME (or permutation as fallback)
    lme_df = results_dict.get('LME_KR') or results_dict.get('LME_noKR')
    if lme_df is not None and len(lme_df) and 'cohens_d' in lme_df.columns:
        cons['cohens_d_LME'] = cons['feature'].map(
            lme_df.set_index('feature')['cohens_d'].to_dict())
        if 'd_ci_lo' in lme_df.columns:
            cons['d_ci_lo'] = cons['feature'].map(
                lme_df.set_index('feature')['d_ci_lo'].to_dict())
            cons['d_ci_hi'] = cons['feature'].map(
                lme_df.set_index('feature')['d_ci_hi'].to_dict())
    return cons.sort_values('n_methods_sig', ascending=False)

# ── Step 8: Consistency gate (across age bands) ───────────────────
def run_consistency_gate(feat_df, feat_cols, sig_feats):
    """
    For spinning (single behavior), consistency gate checks whether
    Group effects hold in the same direction across both STAT_BANDS.
    """
    band_mwu = {}
    for band in STAT_BANDS:
        sub    = feat_df[feat_df['age_band'] == band]
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
                         'p_raw': p, 'age_band': band})
        if recs: band_mwu[band] = pd.DataFrame(recs)

    cons_recs = []; consistent_feats = []
    band_all = pd.concat(band_mwu.values(), ignore_index=True) if band_mwu else pd.DataFrame()
    for feat in sig_feats:
        if len(band_all) == 0: break
        sub = band_all[band_all['feature'] == feat]
        if len(sub) < 2: continue
        signs  = np.sign(sub['cohens_d'].values)
        n_same = int((signs == signs[0]).sum())
        passed = (n_same == len(sub))
        cons_recs.append({
            'feature': feat, 'n_bands_tested': len(sub),
            'n_same_direction': n_same, 'consistent': passed,
        })
        if passed: consistent_feats.append(feat)
    return pd.DataFrame(cons_recs), consistent_feats, band_mwu

# ── Spearman age correlations ──────────────────────────────────────

# ── RUN FULL BATTERY ─────────────────────────────────────────────
print("\n--- Running full statistical battery ---")

lme_all  = run_lme_kr(feat_df,   PRIMARY_FEATS, 'combined')
cr2_all  = run_cr2(feat_df,      PRIMARY_FEATS, 'combined')
gee_all  = run_gee(feat_df,      PRIMARY_FEATS, 'combined')
perm_all = run_child_permutation(child_df, PRIMARY_FEATS, n_perm=5000,  subset_label='combined')
boot_all = run_wild_bootstrap(child_df,   PRIMARY_FEATS, n_boot=5000,   subset_label='combined')
mw_all   = run_mwu(child_df,     PRIMARY_FEATS, 'combined')

for name, res in [('LME_KR',lme_all),('CR2',cr2_all),('GEE',gee_all),
                  ('ChildPerm',perm_all),('WildBoot',boot_all),('MWU',mw_all)]:
    if len(res):
        print(f"  {name}: sig_raw={res['sig_raw05'].sum()}  FDR={res['sig_fdr05'].sum()}")
        res.to_csv(os.path.join(OUTPUT_DIR, f'stats_{name.lower()}_combined.csv'), index=False)

all_results = {
    'LME_KR': lme_all, 'CR2': cr2_all, 'GEE': gee_all,
    'ChildPerm': perm_all, 'WildBoot': boot_all, 'PseudobulkMW': mw_all,
}
consensus_all = make_consensus(all_results, PRIMARY_FEATS)
consensus_all.to_csv(os.path.join(OUTPUT_DIR, 'stats_consensus_all.csv'), index=False)

sig_feats = list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else []
cons_df, consistent_feats, band_mwu_dict = run_consistency_gate(feat_df, PRIMARY_FEATS, sig_feats)
if len(cons_df):
    cons_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_consistency_gate.csv'), index=False)
    print(f"\n  Consistency gate: {len(consistent_feats)}/{len(sig_feats)} passed (same direction across age bands)")
    for f in consistent_feats: print(f"    ✓ {f}")

# Age-band stratified
print("\n--- Age-band stratified analysis ---")
band_lme_results = {}
for band in STAT_BANDS:
    sub     = feat_df[feat_df['age_band'] == band]
    sub_c   = child_df[child_df['age_band'] == band]
    asd_n   = sub[sub['Group']=='ASD']['pid'].nunique()
    nasd_n  = sub[sub['Group']=='Non-ASD']['pid'].nunique()
    print(f"\n  [{band}] ASD={asd_n} Non-ASD={nasd_n}")
    if asd_n < 3 or nasd_n < 3: print("  → skip"); continue
    slme  = run_lme_kr(sub,   PRIMARY_FEATS, band)
    sperm = run_child_permutation(sub_c, PRIMARY_FEATS, n_perm=2000, subset_label=band)
    sboot = run_wild_bootstrap(sub_c,  PRIMARY_FEATS, n_boot=2000,  subset_label=band)
    smw   = run_mwu(sub_c, PRIMARY_FEATS, band)
    sd    = {k: v for k, v in {'LME_KR':slme,'ChildPerm':sperm,'WildBoot':sboot,'MWU':smw}.items() if len(v)>0}
    if not sd: continue
    scons = make_consensus(sd, PRIMARY_FEATS); scons['age_band'] = band
    scons.to_csv(os.path.join(OUTPUT_DIR, f'stats_{band.replace("-","_")}_consensus.csv'), index=False)
    band_lme_results[band] = slme
    top3 = scons[scons['n_methods_sig']>0].head(3)
    for _, r in top3.iterrows(): print(f"    {r['feature']:<42} n_sig={r['n_methods_sig']}")

# Spearman age
print("\n--- Spearman age correlations ---")
sp_df = run_spearman_age(feat_df, PRIMARY_FEATS)
if len(sp_df):
    sp_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_spearman_age.csv'), index=False)
    print(f"  Significant age-feature correlations: {sp_df['sig_p05'].sum()}")

# Within-group trajectories (MWU)
print("\n--- Within-group trajectories ---")
for grp in GROUPS:
    early = feat_df[(feat_df['Group']==grp)&(feat_df['age_band']=='19-31mo')]
    late  = feat_df[(feat_df['Group']==grp)&(feat_df['age_band']=='32-38mo')]
    print(f"  {grp}: 19-31mo n={early['pid'].nunique()}  32-38mo n={late['pid'].nunique()}")
    if len(early) >= 5 and len(late) >= 5:
        combined_wg = pd.concat([early, late])
        r_wg = run_mwu(
            combined_wg.groupby(['pid','Group'])[PRIMARY_FEATS].mean().reset_index()
            .assign(Group=lambda x: x['pid'].map(
                dict(zip(combined_wg['pid'], combined_wg['age_band'])))),
            PRIMARY_FEATS, subset_label=f'{grp}_traj'
        )
        if len(r_wg): r_wg.to_csv(
            os.path.join(OUTPUT_DIR, f'stats_within_{grp.replace("-","_")}_trajectory.csv'),
            index=False)

# ═══════════════════════════════════════════════════════════════════
# PART 3: BAYESIAN HIERARCHICAL LMM
# ═══════════════════════════════════════════════════════════════════
hr("PART 3: BAYESIAN HIERARCHICAL LMM")

bayes_main_results = {}

if not RUN_BAYESIAN or not _PYMC_OK:
    print("  Bayesian skipped")
else:
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
        tmp = df[['pid','Group','age_mo',feat]].dropna().copy()
        if len(tmp) < 8 or tmp['pid'].nunique() < 4: return None
        tmp['Group_bin'] = (tmp['Group'] == 'ASD').astype(float)
        tmp['age_c']     = tmp['age_mo'] - tmp['age_mo'].mean()
        y_z, ym, ys      = _standardise(tmp[feat])
        pids, pid_idx     = np.unique(tmp['pid'].values, return_inverse=True)
        return {
            'df': tmp, 'y_z': y_z.astype(float),
            'group_bin': tmp['Group_bin'].values.astype(float),
            'age_c':     tmp['age_c'].values.astype(float),
            'pid_idx': pid_idx, 'n_pids': len(pids),
            'y_mean': ym, 'y_std': ys, 'n_obs': len(tmp),
        }

    def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                        chains=BAYES_CHAINS, seed=42):
        with pm.Model():
            alpha     = pm.Normal('alpha', 0, 1)
            b_group   = pm.Normal('b_group', 0, prior_sd)
            b_age     = pm.Normal('b_age', 0, 0.5)
            sigma_pid = pm.HalfNormal('sigma_pid', 1)
            sigma     = pm.HalfNormal('sigma', 1)
            alpha_pid = pm.Normal('alpha_pid', 0, sigma_pid, shape=bd['n_pids'])
            mu        = (alpha + alpha_pid[bd['pid_idx']]
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

    def prior_predictive_check(bd, feat, prior_sd=0.5):
        with pm.Model():
            b_group = pm.Normal('b_group', 0, prior_sd)
            b_age   = pm.Normal('b_age', 0, 0.5)
            sigma   = pm.HalfNormal('sigma', 1)
            alpha   = pm.Normal('alpha', 0, 1)
            mu      = alpha + b_group * bd['group_bin'] + b_age * bd['age_c']
            pm.Normal('y_obs', mu=mu, sigma=sigma, observed=bd['y_z'])
            ppc = pm.sample_prior_predictive(samples=200, random_seed=42)
        prior_ys  = ppc.prior_predictive['y_obs'].values.flatten()
        obs_range = (bd['y_z'].min(), bd['y_z'].max())
        pri_range = (float(np.percentile(prior_ys, 1)),
                     float(np.percentile(prior_ys, 99)))
        return {
            'feature': feat, 'obs_min': obs_range[0], 'obs_max': obs_range[1],
            'prior_p1': pri_range[0], 'prior_p99': pri_range[1],
            'plausible': pri_range[0] <= obs_range[0] and pri_range[1] >= obs_range[1],
        }

    # Select top features by permutation p-value
    bayes_feats = (perm_all.sort_values('p_raw').head(15)['feature'].tolist()
                   if len(perm_all) else PRIMARY_FEATS[:10])
    print(f"\nRunning Bayesian models on {len(bayes_feats)} features...")
    ppc_records = []; sensitivity_records = []; bayes_records = []
    idata_store = {}

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

        # Prior sensitivity: fit at 3 prior widths
        bf_vals = {}
        for psd in PRIOR_SDS:
            try:
                idata, summ = _fit_bayes_main(bd, prior_sd=psd)
                summ['feature'] = feat; summ['prior_sd'] = psd
                sensitivity_records.append(summ)
                bf_vals[psd] = summ['bf10']
                if psd == 0.5:
                    idata_store[feat] = idata
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")

        # Main result (prior=0.5) + robustness flag
        if 0.5 in bf_vals:
            match = [r for r in sensitivity_records
                     if r['feature']==feat and r['prior_sd']==0.5]
            if match:
                rec = match[-1].copy()
                bfs = [bf_vals[p] for p in PRIOR_SDS
                       if p in bf_vals and not np.isnan(bf_vals[p])]
                rec['bf_robust'] = bool(len(bfs) >= 2 and
                                        all((b > 1) == (bfs[0] > 1) for b in bfs))
                bayes_records.append(rec)
                flag    = '✓' if rec.get('converged') else '⚠'
                bf_str  = ' | '.join([f"sd={p}:BF={bf_vals.get(p, np.nan):.2f}"
                                      for p in PRIOR_SDS])
                print(f"  {feat:<42} {bf_str} {flag}")

    if ppc_records:
        pd.DataFrame(ppc_records).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_ppc.csv'), index=False)
    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_sensitivity.csv'), index=False)
    if bayes_records:
        bayes_df = pd.DataFrame(bayes_records).sort_values('bf10', ascending=False)
        bayes_df.to_csv(os.path.join(OUTPUT_DIR, 'bayes_main_combined.csv'), index=False)
        bayes_main_results['combined'] = bayes_df
        print(f"\n  BF10>3  : {(bayes_df['bf10']>3).sum()}/{len(bayes_df)}")
        print(f"  BF10>10 : {(bayes_df['bf10']>10).sum()}/{len(bayes_df)}")
        print(f"  BF robust: {bayes_df['bf_robust'].sum()}/{len(bayes_df)}")
        bad_conv = bayes_df[~bayes_df['converged']]
        if len(bad_conv):
            print(f"  ⚠ {len(bad_conv)} convergence issues — increase BAYES_TUNE")
            for _, r in bad_conv.iterrows():
                print(f"    {r['feature']}  rhat={r.get('rhat',np.nan):.3f}")

    # Age-stratified Bayesian (STAT_BANDS only)
    print("\n--- Bayesian: age-band stratified ---")
    band_bayes_recs = []
    for band in STAT_BANDS:
        sub    = feat_df[feat_df['age_band'] == band].copy()
        asd_n  = sub[sub['Group']=='ASD']['pid'].nunique()
        nasd_n = sub[sub['Group']=='Non-ASD']['pid'].nunique()
        print(f"  {band}: ASD={asd_n} Non-ASD={nasd_n}", end='')
        if asd_n < 3 or nasd_n < 3: print(" → skip"); continue
        band_recs = []
        for feat in bayes_feats:
            bd = _build_bayes_df(sub, feat)
            if bd is None: continue
            try:
                _, summ = _fit_bayes_main(bd)
                summ['feature'] = feat; summ['age_band'] = band
                band_recs.append(summ)
            except: pass
        band_bayes_recs.extend(band_recs)
        sig = sum(1 for r in band_recs
                  if not np.isnan(r.get('bf10', np.nan)) and r['bf10'] > 3)
        print(f" → BF10>3: {sig}/{len(band_recs)}")
    if band_bayes_recs:
        pd.DataFrame(band_bayes_recs).to_csv(
            os.path.join(OUTPUT_DIR, 'bayes_age_stratified.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════
# PART 4: CLASSIFICATION — LOSO (LR + RF)
# ═══════════════════════════════════════════════════════════════════
hr("PART 4: CLASSIFICATION — CHILD-LEVEL LOSO")

def run_loso_child(cdf, feat_cols, clf_name='LR', n_perm=500, seed=42):
    df_ = cdf.copy(); df_['y'] = (df_['Group'] == 'ASD').astype(int)
    if df_['y'].sum() < 4 or (1-df_['y']).sum() < 4: return None
    usable = [f for f in feat_cols if f in df_.columns and df_[f].notna().mean() > 0.5]
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
            y_score.extend(pipe.predict_proba(test[usable].values)[:, 1].tolist())
            y_true.extend(test['y'].values.tolist())
        except: continue
    if len(set(y_true)) < 2: return None
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    rng  = np.random.default_rng(seed)
    perm = [roc_auc_score(rng.permuted(np.array(y_true)), y_score)
            for _ in range(n_perm)]
    p_perm = float((np.array(perm) >= auc).mean())
    cm = confusion_matrix(y_true, (np.array(y_score) >= 0.5).astype(int))
    print(f"  [{clf_name}] AUC={auc:.3f}  AP={ap:.3f}  p_perm={p_perm:.4f}  "
          f"n_feat={len(usable)}")
    return {'auc': auc, 'ap': ap, 'perm_p': p_perm,
            'n_features': len(usable), 'n_subjects': df_['pid'].nunique(),
            'y_true': y_true, 'y_score': y_score, 'perm_aucs': perm,
            'confusion_matrix': cm, 'clf': clf_name}

clf_results = {}
CHILD_FEATS = [f for f in PRIMARY_FEATS if f in child_df.columns]

# Combined child-level
print("\n--- Combined child level ---")
for cname in ['LR', 'RF']:
    r = run_loso_child(child_df, CHILD_FEATS, clf_name=cname)
    if r: clf_results[f'combined_{cname}'] = r

# Band-stratified child level
for band in STAT_BANDS:
    sub   = child_df[child_df['age_band'] == band]
    asd_n = (sub['Group']=='ASD').sum(); nasd_n = (sub['Group']=='Non-ASD').sum()
    print(f"\n--- {band} child (ASD={asd_n}, Non-ASD={nasd_n}) ---")
    if asd_n < 4 or nasd_n < 4: print("  Skipped"); continue
    for cname in ['LR', 'RF']:
        r = run_loso_child(sub, CHILD_FEATS, clf_name=cname)
        if r: clf_results[f'{band}_{cname}'] = r

if clf_results:
    pd.DataFrame([{'subset': k, 'clf': v.get('clf',''), 'auc': v['auc'],
                   'ap': v['ap'], 'perm_p': v['perm_p'],
                   'n_features': v['n_features'], 'n_subjects': v['n_subjects']}
                  for k, v in clf_results.items()]
                 ).to_csv(os.path.join(OUTPUT_DIR, 'classification_summary.csv'), index=False)

# RF feature importances
feat_importance_df = pd.DataFrame()
try:
    tmp = child_df.copy(); tmp['y'] = (tmp['Group'] == 'ASD').astype(int)
    usable = [f for f in CHILD_FEATS if tmp[f].notna().mean() > 0.5]
    tmp[usable] = tmp[usable].fillna(tmp[usable].median())
    sc = StandardScaler(); X = sc.fit_transform(tmp[usable].values)
    rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
    rf_full.fit(X, tmp['y'].values)
    feat_importance_df = pd.DataFrame({
        'feature': usable, 'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False)
    feat_importance_df['label'] = feat_importance_df['feature'].map(
        SPINNING_SHORT).fillna(feat_importance_df['feature'])
    feat_importance_df.to_csv(
        os.path.join(OUTPUT_DIR, 'rf_feature_importances.csv'), index=False)
    print("\nTop 10 RF importances:")
    for _, r in feat_importance_df.head(10).iterrows():
        print(f"  {r['feature']:<45} {r['importance']:.4f}")
except Exception as e:
    print(f"RF importance failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# PART 5: FIGURES
# ═══════════════════════════════════════════════════════════════════
hr("PART 5: FIGURES")

# ── Fig 1: Sample overview ────────────────────────────────────────
print("  Fig 1: Sample overview...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Spinning Sample Overview', fontweight='bold')
ax = axes[0]
gc = child_df['Group'].value_counts()
bars = ax.bar(GROUPS, [gc.get(g, 0) for g in GROUPS],
              color=[COLORS[g] for g in GROUPS], width=0.5, edgecolor='white')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            str(int(bar.get_height())), ha='center', fontweight='bold')
ax.set_title('(a) Children per group'); ax.set_ylabel('N')
ax = axes[1]
bc = feat_df.groupby(['age_band', 'Group']).size().unstack(fill_value=0)
x = np.arange(len(bc)); w = 0.35
for i, grp in enumerate(GROUPS):
    if grp in bc.columns:
        ax.bar(x+i*w, bc[grp], w, color=COLORS[grp], label=grp,
               alpha=0.85, edgecolor='white')
ax.set_xticks(x+w/2); ax.set_xticklabels(bc.index, fontsize=8)
ax.set_title('(b) Clips by age band'); ax.set_ylabel('N'); ax.legend(fontsize=8)
ax = axes[2]
for grp in GROUPS:
    ax.hist(child_df[child_df['Group']==grp]['age_mo'],
            bins=8, alpha=0.6, color=COLORS[grp], label=grp, edgecolor='white')
for band, (lo, hi) in AGE_BANDS.items():
    ax.axvspan(lo, hi, alpha=0.08, color=BAND_COLORS[band], label=band)
ax.set_title('(c) Age distribution'); ax.set_xlabel('Age (months)'); ax.legend(fontsize=7)
plt.tight_layout(); save_fig(fig, 'fig01_sample_overview.png')

# ── Fig 2: Effect size forest plot (all methods) ─────────────────
print("  Fig 2: Effect sizes / forest...")
if len(lme_all):
    ht = lme_all.copy()
    ht['label'] = ht['feature'].map(SPINNING_SHORT).fillna(ht['feature'])
    ht = ht.reindex(ht['cohens_d'].abs().sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(12, max(6, len(ht)*0.38)))
    bar_colors = [ASD_COLOR if d > 0 else NONASD_COLOR for d in ht['cohens_d']]
    ax.barh(ht['label'], ht['cohens_d'], color=bar_colors,
            edgecolor='white', height=0.6, alpha=0.85)
    ax.errorbar(ht['cohens_d'], range(len(ht)),
                xerr=[ht['cohens_d']-ht['d_ci_lo'], ht['d_ci_hi']-ht['cohens_d']],
                fmt='none', color='black', lw=1.2, capsize=3, alpha=0.7)
    ax.axvline(0, color='black', lw=0.8)
    for t, ls in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
        ax.axvline(t, color='gray', lw=0.6, ls=ls, alpha=0.4)
        ax.axvline(-t, color='gray', lw=0.6, ls=ls, alpha=0.4)
    for j, (_, row) in enumerate(ht.iterrows()):
        if row.get('sig_fdr05'):
            ax.text(row['d_ci_hi']+0.02, j, '★', va='center', fontsize=10, color='gold')
        elif row.get('sig_raw05'):
            ax.text(row['d_ci_hi']+0.02, j, '●', va='center', fontsize=8)
    ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)")
    ax.set_title("Effect Sizes — LME+KR (95% bootstrap CI)\n★=FDR  ●=raw p<0.05",
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
    plt.tight_layout(); save_fig(fig, 'fig02_effect_sizes_forest.png')

# ── Fig 3: Violin plots ───────────────────────────────────────────
print("  Fig 3: Violin plots...")
DISP = [(f,l) for f,l in [
    ('sw_amplitude',           'Shoulder Width Amplitude'),
    ('sw_cv',                  'Shoulder Width CV'),
    ('sw_dom_freq',            'Shoulder Width Dom Freq'),
    ('sw_band_power_0p5_2p5hz','Shoulder Width 0.5-2.5Hz Power'),
    ('ls_x_spectral_entropy',  'Left Shoulder X Entropy'),
    ('rs_y_spectral_entropy',  'Right Shoulder Y Entropy'),
    ('ls_y_band_power_0p5_2p5hz','Left Shoulder Y 0.5-2.5Hz'),
    ('lw_x_band_power_0p5_2p5hz','Left Wrist X 0.5-2.5Hz'),
    ('wrist_x_LR_corr',        'Wrist X L-R Correlation'),
    ('sh_x_LR_corr',           'Shoulder X L-R Correlation'),
    ('nose_x_amplitude',       'Nose X Amplitude'),
    ('spin_intensity_mean',    'Spin Intensity (mean)'),
] if f in feat_df.columns]

if DISP and len(lme_all):
    ncols = 4; nrows = int(np.ceil(len(DISP)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    fig.suptitle('Spinning Kinematics — ASD vs Non-ASD (clip level)', fontweight='bold')
    axes = axes.flatten()
    for i, (feat, label) in enumerate(DISP):
        ax = axes[i]
        dg = [feat_df.loc[feat_df['Group']==g, feat].dropna().values for g in GROUPS]
        if any(len(d) == 0 for d in dg): ax.set_visible(False); continue
        parts = ax.violinplot(dg, positions=[0,1], showmedians=True, showextrema=False)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for j, vals in enumerate(dg):
            ax.scatter(j+np.random.uniform(-0.07, 0.07, len(vals)), vals,
                       color=list(COLORS.values())[j], alpha=0.2, s=8, zorder=3)
        row = lme_all[lme_all['feature']==feat]
        if len(row):
            p_r = row['p_raw'].values[0]; p_f = row['p_fdr'].values[0]
            d   = row['cohens_d'].values[0]
            col = '#cc0000' if p_f < 0.05 else ('#ff8800' if p_r < 0.05 else 'gray')
            ax.text(0.5, 0.97, f'p={p_r:.3f}|FDR={p_f:.3f}|d={d:.2f}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=7.5, color=col)
            ymax = max(np.percentile(d2, 95) for d2 in dg if len(d2))
            yr   = ymax - min(np.percentile(d2, 5) for d2 in dg if len(d2))
            add_sig_bar(ax, 0, 1, ymax+yr*0.05, p_r, h=max(yr*0.04, 1e-6))
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_title(label, fontsize=9, fontweight='bold')
    for j in range(len(DISP), len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); save_fig(fig, 'fig03_violins_combined.png')

# ── Fig 4: Consensus heatmap ──────────────────────────────────────
print("  Fig 4: Consensus heatmap...")
if len(consensus_all) > 0:
    p_cols    = [c for c in consensus_all.columns if c.startswith('p_')]
    heat_data = consensus_all.set_index('feature')[p_cols].head(20)
    heat_log  = -np.log10(heat_data.clip(lower=1e-5, upper=1.0).astype(float))
    fig, ax   = plt.subplots(figsize=(len(p_cols)*2+2, max(6, len(heat_data)*0.45)))
    im = ax.imshow(heat_log.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=4)
    ax.set_xticks(range(len(p_cols)))
    ax.set_xticklabels([c.replace('p_','') for c in p_cols], rotation=30, ha='right')
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels([SPINNING_SHORT.get(f, f) for f in heat_data.index], fontsize=9)
    for i in range(heat_log.shape[0]):
        for j in range(heat_log.shape[1]):
            raw_p = heat_data.values[i, j]
            ax.text(j, i, f'{raw_p:.3f}{"*" if raw_p < 0.05 else ""}',
                    ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='-log10(p)')
    ax.set_title('Consensus p-values across methods (top 20 features)',
                 fontweight='bold')
    plt.tight_layout(); save_fig(fig, 'fig04_consensus_heatmap.png')

# ── Fig 5: Age-band effect heatmap (Cohen's d) ───────────────────
print("  Fig 5: Age-band Cohen's d heatmap...")
if band_mwu_dict and len(sig_feats):
    band_all = pd.concat(band_mwu_dict.values(), ignore_index=True)
    feats_plot = sig_feats[:15]
    sub_b = band_all[band_all['feature'].isin(feats_plot)]
    if len(sub_b):
        bands_avail = sorted(sub_b['age_band'].unique())
        pivot_d = sub_b.pivot_table(index='feature', columns='age_band', values='cohens_d')
        pivot_d.index = [SPINNING_SHORT.get(f, f) for f in pivot_d.index]
        fig, ax = plt.subplots(figsize=(max(6, len(bands_avail)*2.5),
                                        max(5, len(feats_plot)*0.6)))
        im = ax.imshow(pivot_d.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(bands_avail))); ax.set_xticklabels(bands_avail, fontsize=9)
        ax.set_yticks(range(len(pivot_d))); ax.set_yticklabels(pivot_d.index, fontsize=8)
        for i in range(pivot_d.shape[0]):
            for j in range(pivot_d.shape[1]):
                v = pivot_d.values[i, j]
                if not np.isnan(v): ax.text(j, i, f'{v:.2f}', ha='center',
                                             va='center', fontsize=7)
        plt.colorbar(im, ax=ax, label="Cohen's d", fraction=0.03)
        ax.set_title("Cohen's d by Age Band\nConsistent color = robust group effect",
                     fontweight='bold')
        plt.tight_layout(); save_fig(fig, 'fig05_age_band_cohens_d_heatmap.png')

# ── Fig 6: Consistency gate ───────────────────────────────────────
print("  Fig 6: Consistency gate...")
if len(cons_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(cons_df)*0.45)))
    cols_cg = [ASD_COLOR if v else NONASD_COLOR for v in cons_df['consistent']]
    ax.barh(cons_df['feature'].map(SPINNING_SHORT).fillna(cons_df['feature']),
            cons_df['n_same_direction'] / cons_df['n_bands_tested'],
            color=cols_cg, edgecolor='white', height=0.6)
    ax.axvline(1.0, color='green', lw=1.5, ls='--', label='All bands consistent')
    ax.axvline(0.5, color='orange', lw=1, ls=':', label='50%')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Fraction of age bands with same direction')
    ax.set_title('Consistency Gate (across age bands)\nRed=inconsistent — interpret carefully',
                 fontweight='bold')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig06_consistency_gate.png')

# ── Fig 7: Bayesian forest plot ───────────────────────────────────
print("  Fig 7: Bayesian forest...")
if 'combined' in bayes_main_results and len(bayes_main_results['combined']) > 0:
    bdf = bayes_main_results['combined'].copy()
    bdf['label'] = bdf['feature'].map(SPINNING_SHORT).fillna(bdf['feature'])
    bdf = bdf.sort_values('b_group_mean')
    fig, ax = plt.subplots(figsize=(13, max(5, len(bdf)*0.5)))
    for j, (_, row) in enumerate(bdf.iterrows()):
        col = ASD_COLOR if row['b_group_mean'] > 0 else NONASD_COLOR
        ax.plot([row['hdi94_lo'], row['hdi94_hi']], [j, j],
                color=col, lw=2.5, alpha=0.8)
        ax.scatter(row['b_group_mean'], j, color=col, s=70, zorder=5)
        ax.plot(row['hdi94_lo'], j, '|', color=col, markersize=8)
        ax.plot(row['hdi94_hi'], j, '|', color=col, markersize=8)
        bf  = float(row['bf10']) if not np.isnan(float(row['bf10'])) else 0
        lbl = f"BF={bf:.1f}"
        if not row.get('converged', True):     lbl += ' ⚠'
        if not row.get('bf_robust', True):     lbl += ' [prior-sensitive]'
        ax.text(row['hdi94_hi']+0.01, j, lbl, va='center', fontsize=7)
    ax.axvline(0, color='black', lw=1.2, ls='--')
    ax.set_yticks(range(len(bdf))); ax.set_yticklabels(bdf['label'], fontsize=9)
    ax.set_xlabel('Posterior mean  |  94% HDI  (standardised)')
    ax.set_title('Bayesian Hierarchical LMM — Spinning\n'
                 '⚠=convergence  [prior-sensitive]=BF changed across priors',
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR, label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR, label='Non-ASD higher')])
    plt.tight_layout(); save_fig(fig, 'fig07_bayes_forest.png')

# ── Fig 8: Prior sensitivity ──────────────────────────────────────
print("  Fig 8: Prior sensitivity...")
if os.path.isfile(os.path.join(OUTPUT_DIR, 'bayes_sensitivity.csv')):
    sens = pd.read_csv(os.path.join(OUTPUT_DIR, 'bayes_sensitivity.csv'))
    if len(sens):
        feats_s = sens['feature'].unique()[:12]
        fig, axes = plt.subplots(int(np.ceil(len(feats_s)/3)), 3,
                                  figsize=(15, 4*int(np.ceil(len(feats_s)/3))))
        fig.suptitle('Prior Sensitivity — BF10 across prior widths', fontweight='bold')
        axes = axes.flatten()
        for i, feat in enumerate(feats_s):
            ax  = axes[i]
            sub = sens[sens['feature']==feat].sort_values('prior_sd')
            ax.plot(sub['prior_sd'], sub['bf10'], marker='o', color=ASD_COLOR, lw=2)
            ax.axhline(3,  color='green', lw=1, ls='--', label='BF=3')
            ax.axhline(1,  color='gray',  lw=0.8, ls=':')
            ax.set_xlabel('Prior SD'); ax.set_ylabel('BF10')
            ax.set_title(SPINNING_SHORT.get(feat, feat)[:25], fontsize=9)
            ax.legend(fontsize=7)
        for j in range(len(feats_s), len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); save_fig(fig, 'fig08_prior_sensitivity.png')

# ── Fig 9: Developmental trajectories ────────────────────────────
print("  Fig 9: Trajectories...")
TRAJ_F = [f for f in ['sw_amplitude','sw_cv','ls_x_spectral_entropy',
                        'wrist_x_LR_corr','nose_x_amplitude',
                        'lw_x_band_power_0p5_2p5hz']
          if f in feat_df.columns]
if TRAJ_F:
    ncols = 3; nrows = int(np.ceil(len(TRAJ_F)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    fig.suptitle('Developmental Trajectories — Spinning', fontweight='bold')
    axes = axes.flatten()
    for i, feat in enumerate(TRAJ_F):
        ax = axes[i]
        for grp in GROUPS:
            sub = feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
            if len(sub) < 3: continue
            ax.scatter(sub['age_mo'], sub[feat], color=COLORS[grp], alpha=0.25, s=12)
            if len(sub) >= 5:
                m, b, r, p, _ = stats.linregress(sub['age_mo'], sub[feat])
                xr = np.linspace(sub['age_mo'].min(), sub['age_mo'].max(), 100)
                ax.plot(xr, m*xr+b, color=COLORS[grp], lw=2.5,
                        label=f'{grp} r={r:.2f} p={p:.3f}')
        for band, (lo, hi) in AGE_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.07, color=BAND_COLORS[band])
        ax.set_xlabel('Age (months)'); ax.set_ylabel(SPINNING_SHORT.get(feat, feat)[:20], fontsize=8)
        ax.set_title(SPINNING_SHORT.get(feat, feat), fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
    for j in range(len(TRAJ_F), len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); save_fig(fig, 'fig09_trajectories.png')

# ── Fig 10: Child-level boxplots (by age band) ───────────────────
print("  Fig 10: Child-level boxplots...")
BOX_F = [f for f in ['sw_amplitude','sw_cv','ls_x_spectral_entropy',
                       'wrist_x_LR_corr','nose_x_amplitude']
         if f in child_df.columns]
if BOX_F:
    fig, axes = plt.subplots(len(BOX_F), len(STAT_BANDS)+1,
                              figsize=(5*(len(STAT_BANDS)+1), 4*len(BOX_F)), sharey='row')
    fig.suptitle('Child-Level Boxplots — Each Dot = 1 Child', fontweight='bold')
    stream_labels = ['all'] + STAT_BANDS
    stream_dfs    = [child_df] + [child_df[child_df['age_band']==b] for b in STAT_BANDS]
    for ci, (slbl, sdf) in enumerate(zip(stream_labels, stream_dfs)):
        lme_ref = lme_all if slbl == 'all' else band_lme_results.get(slbl, pd.DataFrame())
        for ri, feat in enumerate(BOX_F):
            ax = axes[ri][ci]
            for j, grp in enumerate(GROUPS):
                vals = sdf[sdf['Group']==grp][feat].dropna().values
                if len(vals) == 0: continue
                bp = ax.boxplot(vals, positions=[j], widths=0.45, patch_artist=True,
                                showfliers=False,
                                medianprops={'color':'black','linewidth':2})
                bp['boxes'][0].set_facecolor(COLORS_LIGHT[grp])
                bp['boxes'][0].set_edgecolor(COLORS[grp])
                bp['boxes'][0].set_linewidth(1.5)
                ax.scatter(j+np.random.uniform(-0.12, 0.12, len(vals)), vals,
                           color=COLORS[grp], alpha=0.65, s=28, zorder=4)
            da = sdf[sdf['Group']=='ASD'][feat].dropna().values
            dn = sdf[sdf['Group']=='Non-ASD'][feat].dropna().values
            if len(da) >= 3 and len(dn) >= 3:
                _, p = stats.mannwhitneyu(da, dn, alternative='two-sided')
                ymax = sdf[feat].dropna().quantile(0.97)
                add_sig_bar(ax, 0, 1, ymax, p, h=abs(ymax)*0.04+1e-6)
            if ri == 0: ax.set_title(slbl, fontsize=9, fontweight='bold')
            if ci == 0: ax.set_ylabel(SPINNING_SHORT.get(feat, feat)[:18], fontsize=7)
            ax.set_xticks([0,1]); ax.set_xticklabels(['ASD','NASD'], fontsize=8)
            ax.text(0.5, -0.18, f'n={len(da)}/{len(dn)}',
                    transform=ax.transAxes, ha='center', fontsize=7, color='gray')
    plt.tight_layout(); save_fig(fig, 'fig10_child_boxplots.png')

# ── Fig 11: ICC bar chart ─────────────────────────────────────────
print("  Fig 11: ICC...")
if len(icc_df) > 0:
    top_icc = icc_df.head(20)
    fig, ax  = plt.subplots(figsize=(10, max(5, len(top_icc)*0.4)))
    colors_icc = ['#2ecc71' if v > 0.1 else '#e74c3c' for v in top_icc['ICC']]
    ax.barh(top_icc['feature'].map(SPINNING_SHORT).fillna(top_icc['feature']),
            top_icc['ICC'], color=colors_icc, edgecolor='white', height=0.65)
    ax.axvline(0.1, color='orange', lw=1.5, ls='--',
               label='ICC=0.10 (clustering threshold)')
    ax.set_xlabel('ICC')
    ax.set_title('Intraclass Correlation (within-child clustering)\n'
                 'Green=significant clustering — justifies LME',
                 fontweight='bold')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig11_icc.png')

# ── Fig 12: Classification ROC ────────────────────────────────────
print("  Fig 12: Classification ROC...")
if clf_results:
    keys  = list(clf_results.keys()); n = len(keys)
    ncols = min(n, 4); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    if nrows*ncols == 1: axes = np.array([[axes]])
    elif nrows == 1: axes = axes.reshape(1, -1)
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
            axins = ax.inset_axes([0.55, 0.05, 0.4, 0.28])
            axins.hist(r['perm_aucs'], bins=20, color='gray', alpha=0.7)
            axins.axvline(r['auc'], color=ASD_COLOR, lw=2)
            axins.set_title('Null', fontsize=6); axins.tick_params(labelsize=5)
    for i in range(len(keys), nrows*ncols):
        axes[i//ncols][i%ncols].set_visible(False)
    plt.tight_layout(); save_fig(fig, 'fig12_roc.png')

# ── Fig 13: RF feature importances ───────────────────────────────
print("  Fig 13: RF importances...")
if len(feat_importance_df) > 0:
    top20 = feat_importance_df.head(20)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top20['label'], top20['importance'],
            color=ASD_COLOR, edgecolor='white', height=0.65, alpha=0.85)
    ax.set_xlabel('Mean decrease in impurity')
    ax.set_title('RF Feature Importances (child level)', fontweight='bold')
    ax.axvline(top20['importance'].mean(), color='gray', lw=1, ls='--', label='Mean')
    ax.legend(); plt.tight_layout(); save_fig(fig, 'fig13_rf_importances.png')

# ── Fig 14: Method agreement scatter (MWU vs LME) ────────────────
print("  Fig 14: Method agreement...")
if len(lme_all) > 0 and len(mw_all) > 0:
    cmp = lme_all[['feature','p_raw','cohens_d','coef_ASD']].rename(
        columns={'p_raw':'p_lme','cohens_d':'d_lme'}).merge(
        mw_all[['feature','p_raw','cohens_d']].rename(
        columns={'p_raw':'p_mw','cohens_d':'d_mw'}), on='feature')
    if len(cmp) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Method Agreement — LME vs Mann-Whitney', fontweight='bold')
        ax = axes[0]
        ax.scatter(-np.log10(cmp['p_mw']+1e-10),
                   -np.log10(cmp['p_lme']+1e-10), color=ASD_COLOR, alpha=0.7, s=40)
        lim = max(-np.log10(cmp[['p_mw','p_lme']].min().min()+1e-10)*1.1, 3)
        ax.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.5)
        ax.axhline(-np.log10(0.05), color='gray', lw=0.8, ls='--')
        ax.axvline(-np.log10(0.05), color='gray', lw=0.8, ls='--')
        ax.set_xlabel('-log10(p) Mann-Whitney'); ax.set_ylabel('-log10(p) LME')
        ax.set_title('p-value agreement')
        ax = axes[1]
        ax.scatter(cmp['d_mw'], cmp['d_lme'], color=NONASD_COLOR, alpha=0.7, s=40)
        ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Cohen's d (MWU)"); ax.set_ylabel("Cohen's d (LME)")
        ax.set_title("Effect size agreement")
        for _, row in cmp.iterrows():
            if abs(row['d_mw']) > 0.4:
                ax.annotate(SPINNING_SHORT.get(row['feature'],
                            row['feature'].replace('_',' ')[:14]),
                            (row['d_mw'], row['d_lme']), fontsize=6, alpha=0.7)
        plt.tight_layout(); save_fig(fig, 'fig14_method_agreement.png')

# ── Fig 15: Age-band stratified bars ─────────────────────────────
print("  Fig 15: Age-band bars...")
KEY_F = [f for f in ['sw_amplitude','sw_cv','ls_x_spectral_entropy',
                       'lw_x_band_power_0p5_2p5hz','nose_x_amplitude']
         if f in feat_df.columns]
if KEY_F:
    band_list = STAT_BANDS
    fig, axes = plt.subplots(len(KEY_F), len(band_list),
                              figsize=(4.5*len(band_list), 4*len(KEY_F)), sharey='row')
    fig.suptitle('Key Features by Age Band — ASD vs Non-ASD', fontweight='bold')
    for ri, feat in enumerate(KEY_F):
        for ci, band in enumerate(band_list):
            ax  = axes[ri][ci]
            sub = feat_df[feat_df['age_band'] == band]
            da  = sub[sub['Group']=='ASD'][feat].dropna().values
            dn  = sub[sub['Group']=='Non-ASD'][feat].dropna().values
            means = [da.mean() if len(da) else 0, dn.mean() if len(dn) else 0]
            sems  = [stats.sem(da) if len(da)>1 else 0,
                     stats.sem(dn) if len(dn)>1 else 0]
            ax.bar([0,1], means, yerr=sems, color=[COLORS['ASD'],COLORS['Non-ASD']],
                   capsize=5, width=0.5, edgecolor='white', alpha=0.85)
            for j, (vals, xp) in enumerate([(da,0),(dn,1)]):
                if len(vals):
                    ax.scatter(xp+np.random.uniform(-0.1, 0.1, len(vals)), vals,
                               color=list(COLORS.values())[j], alpha=0.4, s=10)
            if len(da) >= 3 and len(dn) >= 3:
                _, p = stats.mannwhitneyu(da, dn, alternative='two-sided')
                ymax = max(means)+max(sems)+abs(max(means))*0.05
                add_sig_bar(ax, 0, 1, ymax, p, h=max(abs(ymax)*0.04, 0.001))
            if ri == 0: ax.set_title(band, fontsize=9)
            if ci == 0: ax.set_ylabel(SPINNING_SHORT.get(feat, feat)[:18], fontsize=7)
            ax.set_xticks([0,1]); ax.set_xticklabels(['ASD','NASD'], fontsize=8)
            ax.text(0.5, -0.22, f'n={len(da)}/{len(dn)}',
                    transform=ax.transAxes, ha='center', fontsize=7.5, color='gray')
    plt.tight_layout(); save_fig(fig, 'fig15_age_band_bars.png')

# ── Fig 16: Bayes Factor bar chart ───────────────────────────────
print("  Fig 16: Bayes factors...")
if 'combined' in bayes_main_results and len(bayes_main_results['combined']) > 0:
    bdf = bayes_main_results['combined'].copy()
    bdf['label'] = bdf['feature'].map(SPINNING_SHORT).fillna(bdf['feature'])
    bdf = bdf[bdf['bf10'].notna()].sort_values('bf10', ascending=True)
    if len(bdf):
        fig, ax = plt.subplots(figsize=(12, max(4, len(bdf)*0.45)))
        bar_c = ['#2ecc71' if v > 10 else '#f39c12' if v > 3
                 else '#e74c3c' if v < 0.33 else '#95a5a6'
                 for v in bdf['bf10']]
        ax.barh(bdf['label'], np.log10(bdf['bf10'].clip(lower=0.01)),
                color=bar_c, edgecolor='white', height=0.65)
        for thresh, ls_, lbl in [(np.log10(3),'--','BF=3 (moderate)'),
                                  (np.log10(10),'-','BF=10 (strong)')]:
            ax.axvline(thresh, color='gray', lw=1, ls=ls_, alpha=0.7, label=lbl)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('log₁₀(BF₁₀)  [positive = evidence for group effect]')
        ax.set_title('Bayes Factors — Spinning\n'
                     'Green=strong(>10), Orange=moderate(>3), Grey=anecdotal, Red=against',
                     fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        plt.tight_layout(); save_fig(fig, 'fig16_bayes_factors.png')

# ── Fig 17: Posterior traces ──────────────────────────────────────
print("  Fig 17: Posterior traces...")
if _PYMC_OK and idata_store:
    top_feats_b = list(bayes_main_results.get('combined', pd.DataFrame()).nlargest(
        min(6, len(bayes_main_results.get('combined', pd.DataFrame()))), 'bf10')['feature']) \
        if 'combined' in bayes_main_results else []
    items = [(f, idata_store[f]) for f in top_feats_b if f in idata_store][:6]
    if items:
        n = len(items)
        fig, axes = plt.subplots(n, 2, figsize=(13, 3.5*n))
        if n == 1: axes = [axes]
        fig.suptitle('Posterior Distributions + Traces — Top Features',
                     fontweight='bold')
        for i, (feat, idata) in enumerate(items):
            ax_d, ax_t = axes[i]
            label = SPINNING_SHORT.get(feat, feat.replace('_',' ')[:30])
            try:
                b_post = idata.posterior['b_group'].values.flatten()
                x_grid = np.linspace(b_post.min(), b_post.max(), 300)
                kde    = gaussian_kde(b_post)
                ax_d.plot(x_grid, kde(x_grid), color=ASD_COLOR, lw=2)
                ax_d.fill_between(x_grid, kde(x_grid), alpha=0.3, color=ASD_COLOR)
                ax_d.axvline(0, color='black', lw=1, ls='--')
                hdi  = az.hdi(idata, var_names=['b_group'], hdi_prob=0.94)['b_group'].values
                mask = (x_grid >= hdi[0]) & (x_grid <= hdi[1])
                ax_d.fill_between(x_grid[mask], kde(x_grid[mask]),
                                   alpha=0.5, color=NONASD_COLOR, label='94% HDI')
                ax_d.set_title(label, fontsize=9, fontweight='bold')
                ax_d.set_xlabel('b_group (std)'); ax_d.set_ylabel('Density')
                ax_d.legend(fontsize=7)
                for ch, trace in enumerate(idata.posterior['b_group'].values):
                    ax_t.plot(trace, alpha=0.6, lw=0.8,
                              color=plt.cm.tab10(ch/max(
                                  len(idata.posterior['b_group'].values), 1)))
                ax_t.set_xlabel('Draw'); ax_t.set_ylabel('b_group')
                ax_t.set_title('Trace (chains should mix)', fontsize=8)
            except Exception as e:
                ax_d.set_title(f'{label} — error: {e}', fontsize=8)
        plt.tight_layout(); save_fig(fig, 'fig17_posterior_traces.png')

# ── Fig 18: Three-way comparison (MWU vs LME vs Bayes) ───────────
print("  Fig 18: Three-way comparison...")
if (len(lme_all) > 0 and len(mw_all) > 0
        and 'combined' in bayes_main_results and len(bayes_main_results['combined']) > 0):
    mw  = mw_all[['feature','p_raw','cohens_d']]
    lme = lme_all[['feature','p_raw','cohens_d']].rename(
        columns={'p_raw':'p_lme','cohens_d':'d_lme'})
    bay = bayes_main_results['combined'][['feature','b_group_mean','bf10','p_pos']]
    cmp = mw.merge(lme, on='feature').merge(bay, on='feature')
    if len(cmp) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle('Three-Way Method Comparison — Spinning\n'
                     '✓ Age-balanced: MW / LME / Bayes should agree closely',
                     fontweight='bold')
        ax = axes[0]
        sc = ax.scatter(-np.log10(cmp['p_raw']+1e-10),
                         -np.log10(cmp['p_lme']+1e-10),
                         c=cmp['bf10'].clip(upper=20), cmap='RdYlGn',
                         s=50, alpha=0.8, edgecolors='gray', lw=0.5)
        lim = max(3, (-np.log10(cmp[['p_raw','p_lme']].min().min()+1e-10))*1.1)
        ax.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.4)
        ax.axhline(-np.log10(0.05), color='gray', lw=0.7, ls='--')
        ax.axvline(-np.log10(0.05), color='gray', lw=0.7, ls='--')
        ax.set_xlabel('-log10(p) Mann-Whitney'); ax.set_ylabel('-log10(p) LME')
        ax.set_title('MW vs LME\n(color=BF10)')
        plt.colorbar(sc, ax=ax, label='BF10', fraction=0.04)
        ax = axes[1]
        ax.scatter(cmp['d_lme'], cmp['b_group_mean'],
                   c=np.abs(cmp['cohens_d']), cmap='Blues',
                   s=50, alpha=0.8, edgecolors='gray', lw=0.5)
        ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Cohen's d (LME)"); ax.set_ylabel('Bayesian b_group (std)')
        ax.set_title("LME vs Bayes")
        ax = axes[2]
        ax.scatter(cmp['cohens_d'], np.log10(cmp['bf10'].clip(lower=0.01)),
                   c=[ASD_COLOR if g > 0 else NONASD_COLOR for g in cmp['b_group_mean']],
                   s=50, alpha=0.8)
        ax.axhline(np.log10(3),  color='orange', lw=1, ls='--', label='BF=3')
        ax.axhline(np.log10(10), color='green',  lw=1, ls='-',  label='BF=10')
        ax.axhline(0, color='black', lw=0.8); ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Cohen's d (MWU)"); ax.set_ylabel('log₁₀(BF₁₀)')
        ax.set_title("Cohen's d vs Bayes Factor"); ax.legend(fontsize=8)
        plt.tight_layout(); save_fig(fig, 'fig18_three_way_comparison.png')

# ── Fig 19: Feature heatmap (Group × Age) ────────────────────────
print("  Fig 19: Group × Age heatmap...")
HEAT_F = [f for f in ['sw_amplitude','sw_cv','sw_dom_freq',
                        'ls_x_spectral_entropy','rs_y_spectral_entropy',
                        'ls_y_band_power_0p5_2p5hz','lw_x_band_power_0p5_2p5hz',
                        'wrist_x_LR_corr','sh_x_LR_corr',
                        'nose_x_amplitude','spin_intensity_mean']
          if f in feat_df.columns]
if HEAT_F:
    cells, cell_labels = [], []
    for grp in GROUPS:
        for band in list(AGE_BANDS.keys()):
            sub = feat_df[(feat_df['Group']==grp)&(feat_df['age_band']==band)]
            cells.append(sub[HEAT_F].median() if len(sub) else
                         pd.Series([np.nan]*len(HEAT_F), index=HEAT_F))
            cell_labels.append(f'{grp}\n{band}')
    heat_raw = pd.DataFrame(cells, index=cell_labels, columns=HEAT_F)
    heat_z   = (heat_raw - heat_raw.mean()) / (heat_raw.std() + 1e-8)
    fig, ax  = plt.subplots(figsize=(len(HEAT_F)*1.1+1, len(cell_labels)*0.75+1))
    im = ax.imshow(heat_z.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(HEAT_F)))
    ax.set_xticklabels([SPINNING_SHORT.get(f, f).replace(' ','\n') for f in HEAT_F],
                        rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(cell_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label='Z-score', fraction=0.03)
    for i in range(len(cell_labels)):
        for j in range(len(HEAT_F)):
            v = heat_raw.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='black')
    ax.set_title('Feature × Group × Age Heatmap — Spinning', fontweight='bold')
    plt.tight_layout(); save_fig(fig, 'fig19_group_age_heatmap.png')

# ═══════════════════════════════════════════════════════════════════
# PART 6: SUMMARY
# ═══════════════════════════════════════════════════════════════════
hr("FINAL SUMMARY")
print(f"\nOutput : {OUTPUT_DIR}")
print(f"Figures: {FIGURE_DIR}\n")
print("--- CSVs ---")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith('.csv'):
        try:
            tmp = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
            print(f"  {fname:<65} {tmp.shape[0]:>5}r × {tmp.shape[1]:>3}c")
        except: print(f"  {fname}")
print("\n--- Figures ---")
for fname in sorted(os.listdir(FIGURE_DIR)):
    if fname.endswith('.png'):
        sz = os.path.getsize(os.path.join(FIGURE_DIR, fname)) / 1024
        print(f"  {fname:<55} {sz:.0f} KB")
print("\n--- KEY RESULTS ---")
for name, res in [('LME_KR',lme_all),('ChildPerm',perm_all),
                   ('WildBoot',boot_all),('MWU',mw_all)]:
    if len(res):
        print(f"  {name}: sig_raw={res['sig_raw05'].sum()}  FDR={res['sig_fdr05'].sum()}")
print(f"\nConsistency gate: {len(consistent_feats)}/{len(sig_feats)} features "
      f"passed (same direction across age bands)")
for f in consistent_feats: print(f"  ✓ {f}")
if clf_results:
    print("\nClassification (child-level LOSO):")
    for k, v in clf_results.items():
        print(f"  {k:<45} AUC={v['auc']:.3f}  p_perm={v['perm_p']:.4f}")
hr("SPINNING ANALYSIS v2 COMPLETE")
