#!/usr/bin/env python3
"""
Running kinematic analysis 
"""

import json
import os
import re
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import gaussian_kde
from scipy.stats import norm as spnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from sailsprep.analysis.common.banners import hr_v1 as hr
from sailsprep.analysis.common.parsing import extract_pid, extract_session
from sailsprep.analysis.common.keypoints import get_kp, assign_age_band
from sailsprep.analysis.common.mixed_models import _use_random_slope
from sailsprep.analysis.common.significance import add_sig_bar_v1 as add_sig_bar, fdr_annotate_v1 as fdr_annotate
from sailsprep.analysis.common.effect_size import cohen_d_v1 as cohen_d
from sailsprep.analysis.common.signal_processing import compute_angle_2d_v1 as compute_angle_2d, sparc_smoothness_v1 as sparc_smoothness
from sailsprep.analysis.common.icc import compute_icc
from sailsprep.analysis.common.consensus import make_consensus
from sailsprep.analysis.common.cross_validation import run_loso_child
from sailsprep.analysis.common.crawling_running_stats import (
    bootstrap_ci_d,
    run_consistency_gate_bands,
    run_child_permutation,
    run_lme_kr,
    run_cr2,
)

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Optional imports ────────────────────────────────────────────────

try:
    import arviz as az
    import pymc as pm
    _PYMC_OK = True
    print("[PyMC] available — Bayesian models enabled")
except Exception:
    _PYMC_OK = False
    print("[PyMC] NOT available — skipping Bayesian section")


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
MAIN_CSV   = "/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv"
OUTPUT_DIR = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/analysis/running/v3"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

FPS                    = 15.0
MIN_CONF               = 0.3
SEG_MIN_CONF           = 0.40
TORSO_CV_MAX           = 0.30
MIN_SEG_FRAMES         = 15
GAP_TOL                = 3
RUN_CADENCE_LO         = 1.5
RUN_CADENCE_HI         = 4.0

MIN_N_PER_GROUP        = 8
MIN_SESSIONS_FOR_SLOPE = 2
N_PERM                 = 5000
CONSENSUS_THRESHOLD    = 2

BAYES_DRAWS   = 2000
BAYES_TUNE    = 1000
BAYES_CHAINS  = 4
RUN_BAYESIAN  = True
PRIOR_SDS     = [0.3, 0.5, 1.0]

GROUPS    = ['ASD', 'Non-ASD']
AGE_BANDS = {
    '11-18mo': (11, 18),
    '19-31mo': (19, 31),
    '32-38mo': (32, 38),
}

ASD_COLOR    = '#E05C5C';  NONASD_COLOR = '#5B8DB8'
ASD_LIGHT    = '#F2AEAE';  NONASD_LIGHT = '#A8C8E8'
COLORS       = {'ASD': ASD_COLOR, 'Non-ASD': NONASD_COLOR}
COLORS_LIGHT = {'ASD': ASD_LIGHT, 'Non-ASD': NONASD_LIGHT}
BAND_COLORS  = {'11-18mo': '#7B5EA7', '19-31mo': '#4A9B6F', '32-38mo': '#D47C2A'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

KP = {
    'nose':        'kp_000',
    'L_shoulder':  'kp_005', 'R_shoulder': 'kp_006',
    'L_elbow':     'kp_007', 'R_elbow':    'kp_008',
    'L_wrist':     'kp_009', 'R_wrist':    'kp_010',
    'L_hip':       'kp_011', 'R_hip':      'kp_012',
    'L_knee':      'kp_013', 'R_knee':     'kp_014',
    'L_ankle':     'kp_015', 'R_ankle':    'kp_016',
    'L_big_toe':   'kp_017', 'L_small_toe':'kp_018', 'L_heel':'kp_019',
    'R_big_toe':   'kp_020', 'R_small_toe':'kp_021', 'R_heel':'kp_022',
}

RUNNING_FEAT_SHORT = {
    'hip_y_amplitude':'Hip Y Amp (Bounce)','hip_y_dom_freq':'Hip Y Freq (Cadence)',
    'hip_y_band_power':'Hip Y Band Power','hip_y_spectral_entropy':'Hip Y Entropy',
    'hip_y_sparc':'Hip Y SPARC','hip_y_jerk_mean':'Hip Y Jerk',
    'shoulder_y_amplitude':'Shoulder Bounce Amp','head_bob_amplitude':'Head Bob Amp',
    'knee_y_L_amplitude':'Knee L Lift Amp','knee_y_R_amplitude':'Knee R Lift Amp',
    'ankle_y_L_amplitude':'Ankle L Clearance','ankle_y_R_amplitude':'Ankle R Clearance',
    'vertical_oscillation_ratio':'Vert Osc Ratio',
    'hip_x_vel_mean':'Hip X Speed (mean)','hip_x_vel_std':'Hip X Speed (std)',
    'hip_x_vel_max':'Hip X Speed (max)',
    'ankle_x_L_range':'Stride Len Proxy L','ankle_x_R_range':'Stride Len Proxy R',
    'stride_len_cv':'Stride Len CV',
    'ankle_y_L_dom_freq':'Cadence L (Hz)','ankle_y_R_dom_freq':'Cadence R (Hz)',
    'stride_duration_mean':'Stride Duration','stride_duration_cv':'Stride Duration CV',
    'cadence':'Cadence (strides/s)','step_regularity':'Step Regularity',
    'ground_contact_ratio':'Ground Contact Ratio','flight_phase_ratio':'Flight Phase Ratio',
    'wrist_y_L_amplitude':'Arm Drive L Amp','wrist_y_R_amplitude':'Arm Drive R Amp',
    'wrist_x_L_amplitude':'Arm Crossover L','wrist_x_R_amplitude':'Arm Crossover R',
    'elbow_angle_L_mean':'Elbow L Angle','elbow_angle_R_mean':'Elbow R Angle',
    'elbow_angle_L_range':'Elbow L ROM','elbow_angle_R_range':'Elbow R ROM',
    'arm_drive_freq_L':'Arm Drive Freq L','arm_drive_freq_R':'Arm Drive Freq R',
    'arm_leg_phase_L':'Arm-Leg Phase L','arm_leg_phase_R':'Arm-Leg Phase R',
    'wrist_lr_y_corr':'Wrist LR Y Corr','arm_drive_asymmetry':'Arm Drive Asym',
    'knee_angle_L_mean':'Knee L Angle Mean','knee_angle_R_mean':'Knee R Angle Mean',
    'knee_angle_L_std':'Knee L Angle Std','knee_angle_R_std':'Knee R Angle Std',
    'knee_angle_L_range':'Knee L ROM','knee_angle_R_range':'Knee R ROM',
    'knee_angle_L_cv':'Knee L Angle CV','knee_angle_R_cv':'Knee R Angle CV',
    'knee_drive_height_L':'Knee Drive Height L','knee_drive_height_R':'Knee Drive Height R',
    'knee_angle_asym':'Knee Angle Asym',
    'hip_angle_L_mean':'Hip L Angle Mean','hip_angle_R_mean':'Hip R Angle Mean',
    'hip_angle_L_range':'Hip L ROM','hip_angle_R_range':'Hip R ROM',
    'hip_knee_coord_L':'Hip-Knee Coord L','hip_knee_coord_R':'Hip-Knee Coord R',
    'trunk_lean_mean':'Trunk Forward Lean','trunk_lean_std':'Trunk Lean Std',
    'trunk_lean_amplitude':'Trunk Lean Amp','lateral_sway_std':'Lateral Sway Std',
    'trunk_rotation_amplitude':'Trunk Rotation Amp',
    'counter_rotation_index':'Counter-Rotation Idx',
    'ankle_lr_y_asym':'Ankle LR Y Asym','knee_lr_y_corr':'Knee LR Y Corr',
    'ankle_lr_x_corr':'Ankle LR X Corr',
    'wrist_ankle_phase_L':'Wrist-Ankle Phase L','wrist_ankle_phase_R':'Wrist-Ankle Phase R',
    'ankle_y_L_sparc':'Ankle L SPARC','ankle_y_R_sparc':'Ankle R SPARC',
    'ankle_y_L_jerk_mean':'Ankle L Jerk','ankle_y_R_jerk_mean':'Ankle R Jerk',
    'knee_y_L_jerk_mean':'Knee L Jerk','knee_y_R_jerk_mean':'Knee R Jerk',
    'stride_knee_rom_mean':'Stride Knee ROM','stride_knee_rom_cv':'Stride Knee ROM CV',
    'foot_clearance_L_mean':'Foot Clearance L','foot_clearance_R_mean':'Foot Clearance R',
}

FEAT_LABEL = lambda f: RUNNING_FEAT_SHORT.get(f, f.replace('_', ' '))

# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")





def torso_length(fd):
    ls = get_kp(fd, KP['L_shoulder'], 0.1); rs = get_kp(fd, KP['R_shoulder'], 0.1)
    lh = get_kp(fd, KP['L_hip'],      0.1); rh = get_kp(fd, KP['R_hip'],      0.1)
    if not all([ls, rs, lh, rh]): return None
    sx=(ls['x']+rs['x'])/2; sy=(ls['y']+rs['y'])/2
    hx=(lh['x']+rh['x'])/2; hy=(lh['y']+rh['y'])/2
    d=np.sqrt((sx-hx)**2+(sy-hy)**2)
    return d if d>5 else None

def get_scale(fd):
    tl=torso_length(fd)
    if tl: return tl
    lh=get_kp(fd,KP['L_hip'],0.1); rh=get_kp(fd,KP['R_hip'],0.1)
    if lh and rh:
        d=np.sqrt((lh['x']-rh['x'])**2+(lh['y']-rh['y'])**2)
        if d>5: return d
    ls=get_kp(fd,KP['L_shoulder'],0.1); rs=get_kp(fd,KP['R_shoulder'],0.1)
    if ls and rs:
        d=np.sqrt((ls['x']-rs['x'])**2+(ls['y']-rs['y'])**2)
        if d>5: return d
    return None

def butter_lp(data, cutoff=6.0, fs=15.0, order=2):
    arr=np.array(data,dtype=float)
    if len(arr)<10: return arr
    nyq=0.5*fs
    b,a=butter(order,min(cutoff,nyq*0.9)/nyq,btype='low')
    if len(arr)<3*max(len(b),len(a)): return arr
    return filtfilt(b,a,arr)


def spectral_features(arr, fps, lo=RUN_CADENCE_LO, hi=RUN_CADENCE_HI):
    if len(arr)<16: return np.nan,np.nan,np.nan
    try:
        freqs,psd=welch(arr,fs=fps,nperseg=min(len(arr),64))
        dom_freq=float(freqs[np.argmax(psd)])
        psd_n=psd/(psd.sum()+1e-12)
        entropy=float(-np.sum(psd_n[psd_n>0]*np.log2(psd_n[psd_n>0])))
        band_pwr=float(psd[(freqs>=lo)&(freqs<=hi)].sum()/(psd.sum()+1e-12))
        return dom_freq,entropy,band_pwr
    except: return np.nan,np.nan,np.nan


def mean_jerk(pos, fps):
    if len(pos)<6: return np.nan
    try:
        sm=butter_lp(pos,fs=fps)
        jerk=np.diff(np.diff(np.diff(sm)*fps)*fps)*fps
        return float(np.mean(np.abs(jerk)))
    except: return np.nan

def detect_running_cycles(ankle_y, fps=15.0, min_distance=3):
    if len(ankle_y)<16: return []
    try: sm=butter_lp(ankle_y,cutoff=6.0,fs=fps)
    except: sm=ankle_y
    std_val=np.std(sm)
    if std_val<1e-8: return []
    peaks,_=find_peaks(-sm,distance=min_distance,prominence=std_val*0.2)
    if len(peaks)<2: return []
    return [(int(peaks[i]),int(peaks[i+1]))
            for i in range(len(peaks)-1)
            if 0.2<=(peaks[i+1]-peaks[i])/fps<=1.2]

def cross_correlation_peak(a, b, max_lag=10):
    a=np.asarray(a,float); b=np.asarray(b,float)
    n=min(len(a),len(b))
    if n<6: return np.nan,np.nan
    a,b=a[:n],b[:n]
    a_n=(a-a.mean())/(a.std()+1e-8); b_n=(b-b.mean())/(b.std()+1e-8)
    xcorr=np.correlate(a_n,b_n,mode='full')/n
    lags=np.arange(-(n-1),n)
    mask=np.abs(lags)<=max_lag
    sub_x,sub_l=xcorr[mask],lags[mask]
    idx=np.argmax(np.abs(sub_x))
    return float(sub_x[idx]),float(sub_l[idx])

def detect_flight_phases(ankle_y_L, ankle_y_R, fps=15.0):
    n=min(len(ankle_y_L),len(ankle_y_R))
    if n<10: return 0,n,n
    aL=np.array(ankle_y_L[:n]); aR=np.array(ankle_y_R[:n])
    window=max(3,int(fps*0.3))
    def rolling_min(arr,w):
        return np.array([arr[max(0,i-w):i+w+1].min() for i in range(len(arr))])
    baseline_L=rolling_min(aL,window); baseline_R=rolling_min(aR,window)
    threshold=0.05
    airborne_L=aL<(baseline_L-threshold); airborne_R=aR<(baseline_R-threshold)
    both_air=airborne_L&airborne_R
    flight_frames=int(both_air.sum()); contact_frames=n-flight_frames
    return flight_frames,contact_frames,n





# ═══════════════════════════════════════════════════════════════════
# PART 0: LOAD DATA
# ═══════════════════════════════════════════════════════════════════
hr("PART 0: LOAD DATA")

df_main=pd.read_csv(MAIN_CSV)
df_main['pid']    =df_main['video_path'].apply(extract_pid)
df_main['session']=df_main['video_path'].apply(extract_session)
df_main['age_mo'] =df_main['Age']*12
df_main=df_main[
    df_main['pid'].notna() &
    df_main['Group'].isin(GROUPS) &
    df_main['video_path'].str.contains('bids',case=False,na=False)
].copy()

sessions_per_child=(df_main.groupby('pid')['session'].nunique()
                    .rename('n_sessions').reset_index())

print(f"Valid rows: {len(df_main)}")
print(f"Unique children: {df_main['pid'].nunique()}")
print(f"  ASD:     {df_main[df_main['Group']=='ASD']['pid'].nunique()}")
print(f"  Non-ASD: {df_main[df_main['Group']=='Non-ASD']['pid'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# PART 1: FEATURE EXTRACTION  (identical running features as v1)
# ═══════════════════════════════════════════════════════════════════
hr("PART 1: FEATURE EXTRACTION")

def extract_running_features(pose_frames, frame_indices, fps=FPS):
    ankle_y_L,ankle_y_R=[],[]
    ankle_x_L,ankle_x_R=[],[]
    knee_y_L,knee_y_R=[],[]
    hip_x,hip_y=[],[]
    sh_x,sh_y=[],[]
    wrist_y_L,wrist_y_R=[],[]
    wrist_x_L,wrist_x_R=[],[]
    nose_y=[]
    elbow_angles_L,elbow_angles_R=[],[]
    knee_angles_L,knee_angles_R=[],[]
    hip_angles_L,hip_angles_R=[],[]
    trunk_lean_angles=[]; trunk_rotation=[]; torso_lens=[]; conf_vals=[]
    n_valid=0

    for fi in frame_indices:
        fk=str(fi)
        if fk not in pose_frames: continue
        fd=pose_frames[fk]; scale=get_scale(fd)
        if scale is None: continue
        la=get_kp(fd,KP['L_ankle']); ra=get_kp(fd,KP['R_ankle'])
        lh=get_kp(fd,KP['L_hip']);   rh=get_kp(fd,KP['R_hip'])
        if lh is None and rh is None and la is None and ra is None: continue
        n_valid+=1; torso_lens.append(scale)
        if la: ankle_y_L.append(la['y']/scale); ankle_x_L.append(la['x']/scale); conf_vals.append(la['confidence'])
        if ra: ankle_y_R.append(ra['y']/scale); ankle_x_R.append(ra['x']/scale); conf_vals.append(ra['confidence'])
        lk=get_kp(fd,KP['L_knee']); rk=get_kp(fd,KP['R_knee'])
        if lk: knee_y_L.append(lk['y']/scale)
        if rk: knee_y_R.append(rk['y']/scale)
        if lh and rh:
            hip_x.append((lh['x']+rh['x'])/2/scale); hip_y.append((lh['y']+rh['y'])/2/scale)
            conf_vals.append((lh['confidence']+rh['confidence'])/2)
        elif lh: hip_x.append(lh['x']/scale); hip_y.append(lh['y']/scale)
        elif rh: hip_x.append(rh['x']/scale); hip_y.append(rh['y']/scale)
        ls=get_kp(fd,KP['L_shoulder']); rs=get_kp(fd,KP['R_shoulder'])
        if ls and rs: sh_x.append((ls['x']+rs['x'])/2/scale); sh_y.append((ls['y']+rs['y'])/2/scale)
        lw=get_kp(fd,KP['L_wrist']); rw=get_kp(fd,KP['R_wrist'])
        if lw: wrist_y_L.append(lw['y']/scale); wrist_x_L.append(lw['x']/scale)
        if rw: wrist_y_R.append(rw['y']/scale); wrist_x_R.append(rw['x']/scale)
        ns=get_kp(fd,KP['nose'])
        if ns: nose_y.append(ns['y']/scale)
        for hkp,kkp,akp,store in [
            (KP['L_hip'],KP['L_knee'],KP['L_ankle'],knee_angles_L),
            (KP['R_hip'],KP['R_knee'],KP['R_ankle'],knee_angles_R),
        ]:
            h_=get_kp(fd,hkp); k_=get_kp(fd,kkp); a_=get_kp(fd,akp)
            if h_ and k_ and a_:
                ang=compute_angle_2d((h_['x'],h_['y']),(k_['x'],k_['y']),(a_['x'],a_['y']))
                if not np.isnan(ang): store.append(ang)
        for shkp,hkp,kkp,store in [
            (KP['L_shoulder'],KP['L_hip'],KP['L_knee'],hip_angles_L),
            (KP['R_shoulder'],KP['R_hip'],KP['R_knee'],hip_angles_R),
        ]:
            s_=get_kp(fd,shkp); h_=get_kp(fd,hkp); k_=get_kp(fd,kkp)
            if s_ and h_ and k_:
                ang=compute_angle_2d((s_['x'],s_['y']),(h_['x'],h_['y']),(k_['x'],k_['y']))
                if not np.isnan(ang): store.append(ang)
        for shkp,ekp,wkp,store in [
            (KP['L_shoulder'],KP['L_elbow'],KP['L_wrist'],elbow_angles_L),
            (KP['R_shoulder'],KP['R_elbow'],KP['R_wrist'],elbow_angles_R),
        ]:
            s_=get_kp(fd,shkp); e_=get_kp(fd,ekp); w_=get_kp(fd,wkp)
            if s_ and e_ and w_:
                ang=compute_angle_2d((s_['x'],s_['y']),(e_['x'],e_['y']),(w_['x'],w_['y']))
                if not np.isnan(ang): store.append(ang)
        if ls and rs and lh and rh:
            smx=(ls['x']+rs['x'])/2; smy=(ls['y']+rs['y'])/2
            hmx=(lh['x']+rh['x'])/2; hmy=(lh['y']+rh['y'])/2
            dx,dy=smx-hmx,smy-hmy
            trunk_lean_angles.append(float(np.degrees(np.arctan2(abs(dx),abs(dy)+1e-8))))
            trunk_rotation.append((smx-hmx)/scale)

    if n_valid<5: return None

    rec={
        'n_valid_frames':n_valid,'n_total_frames':len(frame_indices),
        'pct_valid':n_valid/len(frame_indices),'duration_sec':len(frame_indices)/fps,
        'mean_conf':float(np.mean(conf_vals)) if conf_vals else np.nan,
        'torso_cv':float(np.std(torso_lens)/np.mean(torso_lens)) if len(torso_lens)>3 else np.nan,
    }

    def traj_feats(arr, prefix, cutoff=6.0):
        if len(arr)<8: return
        a=np.array(arr)
        rec[f'{prefix}_amplitude']=float(np.ptp(a))
        rec[f'{prefix}_std']=float(np.std(a))
        rec[f'{prefix}_mean']=float(np.mean(a))
        try:
            sm=butter_lp(a,cutoff=cutoff,fs=fps); vel=np.diff(sm)*fps
            rec[f'{prefix}_vel_mean']=float(np.mean(np.abs(vel)))
            rec[f'{prefix}_vel_std']=float(np.std(vel))
            rec[f'{prefix}_vel_max']=float(np.max(np.abs(vel)))
            rec[f'{prefix}_sparc']=sparc_smoothness(vel,fps)
            rec[f'{prefix}_jerk_mean']=mean_jerk(a,fps)
        except: pass
        dom_f,ent,bp=spectral_features(a,fps)
        rec[f'{prefix}_dom_freq']=dom_f; rec[f'{prefix}_spectral_entropy']=ent; rec[f'{prefix}_band_power']=bp

    def angle_feats(arr, prefix):
        if len(arr)<5: return
        a=np.array(arr)
        rec[f'{prefix}_mean']=float(np.mean(a)); rec[f'{prefix}_std']=float(np.std(a))
        rec[f'{prefix}_range']=float(np.ptp(a)); rec[f'{prefix}_median']=float(np.median(a))
        rec[f'{prefix}_cv']=float(np.std(a)/(np.mean(a)+1e-8))

    # Bounce
    if len(hip_y)>=8:
        hy_arr=np.array(hip_y)
        rec['hip_y_amplitude']=float(np.ptp(hy_arr)); rec['hip_y_std']=float(np.std(hy_arr))
        try:
            sm=butter_lp(hy_arr,cutoff=6.0,fs=fps); vel=np.diff(sm)*fps
            rec['hip_y_vel_mean']=float(np.mean(np.abs(vel)))
            rec['hip_y_sparc']=sparc_smoothness(vel,fps)
            rec['hip_y_jerk_mean']=mean_jerk(hy_arr,fps)
        except: pass
        dom_f,ent,bp=spectral_features(hy_arr,fps)
        rec['hip_y_dom_freq']=dom_f; rec['hip_y_spectral_entropy']=ent; rec['hip_y_band_power']=bp
    if len(sh_y)>=8: rec['shoulder_y_amplitude']=float(np.ptp(sh_y)); rec['shoulder_y_std']=float(np.std(sh_y))
    if len(nose_y)>=8: rec['head_bob_amplitude']=float(np.ptp(nose_y)); rec['head_bob_std']=float(np.std(nose_y))
    if len(knee_y_L)>=8: rec['knee_y_L_amplitude']=float(np.ptp(knee_y_L)); rec['knee_drive_height_L']=float(np.ptp(knee_y_L))
    if len(knee_y_R)>=8: rec['knee_y_R_amplitude']=float(np.ptp(knee_y_R)); rec['knee_drive_height_R']=float(np.ptp(knee_y_R))
    traj_feats(ankle_y_L,'ankle_y_L'); traj_feats(ankle_y_R,'ankle_y_R')
    if 'hip_y_amplitude' in rec and ankle_x_L and ankle_x_R:
        stride_len=(np.ptp(ankle_x_L)+np.ptp(ankle_x_R))/2
        if stride_len>1e-6: rec['vertical_oscillation_ratio']=float(rec['hip_y_amplitude']/stride_len)

    # Speed
    if len(hip_x)>=8:
        hx_arr=np.array(hip_x)
        try:
            sm_hx=butter_lp(hx_arr,cutoff=6.0,fs=fps); vel_hx=np.diff(sm_hx)*fps
            rec['hip_x_vel_mean']=float(np.mean(np.abs(vel_hx)))
            rec['hip_x_vel_std']=float(np.std(vel_hx))
            rec['hip_x_vel_max']=float(np.max(np.abs(vel_hx)))
        except: pass
        rec['hip_x_amplitude']=float(np.ptp(hx_arr))
    if len(ankle_x_L)>=5: rec['ankle_x_L_range']=float(np.ptp(ankle_x_L))
    if len(ankle_x_R)>=5: rec['ankle_x_R_range']=float(np.ptp(ankle_x_R))
    if ankle_x_L and ankle_x_R:
        ml=min(len(ankle_x_L),len(ankle_x_R))
        sep=np.abs(np.array(ankle_x_L[:ml])-np.array(ankle_x_R[:ml]))
        rec['stride_len_cv']=float(np.std(sep)/(np.mean(sep)+1e-8))
    if len(hip_x)>=5: rec['lateral_sway_std']=float(np.std(hip_x))

    # Cadence
    for side,ay,kang_list in [('L',ankle_y_L,knee_angles_L),('R',ankle_y_R,knee_angles_R)]:
        if len(ay)<20: continue
        try:
            sm=butter_lp(np.array(ay),cutoff=6.0,fs=fps)
            cycles=detect_running_cycles(sm,fps=fps)
        except: continue
        if len(cycles)<2: continue
        durs=[(e-s)/fps for s,e in cycles]
        rec['stride_duration_mean']=float(np.mean(durs))
        rec['stride_duration_cv']=float(np.std(durs)/(np.mean(durs)+1e-8))
        rec['cadence']=float(len(cycles)/(len(ay)/fps))
        rec['step_regularity']=float(1.0-rec['stride_duration_cv'])
        stride_roms=[]
        for cs,ce in cycles:
            kslice=[kang_list[j] for j in range(len(kang_list)) if cs<=j<=ce]
            if len(kslice)>=3: stride_roms.append(float(np.ptp(kslice)))
        if stride_roms:
            rec['stride_knee_rom_mean']=float(np.mean(stride_roms))
            rec['stride_knee_rom_cv']=float(np.std(stride_roms)/(np.mean(stride_roms)+1e-8))
        clearances=[]
        for cs,ce in cycles:
            if ce>=len(ay): continue
            chunk=np.array(ay[cs:ce+1])
            if len(chunk)>=3: clearances.append(float(np.ptp(chunk)))
        if clearances:
            rec[f'foot_clearance_{side}_mean']=float(np.mean(clearances))
            rec[f'foot_clearance_{side}_cv']=float(np.std(clearances)/(np.mean(clearances)+1e-8))
        break
    if len(ankle_y_L)>=16:
        df_l,_,bp_l=spectral_features(ankle_y_L,fps)
        rec['ankle_y_L_dom_freq']=df_l; rec['ankle_y_L_band_power']=bp_l
    if len(ankle_y_R)>=16:
        df_r,_,bp_r=spectral_features(ankle_y_R,fps)
        rec['ankle_y_R_dom_freq']=df_r; rec['ankle_y_R_band_power']=bp_r
    if len(ankle_y_L)>=10 and len(ankle_y_R)>=10:
        flight_f,contact_f,total_f=detect_flight_phases(ankle_y_L,ankle_y_R,fps=fps)
        if total_f>0:
            rec['flight_phase_ratio']=float(flight_f/total_f)
            rec['ground_contact_ratio']=float(contact_f/total_f)

    # Arm drive
    traj_feats(wrist_y_L,'wrist_y_L'); traj_feats(wrist_y_R,'wrist_y_R')
    traj_feats(wrist_x_L,'wrist_x_L'); traj_feats(wrist_x_R,'wrist_x_R')
    if wrist_y_L and wrist_y_R:
        amp_l=float(np.ptp(wrist_y_L)); amp_r=float(np.ptp(wrist_y_R))
        rec['arm_drive_asymmetry']=float(abs(amp_l-amp_r)/(amp_l+amp_r+1e-8))
        ml=min(len(wrist_y_L),len(wrist_y_R))
        if ml>=8: rec['wrist_lr_y_corr']=float(np.corrcoef(wrist_y_L[:ml],wrist_y_R[:ml])[0,1])
    if len(wrist_y_L)>=16:
        adf_l,_,_=spectral_features(wrist_y_L,fps); rec['arm_drive_freq_L']=adf_l
    if len(wrist_y_R)>=16:
        adf_r,_,_=spectral_features(wrist_y_R,fps); rec['arm_drive_freq_R']=adf_r
    angle_feats(elbow_angles_L,'elbow_angle_L'); angle_feats(elbow_angles_R,'elbow_angle_R')
    if len(wrist_y_L)>=8 and len(ankle_y_L)>=8:
        peak_corr,lag=cross_correlation_peak(wrist_y_L,ankle_y_L)
        rec['arm_leg_phase_L']=lag; rec['wrist_ankle_phase_L']=peak_corr
    if len(wrist_y_R)>=8 and len(ankle_y_R)>=8:
        peak_corr,lag=cross_correlation_peak(wrist_y_R,ankle_y_R)
        rec['arm_leg_phase_R']=lag; rec['wrist_ankle_phase_R']=peak_corr

    # Leg mechanics
    angle_feats(knee_angles_L,'knee_angle_L'); angle_feats(knee_angles_R,'knee_angle_R')
    angle_feats(hip_angles_L,'hip_angle_L');   angle_feats(hip_angles_R,'hip_angle_R')
    if 'knee_angle_L_mean' in rec and 'knee_angle_R_mean' in rec:
        rec['knee_angle_asym']=float(abs(rec['knee_angle_L_mean']-rec['knee_angle_R_mean']))
    if len(knee_y_L)>=6: rec['knee_y_L_jerk_mean']=mean_jerk(knee_y_L,fps)
    if len(knee_y_R)>=6: rec['knee_y_R_jerk_mean']=mean_jerk(knee_y_R,fps)
    if len(hip_y)>=8 and len(knee_y_L)>=8:
        ml=min(len(hip_y),len(knee_y_L))
        peak_corr,_=cross_correlation_peak(hip_y[:ml],knee_y_L[:ml])
        rec['hip_knee_coord_L']=peak_corr
    if len(hip_y)>=8 and len(knee_y_R)>=8:
        ml=min(len(hip_y),len(knee_y_R))
        peak_corr,_=cross_correlation_peak(hip_y[:ml],knee_y_R[:ml])
        rec['hip_knee_coord_R']=peak_corr
    if len(knee_y_L)>=8 and len(knee_y_R)>=8:
        ml=min(len(knee_y_L),len(knee_y_R))
        rec['knee_lr_y_corr']=float(np.corrcoef(knee_y_L[:ml],knee_y_R[:ml])[0,1])

    # Trunk
    if trunk_lean_angles:
        angle_feats(trunk_lean_angles,'trunk_lean')
        rec['trunk_lean_amplitude']=float(np.ptp(trunk_lean_angles))
    if trunk_rotation and len(trunk_rotation)>=5:
        tr=np.array(trunk_rotation)
        rec['trunk_rotation_amplitude']=float(np.ptp(tr))
        rec['trunk_rotation_std']=float(np.std(tr))
        if len(hip_x)>=5:
            ml=min(len(tr),len(hip_x))
            r_val=np.corrcoef(tr[:ml],hip_x[:ml])[0,1]
            rec['counter_rotation_index']=float(-r_val)
    if sh_x and hip_x:
        ml=min(len(sh_x),len(hip_x)); tilt=np.array(sh_x[:ml])-np.array(hip_x[:ml])
        rec['trunk_tilt_amplitude']=float(np.ptp(tilt)); rec['trunk_tilt_std']=float(np.std(tilt))

    # Asymmetry
    if ankle_y_L and ankle_y_R:
        amp_l=float(np.ptp(ankle_y_L)); amp_r=float(np.ptp(ankle_y_R))
        rec['ankle_lr_y_asym']=float(abs(amp_l-amp_r)/(amp_l+amp_r+1e-8))
    if ankle_x_L and ankle_x_R:
        ml=min(len(ankle_x_L),len(ankle_x_R))
        if ml>=5: rec['ankle_lr_x_corr']=float(np.corrcoef(ankle_x_L[:ml],ankle_x_R[:ml])[0,1])

    return rec

# ── Extraction loop ──────────────────────────────────────────────
all_features=[]
n_ok=n_fail_label=n_fail_hrnet=n_fail_run=n_fail_kp=n_skip_conf=n_skip_cam=0

for proc_idx,(_,row) in enumerate(df_main.iterrows()):
    vpath=row['video_path']; lpath=row.get('label_path'); hpath=row.get('hrnet_full_path')
    pid=row['pid']; group=row['Group']; age_mo=row['age_mo']; ses=row.get('session')
    if not isinstance(lpath,str) or not os.path.isfile(lpath): n_fail_label+=1; continue
    if not isinstance(hpath,str) or not os.path.isfile(hpath): n_fail_hrnet+=1; continue
    try: anno=pd.read_csv(lpath)
    except: n_fail_label+=1; continue
    if 'Locomotion' not in anno.columns: n_fail_label+=1; continue
    anno['Locomotion']=anno['Locomotion'].astype(str)
    run_mask=(anno['Locomotion'].str.lower().str.contains('running',na=False)
              & ~(anno['Locomotion'].str.lower()=='nan'))
    if run_mask.sum()<MIN_SEG_FRAMES: n_fail_run+=1; continue
    try:
        with open(hpath,'r') as f: pose_data=json.load(f)
        frames=pose_data.get('frames',{}); ann_fps=float(pose_data.get('ann_fps',FPS))
    except: n_fail_hrnet+=1; continue
    if proc_idx%100==0: print(f"  [{proc_idx}] features so far: {len(all_features)}")
    run_idx=anno.index[run_mask].tolist()
    segments=[]; seg_s=run_idx[0]
    for i in range(1,len(run_idx)):
        if run_idx[i]-run_idx[i-1]>GAP_TOL:
            if run_idx[i-1]-seg_s>=MIN_SEG_FRAMES: segments.append((seg_s,run_idx[i-1]))
            seg_s=run_idx[i]
    if run_idx[-1]-seg_s>=MIN_SEG_FRAMES: segments.append((seg_s,run_idx[-1]))
    for seg_s,seg_e in segments:
        fidx=list(range(seg_s,seg_e+1))
        conf_list=[frames[str(fi)].get(KP.get(kn,''),{}).get('confidence',0)
                   for fi in fidx if str(fi) in frames
                   for kn in ['L_ankle','R_ankle','L_knee','R_knee','L_hip','R_hip']
                   if isinstance(frames[str(fi)].get(KP.get(kn,''),{}),dict)]
        if (np.mean(conf_list) if conf_list else 0)<SEG_MIN_CONF: n_skip_conf+=1; continue
        tl_vals=[torso_length(frames[str(fi)]) for fi in fidx if str(fi) in frames]
        tl_vals=[t for t in tl_vals if t is not None]
        if len(tl_vals)<5: n_skip_cam+=1; continue
        if float(np.std(tl_vals)/np.mean(tl_vals))>TORSO_CV_MAX: n_skip_cam+=1; continue
        feats=extract_running_features(frames,fidx,fps=ann_fps)
        if feats is None: n_fail_kp+=1; continue
        n_ok+=1
        feats.update({'pid':pid,'Group':group,'age_mo':age_mo,'session':ses,
                      'age_band':assign_age_band(age_mo),'video_path':vpath,
                      'seg_start':seg_s,'seg_end':seg_e})
        all_features.append(feats)

print(f"\nExtraction: OK={n_ok}  NoLabel={n_fail_label}  NoHRNet={n_fail_hrnet}  "
      f"NoRun={n_fail_run}  LowConf={n_skip_conf}  BadCam={n_skip_cam}  NoKP={n_fail_kp}")
if n_ok==0:
    print("ERROR: No running segments extracted."); import sys; sys.exit(1)

feat_df=pd.DataFrame(all_features)
feat_df.to_csv(os.path.join(OUTPUT_DIR,'clip_level_features.csv'),index=False)

META_COLS={'pid','Group','age_mo','session','age_band','video_path',
           'seg_start','seg_end','n_valid_frames','n_total_frames',
           'pct_valid','duration_sec','mean_conf','torso_cv'}
ALL_FEAT_COLS=[c for c in feat_df.columns if c not in META_COLS]

PRIMARY_FEATS=[f for f in ALL_FEAT_COLS if any(
    tok in f for tok in [
        'ankle','hip','knee','wrist','trunk','lateral','arm_drive','arm_leg',
        'head_bob','stride','cadence','step_','foot_clearance','elbow',
        'trunk_lean','flight','ground_contact','vertical_osc',
        'sparc','jerk','dom_freq','band_power','spectral',
        'vel_mean','asym','corr','sway','counter_rotation',
        'shoulder_y','knee_drive','hip_knee_coord','stride_len',
    ]) and f in feat_df.columns]

child_grp      =feat_df.groupby(['pid','Group'])
child_feats_df =child_grp[ALL_FEAT_COLS].mean().reset_index()
child_meta     =(feat_df.groupby(['pid','Group'])
                 .agg(age_mo=('age_mo','first'),age_band=('age_band','first'),
                      n_clips=('pid','count'),n_sessions=('session','nunique'))
                 .reset_index())
child_df=child_feats_df.merge(child_meta,on=['pid','Group'])
child_df=child_df.merge(sessions_per_child,on='pid',how='left')
child_df.to_csv(os.path.join(OUTPUT_DIR,'child_level_features.csv'),index=False)
CHILD_FEATS=[f for f in PRIMARY_FEATS if f in child_df.columns]

print(f"\nPrimary features: {len(PRIMARY_FEATS)}  |  Children: {len(child_df)}")
print(f"  ASD={(child_df['Group']=='ASD').sum()}  Non-ASD={(child_df['Group']=='Non-ASD').sum()}")
print(feat_df.groupby(['Group','age_band']).size().reset_index(name='n_clips').to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# PART 2: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
hr("PART 2: STATISTICAL ANALYSIS")


# ── Step 0: ICC ─────────────────────────────────────────────────

print("\n--- Step 0: ICC ---")
icc_df=compute_icc(feat_df,PRIMARY_FEATS)
icc_df.to_csv(os.path.join(OUTPUT_DIR,'stats_icc.csv'),index=False)
print(icc_df.head(10).to_string(index=False))
print(f"\n  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)}")

# ── Step 1: LME with KR + random slope + interaction ────────────

print("\n--- Step 1: LME (KR + random slope + interaction) ---")
lme_all=run_lme_kr(feat_df,PRIMARY_FEATS,'ALL',covariates='age_mo_c',interaction=True)
if len(lme_all):
    lme_all.to_csv(os.path.join(OUTPUT_DIR,'stats_lme_all.csv'),index=False)
    print(f"  Sig raw={lme_all['sig_raw05'].sum()}  FDR={lme_all['sig_fdr05'].sum()}")
    print(f"  Method: {lme_all['method'].mode()[0]}  Random slope: {lme_all['random_slope_used'].any()}")

# ── Step 2: CR2 ─────────────────────────────────────────────────

print("\n--- Step 2: CR2 ---")
cr2_all=run_cr2(feat_df,PRIMARY_FEATS,'ALL')
if len(cr2_all):
    cr2_all.to_csv(os.path.join(OUTPUT_DIR,'stats_cr2_all.csv'),index=False)
    print(f"  Sig raw={cr2_all['sig_raw05'].sum()}  FDR={cr2_all['sig_fdr05'].sum()}")

# ── Step 3: GEE ─────────────────────────────────────────────────
def run_gee(clip_df, feat_cols, subset_label='ALL'):
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    pid_map={p:i for i,p in enumerate(df_use['pid'].unique())}
    df_use['pid_int']=df_use['pid'].map(pid_map)
    for feat in feat_cols:
        safe=re.sub(r'[^A-Za-z0-9_]','_',feat)
        sub=df_use[['pid_int',feat,'Group_bin','age_mo_c']].dropna().copy()
        sub=sub.rename(columns={feat:safe})
        if sub['Group_bin'].nunique()<2: continue
        counts=sub.groupby('pid_int').size()
        sub=sub[sub['pid_int'].isin(counts[counts>=2].index)]
        if len(sub)<20: continue
        try:
            res=GEE.from_formula(f'{safe} ~ Group_bin + age_mo_c','pid_int',
                                  data=sub,family=Gaussian(),
                                  cov_struct=Exchangeable()).fit(maxiter=100)
            av=sub[sub['Group_bin']==1][safe].values
            nv=sub[sub['Group_bin']==0][safe].values
            records.append({'feature':feat,'subset':subset_label,'method':'GEE',
                            'coef_ASD':float(res.params.get('Group_bin',np.nan)),
                            'p_raw':float(res.pvalues.get('Group_bin',np.nan)),
                            'cohens_d':cohen_d(av,nv),'n_clips':len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

print("\n--- Step 3: GEE ---")
gee_all=run_gee(feat_df,PRIMARY_FEATS,'ALL')
if len(gee_all):
    gee_all.to_csv(os.path.join(OUTPUT_DIR,'stats_gee_all.csv'),index=False)
    print(f"  Sig raw={gee_all['sig_raw05'].sum()}  FDR={gee_all['sig_fdr05'].sum()}")

# ── Step 4: Child-level permutation ─────────────────────────────

print(f"\n--- Step 4: Child-level permutation ({N_PERM} perms) ---")
perm_all=run_child_permutation(child_df,CHILD_FEATS,n_perm=N_PERM,subset_label='ALL')
if len(perm_all):
    perm_all.to_csv(os.path.join(OUTPUT_DIR,'stats_permutation_all.csv'),index=False)
    print(f"  Sig raw={perm_all['sig_raw05'].sum()}  FDR={perm_all['sig_fdr05'].sum()}")

# ── Step 5: Wild cluster bootstrap ──────────────────────────────
def run_wild_bootstrap(child_df, feat_cols, n_boot=N_PERM, subset_label='ALL'):
    rng=np.random.default_rng(99); records=[]
    df_use=child_df.copy().dropna(subset=['age_mo'])
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    for feat in feat_cols:
        sub=df_use[['pid','Group_bin','age_mo',feat]].dropna()
        if sub['Group_bin'].nunique()<2: continue
        av=sub[sub['Group_bin']==1][feat].values
        nv=sub[sub['Group_bin']==0][feat].values
        if len(av)<3 or len(nv)<3: continue
        n=len(sub); y=sub[feat].values.astype(float)
        X=np.column_stack([np.ones(n),sub['Group_bin'].values,sub['age_mo'].values])
        try: beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        except: continue
        resid=y-X@beta; t_obs=beta[1]/(np.std(resid)/np.sqrt(n)+1e-10)
        X0=X[:,[0,2]]
        try: beta0,_,_,_=np.linalg.lstsq(X0,y,rcond=None)
        except: continue
        resid0=y-X0@beta0; pids=sub['pid'].values; u_pids=np.unique(pids)
        t_boot=np.zeros(n_boot)
        for b in range(n_boot):
            w_map={p:rng.choice([-1.0,1.0]) for p in u_pids}
            w=np.array([w_map[p] for p in pids])
            y_b=X0@beta0+resid0*w
            try:
                beta_b,_,_,_=np.linalg.lstsq(X,y_b,rcond=None)
                resid_b=y_b-X@beta_b
                t_boot[b]=beta_b[1]/(np.std(resid_b)/np.sqrt(n)+1e-10)
            except: t_boot[b]=0.0
        p_wb=max(float(np.mean(np.abs(t_boot)>=abs(t_obs))),1.0/n_boot)
        d=cohen_d(av,nv); ci_lo,ci_hi=bootstrap_ci_d(av,nv,n_boot=500)
        records.append({'feature':feat,'subset':subset_label,'method':'WildBoot',
                        'coef_ASD':float(beta[1]),'t_obs':float(t_obs),'p_raw':p_wb,
                        'cohens_d':d,'d_ci_lo':ci_lo,'d_ci_hi':ci_hi,
                        'n_asd':int(len(av)),'n_nasd':int(len(nv))})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

print(f"\n--- Step 5: Wild cluster bootstrap ({N_PERM} iters) ---")
boot_all=run_wild_bootstrap(child_df,CHILD_FEATS,n_boot=N_PERM,subset_label='ALL')
if len(boot_all):
    boot_all.to_csv(os.path.join(OUTPUT_DIR,'stats_wildboot_all.csv'),index=False)
    print(f"  Sig raw={boot_all['sig_raw05'].sum()}  FDR={boot_all['sig_fdr05'].sum()}")

# ── Step 6: Pseudo-bulk MWU ─────────────────────────────────────
def run_pseudobulk_mw(child_df, feat_cols, subset_label='ALL'):
    records=[]
    for feat in feat_cols:
        av=child_df[child_df['Group']=='ASD'][feat].dropna().values
        nv=child_df[child_df['Group']=='Non-ASD'][feat].dropna().values
        if len(av)<3 or len(nv)<3: continue
        stat,p=stats.mannwhitneyu(av,nv,alternative='two-sided')
        d=cohen_d(av,nv); ci_lo,ci_hi=bootstrap_ci_d(av,nv,n_boot=500)
        records.append({'feature':feat,'subset':subset_label,'method':'PseudobulkMW',
                        'asd_median':float(np.median(av)),'nasd_median':float(np.median(nv)),
                        'mw_stat':float(stat),'p_raw':float(p),
                        'cohens_d':d,'d_ci_lo':ci_lo,'d_ci_hi':ci_hi,
                        'n_asd':len(av),'n_nasd':len(nv)})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

print("\n--- Step 6: Pseudo-bulk MWU ---")
mw_all=run_pseudobulk_mw(child_df,CHILD_FEATS,'ALL')
if len(mw_all):
    mw_all.to_csv(os.path.join(OUTPUT_DIR,'stats_pseudobulk_mw_all.csv'),index=False)
    print(f"  Sig raw={mw_all['sig_raw05'].sum()}  FDR={mw_all['sig_fdr05'].sum()}")

# ── Step 7: Consensus ────────────────────────────────────────────

print("\n--- Step 7: Consensus ---")
all_results_dict={
    'LME_KR':       lme_all  if len(lme_all)  else None,
    'CR2':          cr2_all  if len(cr2_all)   else None,
    'GEE':          gee_all  if len(gee_all)   else None,
    'ChildPerm':    perm_all if len(perm_all)  else None,
    'WildBoot':     boot_all if len(boot_all)  else None,
    'PseudobulkMW': mw_all   if len(mw_all)    else None,
}
consensus_all=make_consensus(all_results_dict,CHILD_FEATS)
consensus_all.to_csv(os.path.join(OUTPUT_DIR,'stats_consensus_all.csv'),index=False)
robust_all=consensus_all[consensus_all['n_methods_sig']>=CONSENSUS_THRESHOLD]
print(f"\n  Robust features (≥{CONSENSUS_THRESHOLD} methods): {len(robust_all)}")
if len(robust_all):
    p_cols=[c for c in robust_all.columns if c.startswith('p_')]
    print(robust_all[['feature','n_methods_sig']+p_cols].head(10).to_string(index=False))

# ── Step 8: Age-stratified with MIN_N guard ──────────────────────
print(f"\n--- Step 8: Age-stratified (min n={MIN_N_PER_GROUP}/group) ---")
all_band_results=[]
for band in AGE_BANDS.keys():
    sub_clip =feat_df[feat_df['age_band']==band].copy()
    sub_child=child_df[child_df['age_band']==band].copy()
    n_asd =sub_child[sub_child['Group']=='ASD']['pid'].nunique()
    n_nasd=sub_child[sub_child['Group']=='Non-ASD']['pid'].nunique()
    min_n=min(n_asd,n_nasd)
    print(f"\n  [{band}] ASD={n_asd} Non-ASD={n_nasd} {len(sub_clip)} clips")
    if min_n<MIN_N_PER_GROUP:
        print(f"    ⚠ n={min_n}<{MIN_N_PER_GROUP} → HYPOTHESIS-GENERATING only")
        if n_asd>=3 and n_nasd>=3:
            band_perm=run_child_permutation(sub_child,CHILD_FEATS,n_perm=2000,subset_label=band)
            band_mw  =run_pseudobulk_mw(sub_child,CHILD_FEATS,band)
            band_dict={k:v for k,v in {'ChildPerm':band_perm,'PseudobulkMW':band_mw}.items() if v is not None and len(v)>0}
            if band_dict:
                band_cons=make_consensus(band_dict,CHILD_FEATS)
                band_cons['age_band']=band; band_cons['confirmatory']=False
                band_cons['n_asd']=n_asd; band_cons['n_nasd']=n_nasd
                all_band_results.append(band_cons)
                band_cons.to_csv(os.path.join(OUTPUT_DIR,f'stats_{band.replace("-","_")}_exploratory.csv'),index=False)
        continue
    band_lme =run_lme_kr(sub_clip,PRIMARY_FEATS,band,covariates='age_mo_c',interaction=False)
    band_cr2 =run_cr2(sub_clip,PRIMARY_FEATS,band)
    band_gee =run_gee(sub_clip,PRIMARY_FEATS,band) if len(sub_clip)>=30 else pd.DataFrame()
    band_perm=run_child_permutation(sub_child,CHILD_FEATS,n_perm=2000,subset_label=band)
    band_boot=run_wild_bootstrap(sub_child,CHILD_FEATS,n_boot=2000,subset_label=band)
    band_mw  =run_pseudobulk_mw(sub_child,CHILD_FEATS,band)
    band_dict={k:v for k,v in {
        'LME_KR':band_lme,'CR2':band_cr2,'GEE':band_gee,
        'ChildPerm':band_perm,'WildBoot':band_boot,'PseudobulkMW':band_mw,
    }.items() if v is not None and len(v)>0}
    if not band_dict: print("    No results."); continue
    band_cons=make_consensus(band_dict,CHILD_FEATS)
    band_cons['age_band']=band; band_cons['confirmatory']=True
    band_cons['n_asd']=n_asd; band_cons['n_nasd']=n_nasd
    all_band_results.append(band_cons)
    band_cons.to_csv(os.path.join(OUTPUT_DIR,f'stats_{band.replace("-","_")}_consensus.csv'),index=False)
    top3=band_cons[band_cons['n_methods_sig']>0].head(3)
    for _,r in top3.iterrows():
        print(f"    {r['feature']:<40} n_sig={r['n_methods_sig']}")

if all_band_results:
    pd.concat(all_band_results,ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR,'stats_age_stratified_all_methods.csv'),index=False)

# ── Step 9: Consistency gate (across age bands) ──────────────────

print("\n--- Step 9: Consistency gate (across age bands) ---")
sig_feats_for_gate=list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else []
cons_df,consistent_feats=run_consistency_gate_bands(child_df,CHILD_FEATS,sig_feats_for_gate)
if len(cons_df):
    cons_df.to_csv(os.path.join(OUTPUT_DIR,'stats_consistency_gate.csv'),index=False)
    print(f"  {len(consistent_feats)}/{len(sig_feats_for_gate)} features consistent across bands")
    for f in consistent_feats: print(f"    ✓ {f}")

# Exploratory clip-level MWU (pseudoreplication diagnostic)
def run_mwu_clips(df, feat_cols, subset='combined'):
    recs=[]
    for feat in feat_cols:
        av=df[df['Group']=='ASD'][feat].dropna().values
        nv=df[df['Group']=='Non-ASD'][feat].dropna().values
        if len(av)<3 or len(nv)<3: continue
        stat,p=stats.mannwhitneyu(av,nv,alternative='two-sided')
        d=cohen_d(av,nv); ci=bootstrap_ci_d(av,nv,n_boot=500)
        recs.append({'feature':feat,'subset':subset,'ASD_n':len(av),'NonASD_n':len(nv),
                     'ASD_median':float(np.median(av)),'NonASD_median':float(np.median(nv)),
                     'mw_stat':float(stat),'p_raw':float(p),
                     'cohens_d':d,'ci95_lo':ci[0],'ci95_hi':ci[1],
                     'NOTE':'EXPLORATORY_ONLY_clips_not_independent'})
    if not recs: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(recs).sort_values('p_raw'),'p_raw')

r_clip_mwu=run_mwu_clips(feat_df,PRIMARY_FEATS,'combined')
r_clip_mwu.to_csv(os.path.join(OUTPUT_DIR,'exploratory_clip_mwu.csv'),index=False)
print(f"\n  Exploratory clip MWU: sig_raw={r_clip_mwu['sig_raw05'].sum()}  "
      f"sig_FDR={r_clip_mwu['sig_fdr05'].sum()}  (NOT confirmatory)")

# ═══════════════════════════════════════════════════════════════════
# PART 3: BAYESIAN HIERARCHICAL LMM
# ═══════════════════════════════════════════════════════════════════
hr("PART 3: BAYESIAN HIERARCHICAL LMM")

bayes_main_results={}

if not RUN_BAYESIAN or not _PYMC_OK:
    print("  Bayesian skipped")
else:
    def _standardise(series):
        m,s=series.mean(),series.std(); s=s if s>1e-10 else 1.0
        return ((series-m)/s).values,m,s

    def _savage_dickey_bf(post, prior_sd=0.5):
        prior_at_0=spnorm.pdf(0,0,prior_sd)
        try:
            post_at_0=gaussian_kde(post)(0)[0]
            return float(prior_at_0/post_at_0) if post_at_0>0 else np.nan
        except: return np.nan

    def _build_bayes_df(df, feat):
        tmp=df[['pid','Group','age_mo',feat]].dropna().copy()
        if len(tmp)<8 or tmp['pid'].nunique()<4: return None
        tmp['Group_bin']=(tmp['Group']=='ASD').astype(float)
        tmp['age_c']=tmp['age_mo']-tmp['age_mo'].mean()
        y_z,ym,ys=_standardise(tmp[feat])
        pids,pid_idx=np.unique(tmp['pid'].values,return_inverse=True)
        return {'df':tmp,'y_z':y_z.astype(float),'group_bin':tmp['Group_bin'].values.astype(float),
                'age_c':tmp['age_c'].values.astype(float),
                'pid_idx':pid_idx,'n_pids':len(pids),'pid_labels':pids,
                'y_mean':ym,'y_std':ys,'n_obs':len(tmp)}

    def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                        chains=BAYES_CHAINS, seed=42):
        with pm.Model():
            alpha    =pm.Normal('alpha',0,1)
            b_group  =pm.Normal('b_group',0,prior_sd)
            b_age    =pm.Normal('b_age',0,0.5)
            sigma_pid=pm.HalfNormal('sigma_pid',1)
            sigma    =pm.HalfNormal('sigma',1)
            alpha_pid=pm.Normal('alpha_pid',0,sigma_pid,shape=bd['n_pids'])
            mu=(alpha+alpha_pid[bd['pid_idx']]+b_group*bd['group_bin']+b_age*bd['age_c'])
            pm.Normal('y_obs',mu=mu,sigma=sigma,observed=bd['y_z'])
            idata=pm.sample(draws=draws,tune=tune,chains=chains,
                            target_accept=0.9,random_seed=seed,
                            progressbar=False,return_inferencedata=True)
        b_post=idata.posterior['b_group'].values.flatten()
        hdi=az.hdi(idata,var_names=['b_group'],hdi_prob=0.94)['b_group'].values
        diag=az.summary(idata,var_names=['b_group'],hdi_prob=0.94)
        rhat=float(diag['r_hat'].values[0]); ess=float(diag['ess_bulk'].values[0])
        n_div=int(idata.sample_stats['diverging'].values.sum())
        bf10=_savage_dickey_bf(b_post,prior_sd)
        return idata,{'b_group_mean':float(b_post.mean()),'b_group_sd':float(b_post.std()),
                      'hdi94_lo':float(hdi[0]),'hdi94_hi':float(hdi[1]),
                      'p_pos':float((b_post>0).mean()),'bf10':bf10,
                      'rhat':rhat,'ess_bulk':ess,'n_divergences':n_div,
                      'converged':bool(rhat<1.05 and ess>400 and n_div==0),'prior_sd':prior_sd}

    def prior_predictive_check(bd, feat, prior_sd=0.5):
        with pm.Model():
            b_group=pm.Normal('b_group',0,prior_sd); b_age=pm.Normal('b_age',0,0.5)
            sigma=pm.HalfNormal('sigma',1); alpha=pm.Normal('alpha',0,1)
            mu=alpha+b_group*bd['group_bin']+b_age*bd['age_c']
            pm.Normal('y_obs',mu=mu,sigma=sigma,observed=bd['y_z'])
            ppc=pm.sample_prior_predictive(samples=200,random_seed=42)
        prior_ys=ppc.prior_predictive['y_obs'].values.flatten()
        obs_range=(bd['y_z'].min(),bd['y_z'].max())
        prior_range=(float(np.percentile(prior_ys,1)),float(np.percentile(prior_ys,99)))
        return {'feature':feat,'obs_min':obs_range[0],'obs_max':obs_range[1],
                'prior_p1':prior_range[0],'prior_p99':prior_range[1],
                'plausible':prior_range[0]<=obs_range[0] and prior_range[1]>=obs_range[1]}

    bayes_feats=(perm_all.sort_values('p_raw').head(15)['feature'].tolist()
                 if len(perm_all) else CHILD_FEATS[:10])
    print(f"\nRunning Bayesian models on {len(bayes_feats)} features...")
    ppc_records=[]; bayes_records=[]; sensitivity_records=[]

    for feat in bayes_feats:
        bd=_build_bayes_df(feat_df,feat)
        if bd is None: continue
        try:
            ppc_rec=prior_predictive_check(bd,feat)
            ppc_records.append(ppc_rec)
            if not ppc_rec['plausible']: print(f"  ⚠ PPC: prior too narrow for {feat}")
        except: pass
        bf_vals={}
        for psd in PRIOR_SDS:
            try:
                _,summ=_fit_bayes_main(bd,prior_sd=psd)
                summ['feature']=feat; summ['prior_sd']=psd
                sensitivity_records.append(summ); bf_vals[psd]=summ['bf10']
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")
        if 0.5 in bf_vals:
            match=[r for r in sensitivity_records if r['feature']==feat and r['prior_sd']==0.5]
            if match:
                rec=match[-1].copy()
                bfs=[bf_vals[p] for p in PRIOR_SDS if p in bf_vals and not np.isnan(bf_vals[p])]
                rec['bf_robust']=bool(len(bfs)>=2 and all((b>1)==(bfs[0]>1) for b in bfs))
                bayes_records.append(rec)
                flag='✓' if rec.get('converged') else '⚠'
                bf_str=' | '.join([f"sd={p}: BF={bf_vals.get(p,np.nan):.2f}" for p in PRIOR_SDS])
                print(f"  {feat:<40} {bf_str} {flag}")

    if ppc_records:
        pd.DataFrame(ppc_records).to_csv(os.path.join(OUTPUT_DIR,'bayes_ppc.csv'),index=False)
    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(os.path.join(OUTPUT_DIR,'bayes_sensitivity.csv'),index=False)
    if bayes_records:
        bayes_df=pd.DataFrame(bayes_records).sort_values('bf10',ascending=False)
        bayes_df.to_csv(os.path.join(OUTPUT_DIR,'bayes_main.csv'),index=False)
        bayes_main_results['full']=bayes_df
        print(f"\n  BF10>3:  {(bayes_df['bf10']>3).sum()}/{len(bayes_df)}")
        print(f"  BF10>10: {(bayes_df['bf10']>10).sum()}/{len(bayes_df)}")
        print(f"  BF robust: {bayes_df['bf_robust'].sum()}/{len(bayes_df)}")

# ═══════════════════════════════════════════════════════════════════
# PART 4: CLASSIFICATION (LOSO, child-level)
# ═══════════════════════════════════════════════════════════════════
hr("PART 4: CLASSIFICATION — CHILD-LEVEL LOSO")


all_clf_results={}
print("\n--- Combined child level ---")
for cname in ['LR','SVM','RF']:
    r=run_loso_child(child_df,CHILD_FEATS,clf_name=cname,subset_name='combined')
    if r: all_clf_results[f'combined_{cname}']=r

print("\n--- Age-stratified ---")
for band in AGE_BANDS.keys():
    sub=child_df[child_df['age_band']==band]
    n_a=(sub['Group']=='ASD').sum(); n_n=(sub['Group']=='Non-ASD').sum()
    if n_a>=4 and n_n>=4:
        for cname in ['LR','RF']:
            r=run_loso_child(sub,CHILD_FEATS,clf_name=cname,subset_name=band)
            if r: all_clf_results[f'{band}_{cname}']=r
    else:
        print(f"  {band}: ASD={n_a} Non-ASD={n_n} → skipped")

clf_rows=[{'subset':k,'clf':v['clf'],'auc':v['auc'],'ap':v['ap'],
           'perm_p':v['perm_p'],'n_features':v['n_features'],'n_subjects':v['n_subjects']}
          for k,v in all_clf_results.items()]
if clf_rows:
    pd.DataFrame(clf_rows).to_csv(os.path.join(OUTPUT_DIR,'classification_summary.csv'),index=False)

best_rf_key=next((k for k in all_clf_results if 'RF' in k and 'combined' in k),None)
feat_importance_df=pd.DataFrame()
if best_rf_key and 'feature_importance' in all_clf_results[best_rf_key]:
    feat_importance_df=all_clf_results[best_rf_key]['feature_importance']
    feat_importance_df.to_csv(os.path.join(OUTPUT_DIR,'rf_feature_importances.csv'),index=False)

# ═══════════════════════════════════════════════════════════════════
# PART 5: FIGURES
# ═══════════════════════════════════════════════════════════════════
hr("PART 5: FIGURES")

# Fig 1: Sample overview
print("Fig 1: Sample overview...")
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.suptitle('Running v2 — Sample Overview',fontweight='bold')
gc=child_df['Group'].value_counts()
bars=axes[0].bar(GROUPS,[gc.get(g,0) for g in GROUPS],
                 color=[COLORS[g] for g in GROUPS],width=0.5,edgecolor='white')
for bar in bars:
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                 str(int(bar.get_height())),ha='center',fontweight='bold')
axes[0].set_title('(a) Children per group'); axes[0].set_ylabel('N')
for grp in GROUPS:
    axes[1].hist(child_df[child_df['Group']==grp]['age_mo'],
                 bins=10,alpha=0.6,color=COLORS[grp],label=grp,edgecolor='white')
for band,(lo,hi) in AGE_BANDS.items():
    axes[1].axvspan(lo,hi,alpha=0.1,color=BAND_COLORS[band])
axes[1].set_title('(b) Age distribution'); axes[1].set_xlabel('Age (mo)'); axes[1].legend()
for grp in GROUPS:
    axes[2].hist(child_df[child_df['Group']==grp]['n_clips'],
                 bins=range(1,int(child_df['n_clips'].max())+2),
                 alpha=0.6,color=COLORS[grp],label=grp,edgecolor='white')
axes[2].set_title('(c) Running clips per child'); axes[2].set_xlabel('N clips'); axes[2].legend()
plt.tight_layout(); savefig(fig,'fig1_sample_overview.png')

# Fig 2: Effect sizes with bootstrap CI
print("Fig 2: Effect sizes...")
if len(lme_all)>0:
    res_plot=lme_all.copy()
    res_plot['label']=res_plot['feature'].map(RUNNING_FEAT_SHORT).fillna(res_plot['feature'].str.replace('_',' '))
    res_plot=res_plot.sort_values('cohens_d')
    fig,ax=plt.subplots(figsize=(11,max(6,len(res_plot)*0.35)))
    colors_bar=[ASD_COLOR if d>0 else NONASD_COLOR for d in res_plot['cohens_d']]
    ax.barh(res_plot['label'],res_plot['cohens_d'],color=colors_bar,edgecolor='white',height=0.7,alpha=0.85)
    if 'd_ci_lo' in res_plot.columns:
        ax.errorbar(res_plot['cohens_d'],range(len(res_plot)),
                    xerr=[res_plot['cohens_d']-res_plot['d_ci_lo'],
                          res_plot['d_ci_hi']-res_plot['cohens_d']],
                    fmt='none',color='black',capsize=3,lw=1)
    ax.axvline(0,color='black',lw=0.8)
    for t,ls in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
        for sign in [1,-1]: ax.axvline(sign*t,color='gray',lw=0.7,ls=ls,alpha=0.4)
    for j,(_,row) in enumerate(res_plot.iterrows()):
        if row.get('sig_fdr05'): ax.text(row['cohens_d']+0.01,j,'★',va='center',fontsize=10,color='gold')
        elif row.get('sig_raw05'): ax.text(row['cohens_d']+0.01,j,'●',va='center',fontsize=8)
    method_label=lme_all['method'].mode()[0] if 'method' in lme_all.columns else 'LME'
    ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)")
    ax.set_title(f"Effect Sizes — {method_label}\n★=FDR sig  Bars=95% bootstrap CI",fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR,label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR,label='Non-ASD higher')])
    plt.tight_layout(); savefig(fig,'fig2_effect_sizes.png')

# Fig 3: Consensus heatmap
print("Fig 3: Consensus heatmap...")
if len(consensus_all)>0:
    p_cols=[c for c in consensus_all.columns if c.startswith('p_')]
    heat_data=consensus_all.set_index('feature')[p_cols].head(20)
    heat_log=-np.log10(heat_data.clip(lower=1e-5,upper=1.0).astype(float))
    fig,ax=plt.subplots(figsize=(len(p_cols)*2+2,max(6,len(heat_data)*0.4)))
    im=ax.imshow(heat_log.values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=4)
    ax.set_xticks(range(len(p_cols)))
    ax.set_xticklabels([c.replace('p_','') for c in p_cols],rotation=30,ha='right')
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels([FEAT_LABEL(f) for f in heat_data.index],fontsize=9)
    for i in range(heat_log.shape[0]):
        for j in range(heat_log.shape[1]):
            raw_p=heat_data.values[i,j]
            ax.text(j,i,f'{raw_p:.3f}{"*" if raw_p<0.05 else ""}',ha='center',va='center',fontsize=7)
    plt.colorbar(im,ax=ax,label='-log10(p)')
    ax.set_title('Consensus p-values (top 20 features)',fontweight='bold')
    plt.tight_layout(); savefig(fig,'fig3_consensus_heatmap.png')

# Fig 4: Violin plots
print("Fig 4: Violin plots...")
top12_feats=lme_all.head(12)['feature'].tolist() if len(lme_all)>=12 else lme_all['feature'].tolist()
if top12_feats:
    ncols=4; nrows=int(np.ceil(len(top12_feats)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4.5*nrows))
    fig.suptitle('Top Running Features — ASD vs Non-ASD',fontweight='bold')
    axes=axes.flatten()
    for i,feat in enumerate(top12_feats):
        ax=axes[i]
        dg=[feat_df[feat_df['Group']==g][feat].dropna().values for g in GROUPS]
        if any(len(d)==0 for d in dg): ax.set_visible(False); continue
        parts=ax.violinplot(dg,positions=[0,1],showmedians=True,showextrema=False)
        for j,pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for j,vals in enumerate(dg):
            ax.scatter(j+np.random.uniform(-0.07,0.07,len(vals)),vals,
                       color=list(COLORS.values())[j],alpha=0.2,s=8,zorder=3)
        row_s=lme_all[lme_all['feature']==feat]
        if len(row_s):
            p_r=row_s['p_raw'].values[0]; p_f=row_s['p_fdr'].values[0]; d=row_s['cohens_d'].values[0]
            col='#cc0000' if p_f<0.05 else ('#ff8800' if p_r<0.05 else 'gray')
            ax.text(0.5,0.97,f'LME p={p_r:.3f}|FDR={p_f:.3f}|d={d:.2f}',
                    transform=ax.transAxes,ha='center',va='top',fontsize=7.5,color=col)
            ymax=max(np.percentile(d2,95) for d2 in dg if len(d2))
            yr=ymax-min(np.percentile(d2,5) for d2 in dg if len(d2))
            add_sig_bar(ax,0,1,ymax+yr*0.05,p_r,h=yr*0.04)
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS,fontsize=9)
        ax.set_title(FEAT_LABEL(feat),fontsize=9,fontweight='bold')
    for j in range(len(top12_feats),len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); savefig(fig,'fig4_violins.png')

# Fig 5: Child-level boxplots
print("Fig 5: Child boxplots...")
BOX_FEATS=[f for f in ['hip_y_amplitude','knee_angle_L_range','cadence',
                        'flight_phase_ratio','arm_drive_asymmetry','lateral_sway_std']
           if f in child_df.columns]
if BOX_FEATS:
    fig,axes=plt.subplots(1,len(BOX_FEATS),figsize=(4.5*len(BOX_FEATS),5))
    if len(BOX_FEATS)==1: axes=[axes]
    fig.suptitle('Child-Level Averages (each dot = one child)',fontweight='bold')
    for i,feat in enumerate(BOX_FEATS):
        ax=axes[i]
        for j,grp in enumerate(GROUPS):
            vals=child_df[child_df['Group']==grp][feat].dropna().values
            if len(vals)==0: continue
            bp=ax.boxplot(vals,positions=[j],widths=0.45,patch_artist=True,
                          showfliers=False,medianprops={'color':'black','linewidth':2})
            bp['boxes'][0].set_facecolor(COLORS_LIGHT[grp])
            bp['boxes'][0].set_edgecolor(COLORS[grp]); bp['boxes'][0].set_linewidth(1.5)
            ax.scatter(j+np.random.uniform(-0.1,0.1,len(vals)),vals,
                       color=COLORS[grp],alpha=0.6,s=25,zorder=4)
        if len(perm_all):
            row_s=perm_all[perm_all['feature']==feat]
            if len(row_s):
                p_perm=row_s['p_raw'].values[0]
                ymax=child_df[feat].dropna().max()
                add_sig_bar(ax,0,1,ymax*1.05,p_perm,h=ymax*0.04)
                ax.text(0.5,0.97,f'Perm p={p_perm:.3f}',transform=ax.transAxes,
                        ha='center',va='top',fontsize=8,color='gray')
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS,fontsize=9)
        ax.set_title(FEAT_LABEL(feat),fontsize=9)
    plt.tight_layout(); savefig(fig,'fig5_child_boxplots.png')

# Fig 6: Age-stratified bars
print("Fig 6: Age-stratified bars...")
STRAT_PLOT=[f for f in ['hip_y_amplitude','knee_angle_L_range','cadence','flight_phase_ratio']
            if f in feat_df.columns]
if STRAT_PLOT:
    band_list=list(AGE_BANDS.keys())
    fig,axes=plt.subplots(len(STRAT_PLOT),len(band_list),
                           figsize=(4.5*len(band_list),4*len(STRAT_PLOT)),sharey='row')
    fig.suptitle('Running Features by Age Band',fontweight='bold')
    for ri,feat in enumerate(STRAT_PLOT):
        for ci,band in enumerate(band_list):
            ax=axes[ri][ci]
            sub=feat_df[feat_df['age_band']==band]
            da=sub[sub['Group']=='ASD'][feat].dropna().values
            dn=sub[sub['Group']=='Non-ASD'][feat].dropna().values
            means=[da.mean() if len(da) else 0,dn.mean() if len(dn) else 0]
            sems=[stats.sem(da) if len(da)>1 else 0,stats.sem(dn) if len(dn)>1 else 0]
            ax.bar([0,1],means,yerr=sems,color=[COLORS['ASD'],COLORS['Non-ASD']],
                   capsize=5,width=0.5,edgecolor='white',alpha=0.85)
            for j,(vals,xp) in enumerate([(da,0),(dn,1)]):
                if len(vals):
                    ax.scatter(xp+np.random.uniform(-0.1,0.1,len(vals)),vals,
                               color=list(COLORS.values())[j],alpha=0.4,s=10)
            n_a=sub[sub['Group']=='ASD']['pid'].nunique()
            n_n=sub[sub['Group']=='Non-ASD']['pid'].nunique()
            conf_str='' if min(n_a,n_n)>=MIN_N_PER_GROUP else '⚠explor'
            if len(da)>=3 and len(dn)>=3:
                _,p_=stats.mannwhitneyu(da,dn,alternative='two-sided')
                ymax=max(means)+max(sems)+abs(max(means))*0.05
                add_sig_bar(ax,0,1,ymax,p_,h=max(abs(ymax)*0.04,0.001))
            bcolor=BAND_COLORS.get(band,'gray')
            for sp in ax.spines.values():
                sp.set_edgecolor(bcolor)
                sp.set_linewidth(2.0 if min(n_a,n_n)>=MIN_N_PER_GROUP else 0.5)
            if ri==0: ax.set_title(f'{band} {conf_str}',fontsize=9,color=bcolor)
            if ci==0: ax.set_ylabel(FEAT_LABEL(feat),fontsize=8)
            ax.set_xticks([0,1]); ax.set_xticklabels(['ASD','NASD'],fontsize=8)
            ax.text(0.5,-0.22,f'n={n_a}/{n_n}',transform=ax.transAxes,ha='center',fontsize=7.5,color='gray')
    plt.tight_layout(); savefig(fig,'fig6_age_bands.png')

# Fig 7: Consistency gate
print("Fig 7: Consistency gate...")
if len(cons_df)>0:
    fig,ax=plt.subplots(figsize=(10,max(4,len(cons_df)*0.45)))
    cols_cg=[ASD_COLOR if v else NONASD_COLOR for v in cons_df['consistent']]
    ax.barh(cons_df['feature'].map(RUNNING_FEAT_SHORT).fillna(cons_df['feature']),
            cons_df['n_same_direction']/cons_df['n_bands_tested'],
            color=cols_cg,edgecolor='white',height=0.6)
    ax.axvline(1.0,color='green',lw=1.5,ls='--',label='All bands consistent')
    ax.axvline(0.5,color='orange',lw=1,ls=':',label='50%')
    ax.set_xlim(0,1.15); ax.set_xlabel('Fraction of age bands with same direction')
    ax.set_title('Consistency Gate — Effect Direction Across Age Bands',fontweight='bold')
    ax.legend(); plt.tight_layout(); savefig(fig,'fig7_consistency_gate.png')

# Fig 8: Bayesian forest
print("Fig 8: Bayesian forest...")
if 'full' in bayes_main_results and len(bayes_main_results['full'])>0:
    bdf=bayes_main_results['full'].copy()
    bdf['label']=bdf['feature'].map(RUNNING_FEAT_SHORT).fillna(bdf['feature'])
    bdf=bdf.sort_values('b_group_mean')
    fig,ax=plt.subplots(figsize=(13,max(5,len(bdf)*0.5)))
    for j,(_,row) in enumerate(bdf.iterrows()):
        col=ASD_COLOR if row['b_group_mean']>0 else NONASD_COLOR
        ax.plot([row['hdi94_lo'],row['hdi94_hi']],[j,j],color=col,lw=2.5,alpha=0.8)
        ax.scatter(row['b_group_mean'],j,color=col,s=70,zorder=5)
        ax.plot(row['hdi94_lo'],j,'|',color=col,markersize=8)
        ax.plot(row['hdi94_hi'],j,'|',color=col,markersize=8)
        bf=float(row['bf10']) if not np.isnan(float(row['bf10'])) else 0
        bf_str=f"BF={bf:.1f}"
        if not row.get('converged',True): bf_str+=' ⚠'
        if not row.get('bf_robust',True): bf_str+=' [prior-sensitive]'
        ax.text(row['hdi94_hi']+0.01,j,bf_str,va='center',fontsize=7)
    ax.axvline(0,color='black',lw=1.2,ls='--')
    ax.set_yticks(range(len(bdf))); ax.set_yticklabels(bdf['label'],fontsize=9)
    ax.set_xlabel('Posterior mean  |  94% HDI  (standardised units)')
    ax.set_title('Bayesian Hierarchical LMM — Running\n⚠=convergence issue  [prior-sensitive]=BF changed across priors',fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR,label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR,label='Non-ASD higher')])
    plt.tight_layout(); savefig(fig,'fig8_bayes_forest.png')

# Fig 9: Prior sensitivity
print("Fig 9: Prior sensitivity...")
if os.path.isfile(os.path.join(OUTPUT_DIR,'bayes_sensitivity.csv')):
    sens=pd.read_csv(os.path.join(OUTPUT_DIR,'bayes_sensitivity.csv'))
    if len(sens):
        feats_s=sens['feature'].unique()[:12]
        fig,axes=plt.subplots(int(np.ceil(len(feats_s)/3)),3,
                               figsize=(15,4*int(np.ceil(len(feats_s)/3))))
        fig.suptitle('Prior Sensitivity — BF10 across prior widths',fontweight='bold')
        axes=axes.flatten()
        for i,feat in enumerate(feats_s):
            ax=axes[i]; sub=sens[sens['feature']==feat].sort_values('prior_sd')
            ax.plot(sub['prior_sd'],sub['bf10'],marker='o',color=ASD_COLOR,lw=2)
            ax.axhline(3,color='green',lw=1,ls='--',label='BF=3')
            ax.axhline(1,color='gray',lw=0.8,ls=':')
            ax.set_xlabel('Prior SD'); ax.set_ylabel('BF10')
            ax.set_title(FEAT_LABEL(feat)[:25],fontsize=9); ax.legend(fontsize=7)
        for j in range(len(feats_s),len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); savefig(fig,'fig9_prior_sensitivity.png')

# Fig 10: Developmental trajectories
print("Fig 10: Trajectories...")
TRAJ_FEATS=[f for f in ['hip_y_amplitude','hip_x_vel_mean','knee_angle_L_range',
                         'cadence','flight_phase_ratio','arm_drive_asymmetry']
            if f in feat_df.columns]
if TRAJ_FEATS:
    ncols=3; nrows=int(np.ceil(len(TRAJ_FEATS)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,4.5*nrows))
    fig.suptitle('Developmental Trajectories — Running',fontweight='bold')
    axes=axes.flatten()
    for i,feat in enumerate(TRAJ_FEATS):
        ax=axes[i]
        for grp in GROUPS:
            sub=feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
            if len(sub)<3: continue
            ax.scatter(sub['age_mo'],sub[feat],color=COLORS[grp],alpha=0.25,s=12)
            if len(sub)>=5:
                m_,b_,r_,p_,_=stats.linregress(sub['age_mo'],sub[feat])
                xr=np.linspace(sub['age_mo'].min(),sub['age_mo'].max(),100)
                ax.plot(xr,m_*xr+b_,color=COLORS[grp],lw=2.5,
                        label=f'{grp} r={r_:.2f} p={p_:.3f}')
        for band,(lo,hi) in AGE_BANDS.items():
            ax.axvspan(lo,hi,alpha=0.07,color=BAND_COLORS[band])
        ax.set_xlabel('Age (months)'); ax.legend(fontsize=8)
        ax.set_title(FEAT_LABEL(feat),fontsize=9,fontweight='bold')
    for j in range(len(TRAJ_FEATS),len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); savefig(fig,'fig10_trajectories.png')

# Fig 11: Classification ROC
print("Fig 11: Classification ROC...")
if all_clf_results:
    keys=list(all_clf_results.keys()); n=len(keys)
    ncols=min(n,4); nrows=int(np.ceil(n/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4.5*nrows))
    if nrows*ncols==1: axes=np.array([[axes]])
    elif nrows==1: axes=axes.reshape(1,-1)
    fig.suptitle('Classification ROC — Child-Level LOSO',fontweight='bold')
    for i,key in enumerate(keys):
        r=all_clf_results[key]; ax=axes[i//ncols][i%ncols]
        fpr,tpr,_=roc_curve(r['y_true'],r['y_score'])
        ax.plot(fpr,tpr,color=ASD_COLOR,lw=2,label=f"AUC={r['auc']:.3f}  AP={r['ap']:.3f}")
        ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
        ax.fill_between(fpr,tpr,alpha=0.1,color=ASD_COLOR)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f"{key}\np_perm={r['perm_p']:.3f}",fontsize=8); ax.legend(fontsize=8)
        if r.get('perm_aucs'):
            axins=ax.inset_axes([0.55,0.05,0.4,0.28])
            axins.hist(r['perm_aucs'],bins=20,color='gray',alpha=0.7)
            axins.axvline(r['auc'],color=ASD_COLOR,lw=2)
            axins.set_title('Null dist',fontsize=6); axins.tick_params(labelsize=5)
    for i in range(len(keys),nrows*ncols): axes[i//ncols][i%ncols].set_visible(False)
    plt.tight_layout(); savefig(fig,'fig11_roc.png')

# Fig 12: RF importances + running-specific panel
print("Fig 12: RF importances + running-specific...")
fig=plt.figure(figsize=(18,12))
fig.suptitle('Running-Specific Biomechanical Signatures + RF Importances',fontweight='bold',fontsize=13)
gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.38)

ax_rf=fig.add_subplot(gs[:,0])
if len(feat_importance_df)>0:
    top15=feat_importance_df.head(15)
    top15['label']=top15['feature'].map(RUNNING_FEAT_SHORT).fillna(top15['feature'])
    ax_rf.barh(top15['label'],top15['importance'],color=ASD_COLOR,alpha=0.8,edgecolor='white')
    ax_rf.set_xlabel('RF Importance'); ax_rf.set_title('RF Feature Importances',fontweight='bold')

ax_b=fig.add_subplot(gs[0,1])
if 'hip_y_amplitude' in feat_df.columns:
    band_names=list(AGE_BANDS.keys()); x=np.arange(len(band_names)); width=0.35
    for gi,grp in enumerate(GROUPS):
        means=[feat_df[(feat_df['age_band']==band)&(feat_df['Group']==grp)]['hip_y_amplitude'].mean()
               for band in band_names]
        ax_b.bar(x+gi*width,means,width,label=grp,color=COLORS[grp],edgecolor='white',alpha=0.85)
    ax_b.set_xticks(x+width/2); ax_b.set_xticklabels(band_names,fontsize=8)
    ax_b.set_title('(B) Vertical Bounce by Age Band',fontsize=10,fontweight='bold'); ax_b.legend(fontsize=8)

ax_c=fig.add_subplot(gs[0,2])
cad_feat='cadence' if 'cadence' in feat_df.columns else 'ankle_y_L_dom_freq'
if cad_feat in feat_df.columns:
    for grp in GROUPS:
        vals=feat_df[feat_df['Group']==grp][cad_feat].dropna()
        ax_c.hist(vals,bins=20,alpha=0.6,color=COLORS[grp],label=grp,density=True,edgecolor='white')
    ax_c.set_title('(C) Running Cadence Distribution',fontsize=10,fontweight='bold'); ax_c.legend(fontsize=8)

ax_f=fig.add_subplot(gs[1,1])
if 'flight_phase_ratio' in feat_df.columns:
    dg=[feat_df[feat_df['Group']==g]['flight_phase_ratio'].dropna().values for g in GROUPS]
    if all(len(d)>0 for d in dg):
        parts=ax_f.violinplot(dg,positions=[0,1],showmedians=True,showextrema=False)
        for j,pc in enumerate(parts['bodies']): pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        ax_f.set_xticks([0,1]); ax_f.set_xticklabels(GROUPS,fontsize=9)
        ax_f.set_title('(D) Flight Phase Ratio',fontsize=10,fontweight='bold')

ax_ps=fig.add_subplot(gs[1,2])
if len(r_clip_mwu) and len(mw_all):
    merged=r_clip_mwu[['feature','p_raw']].rename(columns={'p_raw':'p_clip'}).merge(
        mw_all[['feature','p_raw']].rename(columns={'p_raw':'p_child'}),on='feature').dropna()
    ax_ps.scatter(-np.log10(merged['p_clip']+1e-10),-np.log10(merged['p_child']+1e-10),
                  alpha=0.6,color=ASD_COLOR,s=35)
    lim=max(-np.log10(merged[['p_clip','p_child']].min(skipna=True).min()+1e-10),1)
    ax_ps.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.6,label='y=x (no inflation)')
    ax_ps.axhline(-np.log10(0.05),color='gray',lw=0.8,ls=':',alpha=0.6)
    ax_ps.axvline(-np.log10(0.05),color='gray',lw=0.8,ls=':',alpha=0.6)
    ax_ps.set_xlabel('-log10(p) clip MWU [inflated]')
    ax_ps.set_ylabel('-log10(p) pseudo-bulk MWU [confirmatory]')
    ax_ps.set_title('(E) Pseudoreplication Diagnostic',fontsize=10,fontweight='bold')
    ax_ps.legend(fontsize=8)

plt.tight_layout(); savefig(fig,'fig12_running_specific_rf.png')

# ═══════════════════════════════════════════════════════════════════
# PART 6: SUMMARY
# ═══════════════════════════════════════════════════════════════════
hr("PART 6: SUMMARY")
print(f"\nOutputs → {OUTPUT_DIR}")
print(f"Figures → {FIG_DIR}\n")
print("--- CSVs ---")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith('.csv'):
        try:
            tmp=pd.read_csv(os.path.join(OUTPUT_DIR,fname))
            print(f"  {fname:<65} {tmp.shape[0]:>5}r × {tmp.shape[1]:>3}c")
        except: print(f"  {fname}")
print("\n--- Figures ---")
for fname in sorted(os.listdir(FIG_DIR)):
    if fname.endswith('.png'): print(f"  {fname}")
print("\n--- KEY RESULTS ---")
print(f"\n  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)}")
if len(lme_all):
    print(f"  LME ({lme_all['method'].mode()[0]}): sig_raw={lme_all['sig_raw05'].sum()} sig_FDR={lme_all['sig_fdr05'].sum()}")
    print(f"  Random slope: {lme_all['random_slope_used'].any()}")
print(f"  Robust features (≥{CONSENSUS_THRESHOLD} methods): {len(robust_all)}")
for _,r in robust_all.head(8).iterrows():
    d_str=f"  d={r['cohens_d_LME']:.2f}" if 'cohens_d_LME' in r and pd.notna(r['cohens_d_LME']) else ''
    print(f"    {r['feature']:<40} n_sig={r['n_methods_sig']}{d_str}")
print(f"  Consistency gate: {len(consistent_feats)}/{len(sig_feats_for_gate)} passed")
if clf_rows:
    print("\n  Classification (LOSO AUC):")
    for row in sorted(clf_rows,key=lambda x:-x['auc'])[:6]:
        sig='✓' if row['perm_p']<0.05 else ''
        print(f"    {row['subset']:<35} {row['clf']}  AUC={row['auc']:.3f}  p_perm={row['perm_p']:.4f} {sig}")
print("\n  Age band status:")
for band in AGE_BANDS.keys():
    sub=child_df[child_df['age_band']==band]
    n_a=(sub['Group']=='ASD').sum(); n_n=(sub['Group']=='Non-ASD').sum()
    status='CONFIRMATORY' if min(n_a,n_n)>=MIN_N_PER_GROUP else f'EXPLORATORY (n<{MIN_N_PER_GROUP})'
    print(f"    {band}: ASD={n_a} Non-ASD={n_n} → {status}")
print("\n  Running-specific features detected:")
for feat in ['hip_y_amplitude','flight_phase_ratio','cadence','arm_drive_asymmetry','vertical_oscillation_ratio']:
    if feat in feat_df.columns:
        n_obs=feat_df[feat].notna().sum()
        print(f"    {feat:<40} n_obs={n_obs}")
hr("RUNNING ANALYSIS v2 COMPLETE")