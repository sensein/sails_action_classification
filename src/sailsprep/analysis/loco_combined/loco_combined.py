#!/usr/bin/env python3
"""
locomotion analysis
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

# Optional imports — graceful degradation
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    _lme4    = importr('lme4')
    _lmerTest = importr('lmerTest')
    _RPY2_OK = True
    print("[rpy2] lme4 + lmerTest available — Kenward-Roger enabled")
except Exception:
    _RPY2_OK = False
    print("[rpy2] NOT available — falling back to statsmodels LME (no KR correction)")

try:
    import arviz as az
    import pymc as pm
    _PYMC_OK = True
    print("[PyMC] available — Bayesian models enabled")
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
OUTPUT_DIR = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/analysis/loco_combined/v3"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

FPS        = 15.0
MIN_CONF   = 0.3
MIN_FRAMES = 15

LOCO_LABELS    = {'Walking', 'Crawling', 'Cruising', 'Running'}
LOCO_REFERENCE = 'Walking'   # dummy-coding reference level

AGE_BINS_MO = [0, 18, 32, 38, 60]
AGE_LABELS  = ['11-18mo', '19-31mo', '32-38mo', '39-49mo']
KEY_BANDS   = ['11-18mo', '32-38mo']

# Age streams (mirrors RMM design)
AGE_STREAMS = {
    'full':    None,
    '11-18mo': (11, 18),
    '32-38mo': (32, 38),
}

BAYES_DRAWS  = 2000
BAYES_TUNE   = 1000
BAYES_CHAINS = 4
RUN_BAYESIAN = True

ASD_COLOR     = '#D55E00'
NONASD_COLOR  = '#0072B2'
ASD_LIGHT     = '#F4A582'
NONASD_LIGHT  = '#92C5DE'
COLORS        = {'ASD': ASD_COLOR,  'Non-ASD': NONASD_COLOR}
COLORS_LIGHT  = {'ASD': ASD_LIGHT,  'Non-ASD': NONASD_LIGHT}
STREAM_COLORS = {'full': '#555555', '11-18mo': '#7B5EA7', '32-38mo': '#D47C2A'}
BAND_COLORS   = {
    '11-18mo': '#D55E00', '19-31mo': '#56B4E9',
    '32-38mo': '#009E73', '39-49mo': '#CC79A7',
}
GROUPS = ['ASD', 'Non-ASD']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
})

# ═══════════════════════════════════════════════════════════════════
# HRNET-133 KEYPOINT MAP
# ═══════════════════════════════════════════════════════════════════
KP = {
    'L_Shoulder': 'kp_005', 'R_Shoulder': 'kp_006',
    'L_Elbow':    'kp_007', 'R_Elbow':    'kp_008',
    'L_Wrist':    'kp_009', 'R_Wrist':    'kp_010',
    'L_Hip':      'kp_011', 'R_Hip':      'kp_012',
    'L_Knee':     'kp_013', 'R_Knee':     'kp_014',
    'L_Ankle':    'kp_015', 'R_Ankle':    'kp_016',
    'L_Heel':     'kp_019', 'R_Heel':     'kp_022',
    'R_BigToe':   'kp_020',
}

FEAT_LABELS = {
    'hip_speed_mean':'Hip Speed (mean)','hip_speed_std':'Hip Speed (variability)',
    'hip_speed_cv':'Hip Speed (CV)','hip_jerk_cost':'Hip Jerk (smoothness)',
    'knee_jerk_L':'Knee Jerk L','knee_jerk_R':'Knee Jerk R',
    'hip_bilateral_corr':'Hip Bilateral Corr','ankle_bilateral_corr':'Ankle Bilateral Corr',
    'knee_bilateral_corr':'Knee Bilateral Corr','bilateral_phase_lag':'Phase Lag (sec)',
    'knee_dom_freq_L':'Knee Freq L (Hz)','knee_dom_freq_R':'Knee Freq R (Hz)',
    'knee_ac_L':'Knee Periodicity L','knee_ac_R':'Knee Periodicity R',
    'knee_spectral_entropy_L':'Cadence Entropy L','hip_dom_freq':'Hip Freq (Hz)',
    'knee_angle_L_mean':'Knee Angle L (mean)','knee_angle_R_mean':'Knee Angle R (mean)',
    'knee_angle_L_std':'Knee Angle L (var)','knee_angle_R_std':'Knee Angle R (var)',
    'hip_angle_L_mean':'Hip Angle L (mean)','hip_angle_R_mean':'Hip Angle R (mean)',
    'step_len_proxy':'Step Length Proxy','ankle_y_asym':'Ankle Asymmetry',
    'hip_sway_lateral_std':'Lateral Sway','hip_vert_std':'Vertical Hip Osc.',
    'toe_walk_proxy_L':'Toe-Walk Proxy L','toe_walk_proxy_R':'Toe-Walk Proxy R',
    'arm_swing_L':'Arm Swing L','arm_swing_R':'Arm Swing R',
}

# ═══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════
def hr(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")

def extract_pid(path):
    if not isinstance(path, str): return None
    m = re.search(r'(sub-[A-Za-z0-9]+)', path)
    return m.group(1) if m else None

def extract_session(path):
    if not isinstance(path, str): return None
    m = re.search(r'ses-(\d+)', path)
    return int(m.group(1)) if m else None

def get_kp(fd, name, min_conf=MIN_CONF):
    key = KP.get(name)
    if key is None or key not in fd: return None
    kp = fd[key]
    if kp.get('confidence', 0) < min_conf: return None
    return kp

def torso_length(fd):
    ls = get_kp(fd,'L_Shoulder',0.1); rs = get_kp(fd,'R_Shoulder',0.1)
    lh = get_kp(fd,'L_Hip',0.1);      rh = get_kp(fd,'R_Hip',0.1)
    if not all([ls,rs,lh,rh]): return None
    sx=(ls['x']+rs['x'])/2; sy=(ls['y']+rs['y'])/2
    hx=(lh['x']+rh['x'])/2; hy=(lh['y']+rh['y'])/2
    d=np.sqrt((sx-hx)**2+(sy-hy)**2)
    return d if d>10 else None

def compute_angle(p1,p2,p3):
    v1=np.array(p1)-np.array(p2); v2=np.array(p3)-np.array(p2)
    cos_a=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))

def butter_lp(data, cutoff=4.0, fs=15.0, order=2):
    arr=np.array(data,dtype=float)
    if len(arr)<12: return arr
    nyq=0.5*fs
    b,a=butter(order,min(cutoff,nyq*0.9)/nyq,btype='low')
    if len(arr)<3*max(len(b),len(a)): return arr
    return filtfilt(b,a,arr)

def dominant_freq(arr, fps=FPS):
    arr=np.array(arr)
    if len(arr)<16: return np.nan
    try:
        freqs,psd=welch(arr,fs=fps,nperseg=min(len(arr),64))
        return float(freqs[np.argmax(psd)])
    except: return np.nan

def spectral_entropy(arr, fps=FPS):
    arr=np.array(arr)
    if len(arr)<16: return np.nan
    try:
        _,psd=welch(arr,fs=fps,nperseg=min(len(arr),64))
        pn=psd/(psd.sum()+1e-12)
        return float(-np.sum(pn*np.log2(pn+1e-12)))
    except: return np.nan

def jerk_cost(arr, fps=FPS):
    arr=np.array(arr)
    if len(arr)<6: return np.nan
    vel=np.diff(arr)*fps; acc=np.diff(vel)*fps; jerk=np.diff(acc)*fps
    dur=len(arr)/fps; amp=np.ptp(arr)
    if amp<1e-8: return np.nan
    return float(np.sqrt(0.5*np.trapz(jerk**2)/dur)*(dur**2.5/amp**2))

def ac_strength(arr):
    arr=np.array(arr)
    if len(arr)<4: return np.nan
    arr=arr-arr.mean()
    denom=np.dot(arr,arr)
    if denom<1e-10: return np.nan
    return float(np.dot(arr[:-1],arr[1:])/denom)

def cohen_d(a, b):
    a,b=np.array(a,dtype=float),np.array(b,dtype=float)
    pooled=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return float((np.mean(a)-np.mean(b))/pooled) if pooled>0 else 0.0

def bootstrap_ci_d(a, b, n_boot=1000, seed=42):
    rng=np.random.default_rng(seed)
    boot=[cohen_d(rng.choice(a,len(a),replace=True),
                  rng.choice(b,len(b),replace=True))
          for _ in range(n_boot)]
    return float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))

def fdr_annotate(df_res, p_col):
    if len(df_res)>1:
        _,p_fdr,_,_=multipletests(df_res[p_col].fillna(1),method='fdr_bh')
        df_res=df_res.copy(); df_res['p_fdr']=p_fdr
    else:
        df_res=df_res.copy(); df_res['p_fdr']=df_res[p_col]
    df_res['sig_fdr05']=df_res['p_fdr']<0.05
    df_res['sig_raw05']=df_res[p_col]<0.05
    return df_res

def add_sig_bar(ax,x1,x2,y,p,h=0.02):
    label='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col='#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],lw=1.2,color='black')
    ax.text((x1+x2)/2,y+h*1.05,label,ha='center',va='bottom',
            fontsize=10,color=col,fontweight='bold')

def get_contiguous_segments(ldf, col, valid_labels):
    ldf=ldf.reset_index(drop=True)
    mask=ldf[col].isin(valid_labels)
    segs,in_seg,start,end=[],False,None,None
    for idx,row in ldf.iterrows():
        if mask[idx]:
            if not in_seg: start=int(row['Frame']); in_seg=True
            end=int(row['Frame'])
        else:
            if in_seg: segs.append((start,end)); in_seg=False
    if in_seg: segs.append((start,end))
    return segs

def stream_filter(df, stream_key):
    bounds=AGE_STREAMS[stream_key]
    if bounds is None: return df.copy()
    lo,hi=bounds
    return df[(df['age_mo']>=lo)&(df['age_mo']<=hi)].copy()

# ═══════════════════════════════════════════════════════════════════
# PART 0: LOAD METADATA
# ═══════════════════════════════════════════════════════════════════
hr("PART 0: LOAD METADATA")

df=pd.read_csv(MAIN_CSV)
df['pid']     =df['video_path'].apply(extract_pid)
df['session'] =df['video_path'].apply(extract_session)
df['age_mo']  =df['Age']*12
df['age_band']=pd.cut(df['age_mo'],bins=AGE_BINS_MO,labels=AGE_LABELS,right=False).astype(str)
df=df[
    df['pid'].notna() &
    df['Group'].isin(['ASD','Non-ASD']) &
    df['label_path'].apply(lambda p: isinstance(p,str) and os.path.isfile(p)) &
    df['hrnet_full_path'].apply(lambda p: isinstance(p,str) and os.path.isfile(p))
].copy()

print(f"Valid rows: {len(df)}")
print(f"Unique children: {df['pid'].nunique()}")
print(f"  ASD:     {df[df['Group']=='ASD']['pid'].nunique()}")
print(f"  Non-ASD: {df[df['Group']=='Non-ASD']['pid'].nunique()}")
print(df.groupby(['age_band','Group']).size().reset_index(name='n').to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# PART 1: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
hr("PART 1: FEATURE EXTRACTION")

def extract_locomotion_features(pose_frames, frame_indices, fps=FPS):
    hip_xy=[];  hip_x_arr=[]; hip_y_arr=[]
    knee_y_L=[]; knee_y_R=[]; ankle_y_L=[]; ankle_y_R=[]
    knee_ang_L=[]; knee_ang_R=[]; hip_ang_L=[]; hip_ang_R=[]
    wrist_y_L=[]; wrist_y_R=[]; lh_y=[]; rh_y=[]; tors=[]; n_valid=0

    for fi in frame_indices:
        fd=pose_frames.get(str(fi))
        if fd is None: continue
        tl=torso_length(fd)
        if tl is None: continue
        lh=get_kp(fd,'L_Hip'); rh=get_kp(fd,'R_Hip')
        lk=get_kp(fd,'L_Knee'); rk=get_kp(fd,'R_Knee')
        la=get_kp(fd,'L_Ankle'); ra=get_kp(fd,'R_Ankle')
        if lh is None and rh is None: continue
        n_valid+=1; tors.append(tl)
        if lh and rh: hx=(lh['x']+rh['x'])/2; hy=(lh['y']+rh['y'])/2
        elif lh:      hx,hy=lh['x'],lh['y']
        else:         hx,hy=rh['x'],rh['y']
        hip_xy.append([hx/tl,hy/tl]); hip_x_arr.append(hx/tl); hip_y_arr.append(hy/tl)
        if lh: lh_y.append(lh['y']/tl)
        if rh: rh_y.append(rh['y']/tl)
        if lk: knee_y_L.append(lk['y']/tl)
        if rk: knee_y_R.append(rk['y']/tl)
        if la: ankle_y_L.append(la['y']/tl)
        if ra: ankle_y_R.append(ra['y']/tl)
        ls=get_kp(fd,'L_Shoulder'); rs=get_kp(fd,'R_Shoulder')
        if lh and lk and la: knee_ang_L.append(compute_angle([lh['x'],lh['y']],[lk['x'],lk['y']],[la['x'],la['y']]))
        if rh and rk and ra: knee_ang_R.append(compute_angle([rh['x'],rh['y']],[rk['x'],rk['y']],[ra['x'],ra['y']]))
        if ls and lh and lk: hip_ang_L.append(compute_angle([ls['x'],ls['y']],[lh['x'],lh['y']],[lk['x'],lk['y']]))
        if rs and rh and rk: hip_ang_R.append(compute_angle([rs['x'],rs['y']],[rh['x'],rh['y']],[rk['x'],rk['y']]))
        lw=get_kp(fd,'L_Wrist'); rw=get_kp(fd,'R_Wrist')
        if lw: wrist_y_L.append(lw['y']/tl)
        if rw: wrist_y_R.append(rw['y']/tl)

    if n_valid<MIN_FRAMES: return None
    rec={'n_valid_frames':n_valid,'n_total_frames':len(frame_indices),
         'pct_valid':n_valid/len(frame_indices),'duration_sec':len(frame_indices)/fps}

    hip_xy_arr=np.array(hip_xy)
    if len(hip_xy_arr)>1:
        disp=np.linalg.norm(np.diff(hip_xy_arr,axis=0),axis=1)*fps
        rec['hip_speed_mean']=float(np.mean(disp))
        rec['hip_speed_std'] =float(np.std(disp))
        rec['hip_speed_cv']  =float(np.std(disp)/np.mean(disp)) if np.mean(disp)>1e-8 else np.nan
    else:
        rec['hip_speed_mean']=rec['hip_speed_std']=rec['hip_speed_cv']=np.nan

    rec['hip_jerk_cost']=jerk_cost(hip_y_arr,fps)
    rec['knee_jerk_L']  =jerk_cost(knee_y_L,fps) if len(knee_y_L)>=6 else np.nan
    rec['knee_jerk_R']  =jerk_cost(knee_y_R,fps) if len(knee_y_R)>=6 else np.nan

    def bilateral_corr(a,b):
        if len(a)<4 or len(b)<4: return np.nan
        mn=min(len(a),len(b))
        return float(np.corrcoef(a[:mn],b[:mn])[0,1])

    def phase_lag(a,b):
        if len(a)<4 or len(b)<4: return np.nan
        mn=min(len(a),len(b))
        aa=np.array(a[:mn])-np.mean(a[:mn]); bb=np.array(b[:mn])-np.mean(b[:mn])
        xcorr=np.correlate(aa,bb,mode='full'); lags=np.arange(-(mn-1),mn)
        return float(abs(lags[np.argmax(xcorr)])/fps)

    rec['hip_bilateral_corr']   =bilateral_corr(lh_y,rh_y)
    rec['ankle_bilateral_corr'] =bilateral_corr(ankle_y_L,ankle_y_R)
    rec['knee_bilateral_corr']  =bilateral_corr(knee_y_L,knee_y_R)
    rec['bilateral_phase_lag']  =phase_lag(lh_y,rh_y)
    rec['knee_dom_freq_L']      =dominant_freq(knee_y_L) if len(knee_y_L)>=16 else np.nan
    rec['knee_dom_freq_R']      =dominant_freq(knee_y_R) if len(knee_y_R)>=16 else np.nan
    rec['knee_ac_L']            =ac_strength(knee_y_L)   if len(knee_y_L)>=4  else np.nan
    rec['knee_ac_R']            =ac_strength(knee_y_R)   if len(knee_y_R)>=4  else np.nan
    rec['knee_spectral_entropy_L']=spectral_entropy(knee_y_L) if len(knee_y_L)>=16 else np.nan
    rec['knee_spectral_entropy_R']=spectral_entropy(knee_y_R) if len(knee_y_R)>=16 else np.nan
    rec['hip_dom_freq']         =dominant_freq(hip_y_arr) if len(hip_y_arr)>=16 else np.nan

    for name,arr in [('knee_angle_L',knee_ang_L),('knee_angle_R',knee_ang_R),
                     ('hip_angle_L',hip_ang_L),('hip_angle_R',hip_ang_R)]:
        a=np.array(arr)
        if len(a)>=3:
            rec[f'{name}_mean']=float(np.mean(a))
            rec[f'{name}_std'] =float(np.std(a))
            rec[f'{name}_cv']  =float(np.std(a)/np.mean(a)) if np.mean(a)>1e-8 else np.nan
        else:
            rec[f'{name}_mean']=rec[f'{name}_std']=rec[f'{name}_cv']=np.nan

    if len(ankle_y_L)>=3 and len(ankle_y_R)>=3:
        mn=min(len(ankle_y_L),len(ankle_y_R))
        sep=np.abs(np.array(ankle_y_L[:mn])-np.array(ankle_y_R[:mn]))
        rec['step_len_proxy']=float(np.mean(sep)); rec['ankle_y_asym']=float(np.std(sep))
    else: rec['step_len_proxy']=rec['ankle_y_asym']=np.nan

    rec['hip_sway_lateral_std']=float(np.std(hip_x_arr)) if len(hip_x_arr)>=3 else np.nan
    rec['hip_vert_std']        =float(np.std(hip_y_arr)) if len(hip_y_arr)>=3 else np.nan

    if len(ankle_y_L)>=3 and len(knee_y_L)>=3:
        mn=min(len(ankle_y_L),len(knee_y_L))
        rec['toe_walk_proxy_L']=float(np.mean(np.array(ankle_y_L[:mn])-np.array(knee_y_L[:mn])))
    else: rec['toe_walk_proxy_L']=np.nan
    if len(ankle_y_R)>=3 and len(knee_y_R)>=3:
        mn=min(len(ankle_y_R),len(knee_y_R))
        rec['toe_walk_proxy_R']=float(np.mean(np.array(ankle_y_R[:mn])-np.array(knee_y_R[:mn])))
    else: rec['toe_walk_proxy_R']=np.nan

    rec['arm_swing_L']=float(np.ptp(wrist_y_L)) if len(wrist_y_L)>=3 else np.nan
    rec['arm_swing_R']=float(np.ptp(wrist_y_R)) if len(wrist_y_R)>=3 else np.nan
    return rec

# ── Extraction loop ──────────────────────────────────────────────
all_features=[]; n_ok=n_no_loco=n_short=n_no_kp=n_err=0

for _,row in df.iterrows():
    try:
        ldf=pd.read_csv(row['label_path'])
        if 'Locomotion' not in ldf.columns: n_no_loco+=1; continue
        segs=get_contiguous_segments(ldf,'Locomotion',LOCO_LABELS)
        if not segs: n_no_loco+=1; continue
        with open(row['hrnet_full_path'],'r') as f: pose_data=json.load(f)
        pose_frames=pose_data.get('frames',{})
        fps_vid=float(pose_data.get('ann_fps',FPS))
        for si,(s,e) in enumerate(segs):
            frame_idx=list(range(s,e+1))
            if len(frame_idx)<MIN_FRAMES: n_short+=1; continue
            seg_frames=ldf[(ldf['Frame']>=s)&(ldf['Frame']<=e)]
            loco_type=seg_frames['Locomotion'].mode()[0] if len(seg_frames)>0 else 'Unknown'
            feats=extract_locomotion_features(pose_frames,frame_idx,fps_vid)
            if feats is None: n_no_kp+=1; continue
            feats.update({'pid':row['pid'],'Group':row['Group'],
                          'age_mo':row['age_mo'],'age_band':row['age_band'],
                          'loco_type':loco_type,'seg_idx':si,
                          'video':os.path.basename(str(row['video_path']))})
            all_features.append(feats); n_ok+=1
    except Exception: n_err+=1; continue

print(f"\nExtraction: OK={n_ok}  NoLoco={n_no_loco}  Short={n_short}  NoKP={n_no_kp}  Err={n_err}")
if n_ok==0:
    print("ERROR: No features extracted."); import sys; sys.exit(1)

feat_df=pd.DataFrame(all_features)
feat_df.to_csv(os.path.join(OUTPUT_DIR,'locomotion_features_clip.csv'),index=False)

META_COLS={'pid','Group','age_mo','age_band','loco_type','seg_idx','video',
           'n_valid_frames','n_total_frames','pct_valid','duration_sec'}
FEAT_COLS=[c for c in feat_df.columns if c not in META_COLS]

def make_child_df(clip_df):
    fc=[f for f in FEAT_COLS if f in clip_df.columns]
    agg=clip_df.groupby(['pid','Group'])[fc].mean().reset_index()
    agg['n_clips'] =clip_df.groupby(['pid','Group']).size().values
    agg['age_mo']  =clip_df.groupby(['pid','Group'])['age_mo'].first().values
    agg['age_band']=clip_df.groupby(['pid','Group'])['age_band'].first().values
    agg['loco_type']=(clip_df.groupby(['pid','Group'])['loco_type']
                      .agg(lambda x: x.mode()[0]).values)
    return agg

# Build stream dataframes
stream_clip_dfs={}; stream_child_dfs={}
for sk in AGE_STREAMS:
    sdf=stream_filter(feat_df,sk)
    stream_clip_dfs[sk]=sdf
    stream_child_dfs[sk]=make_child_df(sdf)
    sdf.to_csv(os.path.join(OUTPUT_DIR,f'clip_features_{sk}.csv'),index=False)
    stream_child_dfs[sk].to_csv(os.path.join(OUTPUT_DIR,f'child_features_{sk}.csv'),index=False)
    cdf=stream_child_dfs[sk]
    print(f"Stream {sk}: {len(sdf)} clips | {len(cdf)} children | "
          f"ASD={cdf[cdf['Group']=='ASD']['pid'].nunique()} "
          f"Non-ASD={cdf[cdf['Group']=='Non-ASD']['pid'].nunique()}")

print(feat_df.groupby(['loco_type','Group']).size().reset_index(name='n').to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# PART 2: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
hr("PART 2: STATISTICAL ANALYSIS")

# ── Step 0: ICC ──────────────────────────────────────────────────
def compute_icc(clip_df, feat_cols):
    records=[]
    for feat in feat_cols:
        sub=clip_df[['pid',feat]].dropna()
        if len(sub)<10: continue
        groups=[g[feat].values for _,g in sub.groupby('pid') if len(g)>=2]
        if len(groups)<5: continue
        f_stat,p_anova=stats.f_oneway(*groups)
        n_total=sum(len(g) for g in groups); k=len(groups)
        n0=(n_total-sum(len(g)**2/n_total for g in groups))/(k-1)
        grand=np.concatenate(groups)
        ms_between=np.sum([len(g)*(np.mean(g)-np.mean(grand))**2 for g in groups])/(k-1)
        ms_within =np.sum([np.sum((g-np.mean(g))**2) for g in groups])/(n_total-k)
        icc=max(0.0,(ms_between-ms_within)/(ms_between+(n0-1)*ms_within))
        records.append({'feature':feat,'ICC':round(icc,4),'f_stat':round(f_stat,3),'p_anova':round(p_anova,4)})
    return pd.DataFrame(records).sort_values('ICC',ascending=False)

print("\n--- Step 0: ICC ---")
icc_df=compute_icc(feat_df,FEAT_COLS)
icc_df.to_csv(os.path.join(OUTPUT_DIR,'stats_icc.csv'),index=False)
print(icc_df.head(10).to_string(index=False))
print(f"\n  ICC>0.10: {(icc_df['ICC']>0.10).sum()}/{len(icc_df)} features")

# ── Helper: dummy-code loco type ────────────────────────────────
def _add_loco_dummies(df, reference=LOCO_REFERENCE):
    df=df.copy()
    types=sorted(df['loco_type'].dropna().unique())
    non_ref=[t for t in types if t!=reference]
    for t in non_ref:
        col='loco_'+re.sub(r'[^A-Za-z0-9]','_',t)
        df[col]=(df['loco_type']==t).astype(float)
    dummy_cols=['loco_'+re.sub(r'[^A-Za-z0-9]','_',t) for t in non_ref]
    return df, dummy_cols

# ── Step 1: LME with Kenward-Roger (rpy2) or statsmodels fallback ─
def run_lme_kr(clip_df, feat_cols, subset_label='ALL'):
    """
    Primary LME.  Uses Kenward-Roger df via lmerTest (rpy2) when available.
    Falls back to statsmodels REML (no KR) with a clear warning.
    Behavior (loco_type) is dummy-coded, not ordinal.
    """
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    df_use,dummy_cols=_add_loco_dummies(df_use)

    for feat in feat_cols:
        keep=['pid','Group_bin','age_mo_c','loco_type',feat]+dummy_cols
        sub=df_use[[c for c in keep if c in df_use.columns]].dropna(subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique()<2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min()<3: continue
        av=sub[sub['Group_bin']==1][feat].values
        nv=sub[sub['Group_bin']==0][feat].values
        d=cohen_d(av,nv)
        ci_lo,ci_hi=bootstrap_ci_d(av,nv,n_boot=500)

        p_val=np.nan; coef=np.nan; se=np.nan; method_used='none'; converged=False

        if _RPY2_OK:
            try:
                safe=re.sub(r'[^A-Za-z0-9_]','_',feat)
                sub2=sub.rename(columns={feat:safe})
                bterm=' + '.join(dummy_cols) if dummy_cols else ''
                formula=(f'{safe} ~ Group_bin + age_mo_c'
                         +(f' + {bterm}' if bterm else '')
                         +' + (1|pid)')
                r_df=pandas2ri.py2rpy(sub2)
                ro.globalenv['r_df']=r_df
                ro.r(f'fit <- lmerTest::lmer({formula}, data=r_df, REML=TRUE)')
                summ=ro.r('as.data.frame(coef(summary(fit, ddf="Kenward-Roger")))')
                summ_pd=pandas2ri.rpy2py(summ)
                if 'Group_bin' in summ_pd.index:
                    coef =float(summ_pd.loc['Group_bin','Estimate'])
                    se   =float(summ_pd.loc['Group_bin','Std. Error'])
                    p_val=float(summ_pd.loc['Group_bin','Pr(>|t|)'])
                    method_used='LME_KR'; converged=True
            except Exception as e:
                pass   # fall through to statsmodels

        if method_used=='none':
            try:
                bterm='+'.join(dummy_cols) if dummy_cols else ''
                formula_sm=(f'{feat} ~ Group_bin + age_mo_c'
                            +(f' + {bterm}' if bterm else ''))
                mdf=smf.mixedlm(formula_sm,sub,groups=sub['pid']).fit(
                    method=['lbfgs'],reml=True,maxiter=300)
                coef =float(mdf.params.get('Group_bin',np.nan))
                se   =float(mdf.bse.get('Group_bin',np.nan))
                p_val=float(mdf.pvalues.get('Group_bin',np.nan))
                method_used='LME_noKR'; converged=bool(mdf.converged)
            except: pass

        records.append({'feature':feat,'subset':subset_label,'method':method_used,
                        'coef_ASD':coef,'se':se,'p_raw':p_val,
                        'cohens_d':d,'d_ci_lo':ci_lo,'d_ci_hi':ci_hi,
                        'converged':converged,
                        'n_asd':sub[sub['Group_bin']==1]['pid'].nunique(),
                        'n_nasd':sub[sub['Group_bin']==0]['pid'].nunique(),
                        'n_clips':len(sub)})
    if not records: return pd.DataFrame()
    res=pd.DataFrame(records)
    return fdr_annotate(res,'p_raw').sort_values('p_raw')

# ── Step 2: CR2 (bias-reduced linearization) ─────────────────────
def run_cr2(clip_df, feat_cols, subset_label='ALL'):
    """
    CR2 cluster-robust standard errors via wildboottest package.
    Falls back gracefully if package unavailable.
    """
    if not _WBT_OK:
        print("  [CR2] wildboottest not available — skipping")
        return pd.DataFrame()
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    df_use,dummy_cols=_add_loco_dummies(df_use)
    for feat in feat_cols:
        sub=df_use[['pid','Group_bin','age_mo_c',feat]+dummy_cols].dropna(subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique()<2 or len(sub)<10: continue
        X_cols=['Group_bin','age_mo_c']+[c for c in dummy_cols if sub[c].std()>1e-8]
        X=sub[X_cols].values.astype(float)
        y=sub[feat].values.astype(float)
        clusters=sub['pid'].values
        try:
            wbt=WildboottestHC(X=X,y=y,cluster=clusters,
                               R=np.eye(len(X_cols))[[0],:],
                               B=999,bootstrap_type='WCR11')
            wbt.get_wildboottest()
            p_cr2=float(wbt.pvalue)
            records.append({'feature':feat,'subset':subset_label,'method':'CR2',
                            'p_raw':p_cr2,'n_clips':len(sub)})
        except Exception: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

# ── Step 3: GEE (supplementary) ──────────────────────────────────
def run_gee(clip_df, feat_cols, subset_label='ALL'):
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    df_use,dummy_cols=_add_loco_dummies(df_use)
    pid_map={p:i for i,p in enumerate(df_use['pid'].unique())}
    df_use['pid_int']=df_use['pid'].map(pid_map)
    for feat in feat_cols:
        sub=df_use[['pid_int','Group_bin','age_mo_c',feat]+dummy_cols].dropna(subset=['pid_int','Group_bin',feat])
        if sub['Group_bin'].nunique()<2 or len(sub)<20: continue
        counts=sub.groupby('pid_int').size()
        sub=sub[sub['pid_int'].isin(counts[counts>=2].index)]
        if len(sub)<20: continue
        try:
            safe=re.sub(r'[^A-Za-z0-9_]','_',feat)
            sub2=sub.rename(columns={feat:safe})
            bterm='+'.join([c for c in dummy_cols if sub2[c].std()>1e-8])
            formula=(f'{safe} ~ Group_bin + age_mo_c'+(f' + {bterm}' if bterm else ''))
            res=GEE.from_formula(formula,'pid_int',data=sub2,
                                 family=Gaussian(),cov_struct=Exchangeable()).fit(maxiter=100)
            av=sub[sub['Group_bin']==1][feat].values
            nv=sub[sub['Group_bin']==0][feat].values
            records.append({'feature':feat,'subset':subset_label,'method':'GEE',
                            'coef_ASD':float(res.params.get('Group_bin',np.nan)),
                            'p_raw':float(res.pvalues.get('Group_bin',np.nan)),
                            'cohens_d':cohen_d(av,nv),'n_clips':len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

# ── Step 4: Child-level permutation on LME t-statistic ───────────
def run_child_permutation_lme(child_df, feat_cols, n_perm=5000, subset_label='ALL'):
    """
    Permutes GROUP LABELS at child level, recomputes mean-diff statistic.
    More conservative than clip-level permutation — correct for clustered data.
    """
    rng=np.random.default_rng(42); records=[]
    for feat in feat_cols:
        sub=child_df[['pid','Group',feat]].dropna()
        if sub['Group'].nunique()<2: continue
        av=sub[sub['Group']=='ASD'][feat].values
        nv=sub[sub['Group']=='Non-ASD'][feat].values
        if len(av)<3 or len(nv)<3: continue
        obs_stat=abs(np.mean(av)-np.mean(nv)); n_asd=len(av)
        vals_arr=sub.set_index('pid')[feat].values
        child_list=sub['pid'].unique(); n_total=len(child_list)
        perm_stats=np.zeros(n_perm)
        for i in range(n_perm):
            sl=rng.permutation(['ASD']*n_asd+['Non-ASD']*(n_total-n_asd))
            a_v=vals_arr[np.array(sl)=='ASD']; n_v=vals_arr[np.array(sl)=='Non-ASD']
            a_v=a_v[~np.isnan(a_v)]; n_v=n_v[~np.isnan(n_v)]
            perm_stats[i]=abs(np.mean(a_v)-np.mean(n_v)) if len(a_v)>0 and len(n_v)>0 else 0
        p_perm=max(float(np.mean(perm_stats>=obs_stat)),1.0/n_perm)
        d=cohen_d(av,nv); ci_lo,ci_hi=bootstrap_ci_d(av,nv,n_boot=500)
        records.append({'feature':feat,'subset':subset_label,'method':'ChildPerm',
                        'obs_stat':float(obs_stat),'p_raw':p_perm,
                        'cohens_d':d,'d_ci_lo':ci_lo,'d_ci_hi':ci_hi,
                        'n_asd':len(av),'n_nasd':len(nv)})
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')

# ── Step 5: Wild cluster bootstrap ───────────────────────────────
def run_wild_bootstrap(child_df, feat_cols, n_boot=5000, subset_label='ALL'):
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

# ── Step 6: Pseudo-bulk Mann-Whitney ─────────────────────────────
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

# ── Step 7: Consensus table ───────────────────────────────────────
def make_consensus(results_dict, feat_cols, threshold=0.05):
    rows=[]
    for feat in feat_cols:
        row={'feature':feat}; n_sig=0
        for mname,res_df in results_dict.items():
            if res_df is None or len(res_df)==0:
                row[f'p_{mname}']=np.nan; continue
            match=res_df[res_df['feature']==feat]
            if len(match)==0: row[f'p_{mname}']=np.nan
            else:
                p=match['p_raw'].values[0]; row[f'p_{mname}']=round(p,4)
                if p<threshold: n_sig+=1
        row['n_methods_sig']=n_sig; rows.append(row)
    cons=pd.DataFrame(rows)
    _lme_kr  = results_dict.get('LME_KR')
    _lme_nokr = results_dict.get('LME_noKR')
    lme_df = (_lme_kr if (_lme_kr is not None and not _lme_kr.empty)
              else (_lme_nokr if (_lme_nokr is not None and not _lme_nokr.empty)
                    else None))
    if lme_df is not None and len(lme_df) and 'cohens_d' in lme_df.columns:
        d_map=lme_df.set_index('feature')['cohens_d'].to_dict()
        cons['cohens_d_LME']=cons['feature'].map(d_map)
        if 'd_ci_lo' in lme_df.columns:
            cons['d_ci_lo']=cons['feature'].map(lme_df.set_index('feature')['d_ci_lo'].to_dict())
            cons['d_ci_hi']=cons['feature'].map(lme_df.set_index('feature')['d_ci_hi'].to_dict())
    return cons.sort_values('n_methods_sig',ascending=False)

# ── Step 8: Consistency gate across loco types ───────────────────
def run_consistency_gate(feat_df, feat_cols, sig_feats):
    """Check whether Group effect is consistent across loco types."""
    beh_mwu={}
    for lt in sorted(feat_df['loco_type'].dropna().unique()):
        sub=feat_df[feat_df['loco_type']==lt]
        asd_n=sub[sub['Group']=='ASD']['pid'].nunique()
        nan_n=sub[sub['Group']=='Non-ASD']['pid'].nunique()
        if asd_n<3 or nan_n<3: continue
        recs=[]
        for feat in feat_cols:
            av=sub[sub['Group']=='ASD'][feat].dropna().values
            nv=sub[sub['Group']=='Non-ASD'][feat].dropna().values
            if len(av)<3 or len(nv)<3: continue
            _,p=stats.mannwhitneyu(av,nv,alternative='two-sided')
            recs.append({'feature':feat,'cohens_d':cohen_d(av,nv),'p_raw':p,'loco_type':lt})
        if recs: beh_mwu[lt]=pd.DataFrame(recs)

    cons_recs=[]; consistent_feats=[]
    beh_all=pd.concat(beh_mwu.values(),ignore_index=True) if beh_mwu else pd.DataFrame()
    for feat in sig_feats:
        if len(beh_all)==0: break
        sub=beh_all[beh_all['feature']==feat]
        if len(sub)<2: continue
        signs=np.sign(sub['cohens_d'].values)
        n_same=int((signs==signs[0]).sum()); passed=(n_same==len(sub))
        cons_recs.append({'feature':feat,'n_loco_tested':len(sub),
                          'n_same_direction':n_same,'consistent':passed})
        if passed: consistent_feats.append(feat)
    return pd.DataFrame(cons_recs), consistent_feats, beh_mwu

# ── RUN ALL STATS ────────────────────────────────────────────────
print("\n--- Running full statistical battery (full stream) ---")
cdf_full=stream_child_dfs['full']
sdf_full=stream_clip_dfs['full']

lme_all   = run_lme_kr(sdf_full, FEAT_COLS, 'full')
cr2_all   = run_cr2(sdf_full, FEAT_COLS, 'full')
gee_all   = run_gee(sdf_full, FEAT_COLS, 'full')
perm_all  = run_child_permutation_lme(cdf_full, FEAT_COLS, n_perm=5000, subset_label='full')
boot_all  = run_wild_bootstrap(cdf_full, FEAT_COLS, n_boot=5000, subset_label='full')
mw_all    = run_pseudobulk_mw(cdf_full, FEAT_COLS, 'full')

for name,res in [('LME',lme_all),('CR2',cr2_all),('GEE',gee_all),
                  ('Perm',perm_all),('WildBoot',boot_all),('MWU',mw_all)]:
    if len(res):
        print(f"  {name}: sig_raw={res['sig_raw05'].sum()}  FDR={res['sig_fdr05'].sum()}")
        res.to_csv(os.path.join(OUTPUT_DIR,f'stats_{name.lower()}_full.csv'),index=False)

all_results={'LME_KR':lme_all,'CR2':cr2_all,'GEE':gee_all,
             'ChildPerm':perm_all,'WildBoot':boot_all,'PseudobulkMW':mw_all}
consensus_all=make_consensus(all_results,FEAT_COLS)
consensus_all.to_csv(os.path.join(OUTPUT_DIR,'stats_consensus_all.csv'),index=False)
robust_all=consensus_all[consensus_all['n_methods_sig']>=2]
print(f"\n  Robust features (sig in 2+ methods): {len(robust_all)}")

# Consistency gate
sig_feats=list(perm_all[perm_all['sig_raw05']]['feature']) if len(perm_all) else []
cons_df,consistent_feats,loco_mwu=run_consistency_gate(feat_df,FEAT_COLS,sig_feats)
if len(cons_df):
    cons_df.to_csv(os.path.join(OUTPUT_DIR,'stats_consistency_gate.csv'),index=False)
    print(f"\n  Consistency gate: {len(consistent_feats)}/{len(sig_feats)} features passed")

# Age-stratified
print("\n--- Age-stratified analysis ---")
all_band_results=[]
for sk in ['11-18mo','32-38mo']:
    sub_clip=stream_clip_dfs[sk]; sub_child=stream_child_dfs[sk]
    n_asd=sub_clip[sub_clip['Group']=='ASD']['pid'].nunique()
    n_nasd=sub_clip[sub_clip['Group']=='Non-ASD']['pid'].nunique()
    print(f"\n  [{sk}] ASD={n_asd} Non-ASD={n_nasd} {len(sub_clip)} clips")
    if n_asd<3 or n_nasd<3: print("    → skip"); continue
    band_lme =run_lme_kr(sub_clip,FEAT_COLS,sk)
    band_perm=run_child_permutation_lme(sub_child,FEAT_COLS,n_perm=2000,subset_label=sk)
    band_boot=run_wild_bootstrap(sub_child,FEAT_COLS,n_boot=2000,subset_label=sk)
    band_mw  =run_pseudobulk_mw(sub_child,FEAT_COLS,sk)
    band_dict={k:v for k,v in {'LME_KR':band_lme,'ChildPerm':band_perm,
                                'WildBoot':band_boot,'PseudobulkMW':band_mw}.items()
               if v is not None and len(v)>0}
    if not band_dict: continue
    band_cons=make_consensus(band_dict,FEAT_COLS); band_cons['age_band']=sk
    all_band_results.append(band_cons)
    band_cons.to_csv(os.path.join(OUTPUT_DIR,f'stats_{sk.replace("-","_")}_consensus.csv'),index=False)
    top3=band_cons[band_cons['n_methods_sig']>0].head(3)
    for _,r in top3.iterrows():
        print(f"    {r['feature']:<35} n_sig={r['n_methods_sig']}")

# ═══════════════════════════════════════════════════════════════════
# PART 3: BAYESIAN HIERARCHICAL LMM
# ═══════════════════════════════════════════════════════════════════
hr("PART 3: BAYESIAN HIERARCHICAL LMM")

bayes_main_results={}

if not RUN_BAYESIAN or not _PYMC_OK:
    print("  Bayesian skipped (RUN_BAYESIAN=False or PyMC not available)")
else:
    def _standardise(series):
        m,s=series.mean(),series.std()
        s=s if s>1e-10 else 1.0
        return ((series-m)/s).values,m,s

    def _savage_dickey_bf(post, prior_sd=0.5):
        prior_at_0=spnorm.pdf(0,0,prior_sd)
        try:
            post_at_0=gaussian_kde(post)(0)[0]
            return float(prior_at_0/post_at_0) if post_at_0>0 else np.nan
        except: return np.nan

    def _build_bayes_df(df, feat, reference=LOCO_REFERENCE):
        tmp=df[['pid','Group','age_mo','loco_type',feat]].dropna().copy()
        if len(tmp)<8 or tmp['pid'].nunique()<4: return None
        tmp['Group_bin']=(tmp['Group']=='ASD').astype(float)
        tmp['age_c']=tmp['age_mo']-tmp['age_mo'].mean()
        types=sorted(tmp['loco_type'].dropna().unique())
        non_ref=[t for t in types if t!=reference]
        beh_mat=np.column_stack(
            [(tmp['loco_type']==t).astype(float).values for t in non_ref]
        ) if non_ref else np.zeros((len(tmp),0))
        y_z,ym,ys=_standardise(tmp[feat])
        pids,pid_idx=np.unique(tmp['pid'].values,return_inverse=True)
        return {'df':tmp,'y_z':y_z.astype(float),'group_bin':tmp['Group_bin'].values.astype(float),
                'age_c':tmp['age_c'].values.astype(float),'beh_mat':beh_mat,
                'n_beh_dum':beh_mat.shape[1],'pid_idx':pid_idx,'n_pids':len(pids),
                'pid_labels':pids,'y_mean':ym,'y_std':ys,'n_obs':len(tmp)}

    def _fit_bayes_main(bd, prior_sd=0.5, draws=BAYES_DRAWS, tune=BAYES_TUNE,
                        chains=BAYES_CHAINS, seed=42):
        with pm.Model():
            alpha    =pm.Normal('alpha',0,1)
            b_group  =pm.Normal('b_group',0,prior_sd)
            b_age    =pm.Normal('b_age',0,0.5)
            beh_contrib=0.0
            if bd['n_beh_dum']>0:
                b_beh=pm.Normal('b_beh',0,0.5,shape=bd['n_beh_dum'])
                beh_contrib=pm.math.dot(bd['beh_mat'],b_beh)
            sigma_pid=pm.HalfNormal('sigma_pid',1)
            sigma    =pm.HalfNormal('sigma',1)
            alpha_pid=pm.Normal('alpha_pid',0,sigma_pid,shape=bd['n_pids'])
            mu=(alpha+alpha_pid[bd['pid_idx']]+beh_contrib
                +b_group*bd['group_bin']+b_age*bd['age_c'])
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
                      'converged':bool(rhat<1.05 and ess>400 and n_div==0),
                      'prior_sd':prior_sd}

    def prior_predictive_check(bd, feat, prior_sd=0.5):
        """Sample from prior and verify implied feature ranges are plausible."""
        with pm.Model():
            b_group=pm.Normal('b_group',0,prior_sd)
            b_age  =pm.Normal('b_age',0,0.5)
            sigma  =pm.HalfNormal('sigma',1)
            alpha  =pm.Normal('alpha',0,1)
            mu=alpha+b_group*bd['group_bin']+b_age*bd['age_c']
            pm.Normal('y_obs',mu=mu,sigma=sigma,observed=bd['y_z'])
            ppc=pm.sample_prior_predictive(samples=200,random_seed=42)
        prior_ys=ppc.prior_predictive['y_obs'].values.flatten()
        obs_range=(bd['y_z'].min(),bd['y_z'].max())
        prior_range=(float(np.percentile(prior_ys,1)),float(np.percentile(prior_ys,99)))
        return {'feature':feat,'obs_min':obs_range[0],'obs_max':obs_range[1],
                'prior_p1':prior_range[0],'prior_p99':prior_range[1],
                'plausible':prior_range[0]<=obs_range[0] and prior_range[1]>=obs_range[1]}

    # Prior sensitivity analysis: three prior widths
    PRIOR_SDS=[0.3, 0.5, 1.0]

    # Select top features by permutation p-value for Bayesian analysis
    bayes_feats=(perm_all.sort_values('p_raw').head(15)['feature'].tolist()
                 if len(perm_all) else FEAT_COLS[:10])

    print(f"\nRunning Bayesian models on {len(bayes_feats)} features...")
    ppc_records=[]; bayes_records=[]; sensitivity_records=[]

    for feat in bayes_feats:
        bd=_build_bayes_df(sdf_full,feat)
        if bd is None: continue

        # Prior predictive check
        try:
            ppc_rec=prior_predictive_check(bd,feat)
            ppc_records.append(ppc_rec)
            if not ppc_rec['plausible']:
                print(f"  ⚠ PPC: prior too narrow for {feat}")
        except: pass

        # Sensitivity: three prior widths
        bf_vals={}
        for psd in PRIOR_SDS:
            try:
                _,summ=_fit_bayes_main(bd,prior_sd=psd)
                summ['feature']=feat; summ['prior_sd']=psd
                sensitivity_records.append(summ)
                bf_vals[psd]=summ['bf10']
            except Exception as e:
                print(f"  [{feat}] prior={psd} failed: {e}")

        # Main result = prior_sd=0.5
        if 0.5 in bf_vals:
            match=[r for r in sensitivity_records if r['feature']==feat and r['prior_sd']==0.5]
            if match:
                rec=match[-1].copy()
                # Robustness: BF consistent across prior widths?
                bfs=[bf_vals[p] for p in PRIOR_SDS if p in bf_vals and not np.isnan(bf_vals[p])]
                rec['bf_robust']=bool(len(bfs)>=2 and
                                      all((b>1)==(bfs[0]>1) for b in bfs))
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
        bayes_df.to_csv(os.path.join(OUTPUT_DIR,'bayes_main_full.csv'),index=False)
        bayes_main_results['full']=bayes_df
        print(f"\n  BF10>3:  {(bayes_df['bf10']>3).sum()}/{len(bayes_df)}")
        print(f"  BF10>10: {(bayes_df['bf10']>10).sum()}/{len(bayes_df)}")
        print(f"  BF robust across priors: {bayes_df['bf_robust'].sum()}/{len(bayes_df)}")
        conv_issues=bayes_df[~bayes_df['converged']]
        if len(conv_issues):
            print(f"  ⚠ Convergence issues ({len(conv_issues)} models):")
            for _,r in conv_issues.iterrows():
                print(f"    {r['feature']}  rhat={r.get('rhat',np.nan):.3f}  ess={r.get('ess_bulk',np.nan):.0f}  div={r.get('n_divergences',0)}")

# ═══════════════════════════════════════════════════════════════════
# PART 4: CLASSIFICATION (LOSO, child-level primary)
# ═══════════════════════════════════════════════════════════════════
hr("PART 4: CLASSIFICATION — CHILD-LEVEL LOSO")

def run_loso_child(cdf, feat_cols, clf_name='LR', n_perm=500, seed=42):
    df_=cdf.copy()
    df_['y']=(df_['Group']=='ASD').astype(int)
    if df_['y'].sum()<4 or (1-df_['y']).sum()<4: return None
    usable=[f for f in feat_cols if f in df_.columns and df_[f].notna().mean()>0.5]
    if len(usable)<2: return None
    df_[usable]=df_[usable].fillna(df_[usable].median())
    if clf_name=='LR':
        clf=LogisticRegression(max_iter=1000,C=0.1,class_weight='balanced',random_state=seed)
    else:
        clf=RandomForestClassifier(n_estimators=200,class_weight='balanced',
                                   random_state=seed,n_jobs=-1)
    pipe=Pipeline([('sc',StandardScaler()),('clf',clf)])
    y_true,y_score=[],[]
    for pid in df_['pid'].unique():
        test=df_[df_['pid']==pid]; train=df_[df_['pid']!=pid]
        if len(train['y'].unique())<2: continue
        try:
            pipe.fit(train[usable].values,train['y'].values)
            y_score.extend(pipe.predict_proba(test[usable].values)[:,1].tolist())
            y_true.extend(test['y'].values.tolist())
        except: continue
    if len(set(y_true))<2: return None
    auc=roc_auc_score(y_true,y_score); ap=average_precision_score(y_true,y_score)
    rng=np.random.default_rng(seed)
    perm=[roc_auc_score(rng.permuted(np.array(y_true)),y_score) for _ in range(n_perm)]
    p_perm=float((np.array(perm)>=auc).mean())
    cm=confusion_matrix(y_true,(np.array(y_score)>=0.5).astype(int))
    print(f"  [{clf_name}] AUC={auc:.3f}  AP={ap:.3f}  p_perm={p_perm:.4f}  n_feat={len(usable)}")
    return {'auc':auc,'ap':ap,'perm_p':p_perm,'n_features':len(usable),
            'n_subjects':df_['pid'].nunique(),'y_true':y_true,'y_score':y_score,
            'perm_aucs':perm,'confusion_matrix':cm,'clf':clf_name}

clf_results={}
for sk in AGE_STREAMS:
    cdf=stream_child_dfs[sk]
    fc=[f for f in FEAT_COLS if f in cdf.columns]
    asd_n=(cdf['Group']=='ASD').sum(); nan_n=(cdf['Group']=='Non-ASD').sum()
    print(f"\n--- Stream {sk} (ASD={asd_n}, Non-ASD={nan_n}) ---")
    if asd_n<4 or nan_n<4: print("  Skipped (n<4)"); continue
    for cname in ['LR','RF']:
        r=run_loso_child(cdf,fc,clf_name=cname)
        if r: clf_results[f'{sk}_{cname}']=r

# Per loco-type classification
for lt in sorted(feat_df['loco_type'].dropna().unique()):
    sub=stream_child_dfs['full'][stream_child_dfs['full']['loco_type']==lt]
    asd_n=(sub['Group']=='ASD').sum(); nan_n=(sub['Group']=='Non-ASD').sum()
    print(f"\n--- Loco type: {lt} (ASD={asd_n}, Non-ASD={nan_n}) ---")
    if asd_n>=4 and nan_n>=4:
        r=run_loso_child(sub,[f for f in FEAT_COLS if f in sub.columns],clf_name='LR')
        if r: clf_results[f'loco_{lt.lower()}_LR']=r

if clf_results:
    pd.DataFrame([{'subset':k,'clf':v.get('clf',''),'auc':v['auc'],'ap':v['ap'],
                   'perm_p':v['perm_p'],'n_features':v['n_features'],
                   'n_subjects':v['n_subjects']} for k,v in clf_results.items()
                  ]).to_csv(os.path.join(OUTPUT_DIR,'classification_summary.csv'),index=False)

# RF feature importances
feat_importance_df=pd.DataFrame()
try:
    tmp=stream_child_dfs['full'].copy()
    tmp['y']=(tmp['Group']=='ASD').astype(int)
    fc_full=[f for f in FEAT_COLS if f in tmp.columns]
    usable=[f for f in fc_full if tmp[f].notna().mean()>0.5]
    tmp[usable]=tmp[usable].fillna(tmp[usable].median())
    sc=StandardScaler(); X=sc.fit_transform(tmp[usable].values)
    rf=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1)
    rf.fit(X,tmp['y'].values)
    feat_importance_df=pd.DataFrame({'feature':usable,'importance':rf.feature_importances_}
                                    ).sort_values('importance',ascending=False)
    feat_importance_df.to_csv(os.path.join(OUTPUT_DIR,'rf_feature_importances.csv'),index=False)
    print("\nTop 10 RF importances:")
    for _,r in feat_importance_df.head(10).iterrows():
        print(f"  {r['feature']:<40} {r['importance']:.4f}")
except Exception as e:
    print(f"RF importance failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# PART 5: FIGURES
# ═══════════════════════════════════════════════════════════════════
hr("PART 5: FIGURES")

# Fig 1: Sample overview
print("Fig 1: Sample overview...")
fig,axes=plt.subplots(1,3,figsize=(16,5))
fig.suptitle('Locomotion Sample Overview',fontweight='bold')
ax=axes[0]
gc=stream_child_dfs['full']['Group'].value_counts()
bars=ax.bar(GROUPS,[gc.get(g,0) for g in GROUPS],color=[COLORS[g] for g in GROUPS],
            width=0.5,edgecolor='white')
for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                          str(int(bar.get_height())),ha='center',fontweight='bold')
ax.set_title('(a) Children (full stream)'); ax.set_ylabel('N')
ax=axes[1]
for grp in GROUPS:
    ax.hist(feat_df[feat_df['Group']==grp]['duration_sec'],
            bins=20,alpha=0.6,color=COLORS[grp],label=grp,edgecolor='white')
ax.set_title('(b) Segment durations'); ax.set_xlabel('sec'); ax.legend()
ax=axes[2]
loco_counts=feat_df.groupby(['loco_type','Group']).size().reset_index(name='n')
lts=sorted(feat_df['loco_type'].dropna().unique()); x=np.arange(len(lts)); w=0.35
for i,grp in enumerate(GROUPS):
    vals=[loco_counts[(loco_counts['loco_type']==lt)&(loco_counts['Group']==grp)]['n'].values[0]
          if len(loco_counts[(loco_counts['loco_type']==lt)&(loco_counts['Group']==grp)])>0 else 0
          for lt in lts]
    ax.bar(x+i*w,vals,w,label=grp,color=COLORS[grp],edgecolor='white')
ax.set_xticks(x+w/2); ax.set_xticklabels(lts,rotation=20,ha='right')
ax.set_title('(c) Clips per loco type'); ax.set_ylabel('N'); ax.legend()
plt.tight_layout(); savefig(fig,'fig1_sample_overview.png')

# Fig 2: Effect sizes with bootstrap CI (LME)
print("Fig 2: Effect sizes...")
if len(lme_all)>0:
    res_plot=lme_all.copy()
    res_plot['label']=res_plot['feature'].map(FEAT_LABELS).fillna(res_plot['feature'].str.replace('_',' '))
    res_plot=res_plot.sort_values('cohens_d')
    fig,ax=plt.subplots(figsize=(11,max(6,len(res_plot)*0.35)))
    colors_bar=[ASD_COLOR if d>0 else NONASD_COLOR for d in res_plot['cohens_d']]
    ax.barh(res_plot['label'],res_plot['cohens_d'],color=colors_bar,
            edgecolor='white',height=0.7,alpha=0.85)
    if 'd_ci_lo' in res_plot.columns and 'd_ci_hi' in res_plot.columns:
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
    method_label=lme_all['method'].mode()[0] if 'method' in lme_all.columns and len(lme_all) else 'LME'
    ax.set_xlabel("Cohen's d  (positive = ASD > Non-ASD)")
    ax.set_title(f"Effect Sizes — {method_label}\nBars=95% bootstrap CI on d  ★=FDR sig",fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR,label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR,label='Non-ASD higher')])
    plt.tight_layout(); savefig(fig,'fig2_effect_sizes.png')

# Fig 3: Violin plots
print("Fig 3: Violin plots...")
KEY_FEATS=[f for f in ['hip_speed_mean','hip_speed_cv','hip_jerk_cost',
    'knee_bilateral_corr','hip_bilateral_corr','knee_dom_freq_L','knee_ac_L',
    'knee_spectral_entropy_L','hip_sway_lateral_std','step_len_proxy',
    'arm_swing_L','toe_walk_proxy_L'] if f in feat_df.columns]
if KEY_FEATS:
    ncols=4; nrows=int(np.ceil(len(KEY_FEATS)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows))
    fig.suptitle('Locomotion Kinematics: ASD vs Non-ASD',fontweight='bold')
    axes=axes.flatten()
    for i,feat in enumerate(KEY_FEATS):
        ax=axes[i]
        dg=[feat_df[feat_df['Group']==g][feat].dropna().values for g in GROUPS]
        if any(len(d)==0 for d in dg): ax.set_visible(False); continue
        parts=ax.violinplot(dg,positions=[0,1],showmedians=True,showextrema=False)
        for j,pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(COLORS.values())[j]); pc.set_alpha(0.7)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for j,vals in enumerate(dg):
            ax.scatter(j+np.random.uniform(-0.07,0.07,len(vals)),vals,
                       color=list(COLORS.values())[j],alpha=0.2,s=6,zorder=3)
        p_txt=''; p_col='gray'
        if len(perm_all):
            row_s=perm_all[perm_all['feature']==feat]
            if len(row_s):
                p_r=row_s['p_raw'].values[0]; p_f=row_s['p_fdr'].values[0]
                d=row_s['cohens_d'].values[0]
                p_col='#cc0000' if p_f<0.05 else ('#ff8800' if p_r<0.05 else 'gray')
                p_txt=f'Perm p={p_r:.3f}|FDR={p_f:.3f}|d={d:.2f}'
                ymax=max(np.percentile(d_,95) for d_ in dg if len(d_))
                yr=ymax-min(np.percentile(d_,5) for d_ in dg if len(d_))
                add_sig_bar(ax,0,1,ymax+yr*0.05,p_r,h=yr*0.04)
        ax.text(0.5,0.97,p_txt,transform=ax.transAxes,ha='center',va='top',fontsize=7,color=p_col)
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS,fontsize=9)
        ax.set_title(FEAT_LABELS.get(feat,feat),fontsize=9,fontweight='bold')
    for j in range(len(KEY_FEATS),len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); savefig(fig,'fig3_violins.png')

# Fig 4: Consensus heatmap
print("Fig 4: Consensus heatmap...")
if len(consensus_all)>0:
    p_cols=[c for c in consensus_all.columns if c.startswith('p_')]
    heat_data=consensus_all.set_index('feature')[p_cols].head(20)
    heat_log=-np.log10(heat_data.clip(lower=1e-5,upper=1.0).astype(float))
    fig,ax=plt.subplots(figsize=(len(p_cols)*2+2,max(6,len(heat_data)*0.4)))
    im=ax.imshow(heat_log.values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=4)
    ax.set_xticks(range(len(p_cols)))
    ax.set_xticklabels([c.replace('p_','') for c in p_cols],rotation=30,ha='right')
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels([FEAT_LABELS.get(f,f) for f in heat_data.index],fontsize=9)
    for i in range(heat_log.shape[0]):
        for j in range(heat_log.shape[1]):
            raw_p=heat_data.values[i,j]
            ax.text(j,i,f'{raw_p:.3f}{"*" if raw_p<0.05 else ""}',
                    ha='center',va='center',fontsize=7)
    plt.colorbar(im,ax=ax,label='-log10(p)')
    ax.set_title('Consensus p-values across methods (top 20)',fontweight='bold')
    plt.tight_layout(); savefig(fig,'fig4_consensus_heatmap.png')

# Fig 5: Age streams comparison (forest plot)
print("Fig 5: Stream forest plot...")
stream_lme_results={}
for sk in AGE_STREAMS:
    r=run_lme_kr(stream_clip_dfs[sk],FEAT_COLS,sk)
    if len(r): stream_lme_results[sk]=r

all_stream_rows=[]
for sk,res in stream_lme_results.items():
    r=res.copy(); r['stream']=sk; all_stream_rows.append(r)
if all_stream_rows:
    combined=pd.concat(all_stream_rows,ignore_index=True)
    top_feats=(combined.groupby('feature')['cohens_d']
               .apply(lambda x:x.abs().max()).sort_values(ascending=False).head(15).index.tolist())
    sub_c=combined[combined['feature'].isin(top_feats)].copy()
    fig,ax=plt.subplots(figsize=(12,max(6,len(top_feats)*0.6)))
    y_pos={f:i for i,f in enumerate(top_feats)}
    offsets={'full':0.0,'11-18mo':0.22,'32-38mo':-0.22}
    for sk in AGE_STREAMS:
        sub=sub_c[sub_c['stream']==sk]
        for _,row in sub.iterrows():
            y=y_pos[row['feature']]+offsets[sk]
            ax.scatter(row['cohens_d'],y,color=STREAM_COLORS[sk],s=60,zorder=5,alpha=0.9)
            if 'd_ci_lo' in row.index and not np.isnan(row['d_ci_lo']):
                ax.plot([row['d_ci_lo'],row['d_ci_hi']],[y,y],
                        color=STREAM_COLORS[sk],lw=2,alpha=0.7)
            if row.get('sig_fdr05'):
                ax.scatter(row['cohens_d'],y,color=STREAM_COLORS[sk],s=120,marker='*',zorder=6)
    ax.axvline(0,color='black',lw=0.8)
    for t,ls_ in [(0.2,'--'),(0.5,'-.'),(0.8,':')]:
        ax.axvline(t,color='gray',lw=0.6,ls=ls_,alpha=0.4)
        ax.axvline(-t,color='gray',lw=0.6,ls=ls_,alpha=0.4)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([FEAT_LABELS.get(f,f) for f in top_feats],fontsize=8)
    ax.set_xlabel("Cohen's d  (positive=ASD>Non-ASD)")
    ax.set_title("Effect Sizes Across Age Streams\n★=FDR sig  Line=95% bootstrap CI on d",fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=STREAM_COLORS[sk],label=sk) for sk in AGE_STREAMS])
    plt.tight_layout(); savefig(fig,'fig5_stream_forest.png')

# Fig 6: Consistency gate
print("Fig 6: Consistency gate...")
if len(cons_df)>0:
    fig,ax=plt.subplots(figsize=(10,max(4,len(cons_df)*0.45)))
    cols_cg=[ASD_COLOR if v else NONASD_COLOR for v in cons_df['consistent']]
    ax.barh(cons_df['feature'].map(FEAT_LABELS).fillna(cons_df['feature']),
            cons_df['n_same_direction']/cons_df['n_loco_tested'],
            color=cols_cg,edgecolor='white',height=0.6)
    ax.axvline(1.0,color='green',lw=1.5,ls='--',label='All consistent')
    ax.axvline(0.5,color='orange',lw=1,ls=':',label='50%')
    ax.set_xlim(0,1.15); ax.set_xlabel('Fraction of loco types with same direction')
    ax.set_title('Consistency Gate — Effect Direction Across Loco Types\nRed=failed',fontweight='bold')
    ax.legend(); plt.tight_layout(); savefig(fig,'fig6_consistency_gate.png')

# Fig 7: Bayesian forest
print("Fig 7: Bayesian forest...")
if 'full' in bayes_main_results and len(bayes_main_results['full'])>0:
    bdf=bayes_main_results['full'].copy()
    bdf['label']=bdf['feature'].map(FEAT_LABELS).fillna(bdf['feature'])
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
    ax.set_title('Bayesian Hierarchical LMM — Locomotion\n⚠=convergence issue  [prior-sensitive]=BF changed across priors',
                 fontweight='bold')
    ax.legend(handles=[mpatches.Patch(color=ASD_COLOR,label='ASD higher'),
                       mpatches.Patch(color=NONASD_COLOR,label='Non-ASD higher')])
    plt.tight_layout(); savefig(fig,'fig7_bayes_forest.png')

# Fig 8: Prior sensitivity
print("Fig 8: Prior sensitivity...")
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
            ax.set_title(FEAT_LABELS.get(feat,feat)[:25],fontsize=9); ax.legend(fontsize=7)
        for j in range(len(feats_s),len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); savefig(fig,'fig8_prior_sensitivity.png')

# Fig 9: Developmental trajectories
print("Fig 9: Trajectories...")
TRAJ_FEATS=[f for f in ['hip_speed_mean','hip_jerk_cost','knee_bilateral_corr',
                          'knee_spectral_entropy_L'] if f in feat_df.columns]
if TRAJ_FEATS:
    fig,axes=plt.subplots(1,len(TRAJ_FEATS),figsize=(5.5*len(TRAJ_FEATS),5))
    if len(TRAJ_FEATS)==1: axes=[axes]
    fig.suptitle('Developmental Trajectories',fontweight='bold')
    for i,feat in enumerate(TRAJ_FEATS):
        ax=axes[i]
        for grp in GROUPS:
            sub=feat_df[feat_df['Group']==grp].dropna(subset=[feat,'age_mo'])
            ax.scatter(sub['age_mo'],sub[feat],color=COLORS[grp],alpha=0.25,s=12)
            if len(sub)>=5:
                m,b,r,p,_=stats.linregress(sub['age_mo'],sub[feat])
                xr=np.linspace(sub['age_mo'].min(),sub['age_mo'].max(),100)
                ax.plot(xr,m*xr+b,color=COLORS[grp],lw=2.5,
                        label=f'{grp} r={r:.2f} p={p:.3f}')
        for band,(lo,hi) in zip(KEY_BANDS,[(11,18),(32,38)]):
            ax.axvspan(lo,hi,alpha=0.08,color=BAND_COLORS[band],label=band)
        ax.set_xlabel('Age (months)'); ax.set_ylabel(FEAT_LABELS.get(feat,feat),fontsize=9)
        ax.set_title(FEAT_LABELS.get(feat,feat)); ax.legend(fontsize=7)
    plt.tight_layout(); savefig(fig,'fig9_trajectories.png')

# Fig 10: Child-level boxplots
print("Fig 10: Child boxplots...")
BOX_FEATS=[f for f in ['hip_speed_mean','hip_jerk_cost','knee_bilateral_corr','step_len_proxy']
           if f in stream_child_dfs['full'].columns]
if BOX_FEATS:
    fig,axes=plt.subplots(1,len(BOX_FEATS),figsize=(4.5*len(BOX_FEATS),5))
    if len(BOX_FEATS)==1: axes=[axes]
    fig.suptitle('Child-Level Averages (each dot = one child)',fontweight='bold')
    for i,feat in enumerate(BOX_FEATS):
        ax=axes[i]
        cdf=stream_child_dfs['full']
        for j,grp in enumerate(GROUPS):
            vals=cdf[cdf['Group']==grp][feat].dropna().values
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
                ymax=cdf[feat].dropna().max()
                add_sig_bar(ax,0,1,ymax*1.05,p_perm,h=ymax*0.04)
                ax.text(0.5,0.97,f'Perm p={p_perm:.3f}',transform=ax.transAxes,
                        ha='center',va='top',fontsize=8,color='gray')
        ax.set_xticks([0,1]); ax.set_xticklabels(GROUPS,fontsize=9)
        ax.set_title(FEAT_LABELS.get(feat,feat),fontsize=9)
    plt.tight_layout(); savefig(fig,'fig10_child_boxplots.png')

# Fig 11: Classification ROC
print("Fig 11: Classification ROC...")
if clf_results:
    keys=list(clf_results.keys()); n=len(keys)
    ncols=min(n,4); nrows=int(np.ceil(n/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4.5*nrows))
    if nrows*ncols==1: axes=np.array([[axes]])
    elif nrows==1: axes=axes.reshape(1,-1)
    fig.suptitle('Classification ROC — Child-Level LOSO (primary)',fontweight='bold')
    for i,key in enumerate(keys):
        r=clf_results[key]; ax=axes[i//ncols][i%ncols]
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

# Fig 12: RF importances
print("Fig 12: RF importances...")
if len(feat_importance_df)>0:
    top20=feat_importance_df.head(20)
    fig,ax=plt.subplots(figsize=(11,7))
    ax.barh(top20['feature'].map(FEAT_LABELS).fillna(top20['feature']),
            top20['importance'],color=ASD_COLOR,edgecolor='white',height=0.65,alpha=0.85)
    ax.set_xlabel('Mean decrease in impurity')
    ax.set_title('RF Feature Importances (full stream, child level)',fontweight='bold')
    plt.tight_layout(); savefig(fig,'fig12_rf_importances.png')

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
print("\n--- TOP ROBUST FEATURES (consensus, full stream) ---")
if len(consensus_all):
    p_cols=[c for c in consensus_all.columns if c.startswith('p_')]
    top5=consensus_all.head(5)[['feature','n_methods_sig']+
                               (['cohens_d_LME'] if 'cohens_d_LME' in consensus_all.columns else [])+p_cols]
    print(top5.to_string(index=False))
print(f"\nConsistency gate: {len(consistent_feats)} features passed")
for f in consistent_feats: print(f"  ✓ {f}")
if clf_results:
    print("\nClassification (child-level LOSO):")
    for k,v in clf_results.items():
        print(f"  {k:<40} AUC={v['auc']:.3f}  p_perm={v['perm_p']:.4f}")
hr("COMPLETE")