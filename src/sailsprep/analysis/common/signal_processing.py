"""Shared signal-processing helpers (exact duplicates across analysis modules)."""

import numpy as np
from scipy.signal import butter, filtfilt, welch


def butter_lp_v1(data, cutoff=6.0, fs=15.0, order=2):
    arr = np.array(data, dtype=float)
    if len(arr) < 10: return arr
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff, nyq*0.9)/nyq, btype='low')
    if len(arr) < 3*max(len(b), len(a)): return arr
    return filtfilt(b, a, arr)


def butter_lp_v2(data, cutoff=4.0, fs=15.0, order=2):
    arr = np.array(data, dtype=float)
    if len(arr) < 10: return arr
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff, nyq*0.9)/nyq, btype='low')
    if len(arr) < 3*max(len(b), len(a)): return arr
    return filtfilt(b, a, arr)


def compute_angle_2d_v1(p1,p2,p3):
    v1=np.array([p1[0]-p2[0],p1[1]-p2[1]]); v2=np.array([p3[0]-p2[0],p3[1]-p2[1]])
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-8 or n2<1e-8: return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))


def compute_angle_2d_v2(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))))


def sparc_smoothness_v1(vel, fps):
    if len(vel)<8: return np.nan
    try:
        fv,pv=welch(vel,fs=fps,nperseg=min(len(vel),32))
        pv_n=pv/(pv.max()+1e-12)
        return float(-np.sum(np.sqrt(np.diff(fv)**2+np.diff(pv_n)**2)))
    except: return np.nan


def sparc_smoothness_v2(vel, fps):
    if len(vel) < 8: return np.nan
    try:
        fv, pv = welch(vel, fs=fps, nperseg=min(len(vel), 32))
        pv_n   = pv/(pv.max()+1e-12)
        return float(-np.sum(np.sqrt(np.diff(fv)**2+np.diff(pv_n)**2)))
    except: return np.nan


def spectral_features(arr, fps, lo=0.5, hi=2.0):
    if len(arr) < 16: return np.nan, np.nan, np.nan
    try:
        freqs, psd = welch(arr, fs=fps, nperseg=min(len(arr), 64))
        dom_freq   = float(freqs[np.argmax(psd)])
        psd_n      = psd/(psd.sum()+1e-12)
        entropy    = float(-np.sum(psd_n[psd_n>0]*np.log2(psd_n[psd_n>0])))
        band_pwr   = float(psd[(freqs>=lo)&(freqs<=hi)].sum()/(psd.sum()+1e-12))
        return dom_freq, entropy, band_pwr
    except: return np.nan, np.nan, np.nan
