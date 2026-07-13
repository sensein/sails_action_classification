"""Shared keypoint helpers (exact duplicates across analysis modules)."""

import numpy as np

MIN_CONF = 0.3

def get_kp(fd, key, min_conf=MIN_CONF):
    if key not in fd: return None
    kp = fd[key]
    if not isinstance(kp, dict): return None
    if kp.get('confidence', 0) < min_conf: return None
    return kp


AGE_BANDS = {
    '11-18mo': (11, 18),
    '19-31mo': (19, 31),
    '32-38mo': (32, 38),
}


def assign_age_band(age_mo):
    for band, (lo, hi) in AGE_BANDS.items():
        if lo <= age_mo <= hi: return band
    return None



# KP / torso_length / get_scale below are shared only by crusing.py and
# walking.py, whose KP dicts and torso_length/get_scale bodies are verified
# byte-for-byte identical.
KP = {
    'nose':        'kp_000',
    'L_shoulder':  'kp_005', 'R_shoulder': 'kp_006',
    'L_elbow':     'kp_007', 'R_elbow':    'kp_008',
    'L_wrist':     'kp_009', 'R_wrist':    'kp_010',
    'L_hip':       'kp_011', 'R_hip':      'kp_012',
    'L_knee':      'kp_013', 'R_knee':     'kp_014',
    'L_ankle':     'kp_015', 'R_ankle':    'kp_016',
    'L_big_toe':   'kp_017', 'L_small_toe':'kp_018', 'L_heel': 'kp_019',
    'R_big_toe':   'kp_020', 'R_small_toe':'kp_021', 'R_heel': 'kp_022',
}


def torso_length(fd):
    ls = get_kp(fd, KP['L_shoulder'], 0.1); rs = get_kp(fd, KP['R_shoulder'], 0.1)
    lh = get_kp(fd, KP['L_hip'],      0.1); rh = get_kp(fd, KP['R_hip'],      0.1)
    if not all([ls, rs, lh, rh]): return None
    sx = (ls['x']+rs['x'])/2; sy = (ls['y']+rs['y'])/2
    hx = (lh['x']+rh['x'])/2; hy = (lh['y']+rh['y'])/2
    d  = np.sqrt((sx-hx)**2+(sy-hy)**2)
    return d if d > 5 else None


def get_scale(fd):
    tl = torso_length(fd)
    if tl: return tl
    lh = get_kp(fd, KP['L_hip'], 0.1); rh = get_kp(fd, KP['R_hip'], 0.1)
    if lh and rh:
        d = np.sqrt((lh['x']-rh['x'])**2+(lh['y']-rh['y'])**2)
        if d > 5: return d
    ls = get_kp(fd, KP['L_shoulder'], 0.1); rs = get_kp(fd, KP['R_shoulder'], 0.1)
    if ls and rs:
        d = np.sqrt((ls['x']-rs['x'])**2+(ls['y']-rs['y'])**2)
        if d > 5: return d
    return None
