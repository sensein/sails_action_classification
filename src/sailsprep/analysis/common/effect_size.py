"""Shared Cohen's d effect-size helpers (exact duplicates across analysis modules)."""

import numpy as np


def cohen_d_v1(a, b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    pooled=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return float((np.mean(a)-np.mean(b))/pooled) if pooled>1e-10 else 0.0


def cohen_d_v2(a, b):
    a, b   = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((np.var(a, ddof=1)+np.var(b, ddof=1))/2)
    return float((np.mean(a)-np.mean(b))/pooled) if pooled > 1e-10 else 0.0


def cohen_d_v3(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
