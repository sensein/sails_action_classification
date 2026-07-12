"""Shared Bayesian-analysis helpers (exact duplicates across analysis modules)."""

import numpy as np
from scipy.stats import gaussian_kde, norm as spnorm


def _savage_dickey_bf(post, prior_sd=0.5):
    prior_at_0 = spnorm.pdf(0, 0, prior_sd)
    try:
        post_at_0 = gaussian_kde(post)(0)[0]
        return float(prior_at_0 / post_at_0) if post_at_0 > 0 else np.nan
    except: return np.nan


def _standardise(series):
    m, s = series.mean(), series.std()
    s = s if s > 1e-10 else 1.0
    return ((series - m) / s).values, m, s
