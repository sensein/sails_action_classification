"""Shared mixed-effects-model helper (exact duplicate: crawling, crusing, running, walking)."""

MIN_SESSIONS_FOR_SLOPE = 2


def _use_random_slope(df, pid_col='pid'):
    ns=df.groupby(pid_col)['session'].nunique()
    return float(ns.median())>=MIN_SESSIONS_FOR_SLOPE
