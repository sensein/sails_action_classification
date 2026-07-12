"""Shared Spearman age-correlation helper (exact duplicate: jumping, spinning)."""

import pandas as pd
from scipy import stats

GROUPS = ['ASD', 'Non-ASD']


def run_spearman_age(clip_df, feat_cols):
    sp_recs = []
    for grp in GROUPS:
        sub = clip_df[clip_df['Group'] == grp]
        for feat in feat_cols:
            vals = sub[['age_mo', feat]].dropna()
            if len(vals) < 5: continue
            r, p = stats.spearmanr(vals['age_mo'], vals[feat])
            sp_recs.append({'Group': grp, 'feature': feat,
                            'spearman_r': r, 'p_raw': p, 'n': len(vals)})
    if not sp_recs: return pd.DataFrame()
    sp_df = pd.DataFrame(sp_recs); sp_df['sig_p05'] = sp_df['p_raw'] < 0.05
    return sp_df
