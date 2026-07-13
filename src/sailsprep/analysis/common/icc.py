"""Shared ICC helper (exact duplicate: crawling, running, walking)."""

import numpy as np
import pandas as pd
from scipy import stats


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
        ms_b=sum(len(g)*(np.mean(g)-np.mean(grand))**2 for g in groups)/(k-1)
        ms_w=sum(sum((g-np.mean(g))**2) for g in groups)/(n_total-k)
        icc=max(0.0,(ms_b-ms_w)/(ms_b+(n0-1)*ms_w))
        records.append({'feature':feat,'ICC':round(icc,4),'f_stat':round(f_stat,3),'p_anova':round(p_anova,4)})
    return pd.DataFrame(records).sort_values('ICC',ascending=False)
