"""Shared consensus helper (exact duplicate: crawling, running, walking)."""

import numpy as np
import pandas as pd


def make_consensus(results_dict, feat_cols, threshold=0.05):
    rows=[]
    for feat in feat_cols:
        row={'feature':feat}; n_sig=0
        for mname,res_df in results_dict.items():
            if res_df is None or len(res_df)==0: row[f'p_{mname}']=np.nan; continue
            match=res_df[res_df['feature']==feat]
            if len(match)==0: row[f'p_{mname}']=np.nan
            else:
                p=match['p_raw'].values[0]; row[f'p_{mname}']=round(p,4)
                if p<threshold: n_sig+=1
        row['n_methods_sig']=n_sig; rows.append(row)
    cons=pd.DataFrame(rows)
    lme_df = next(
        (results_dict.get(k) for k in ('LME_KR', 'LME_noKR', 'LME')
         if results_dict.get(k) is not None and not results_dict[k].empty),
        None
    )
    if lme_df is not None and len(lme_df) and 'cohens_d' in lme_df.columns:
        cons['cohens_d_LME']=cons['feature'].map(lme_df.set_index('feature')['cohens_d'].to_dict())
        if 'd_ci_lo' in lme_df.columns:
            cons['d_ci_lo']=cons['feature'].map(lme_df.set_index('feature')['d_ci_lo'].to_dict())
            cons['d_ci_hi']=cons['feature'].map(lme_df.set_index('feature')['d_ci_hi'].to_dict())
    return cons.sort_values('n_methods_sig',ascending=False)
