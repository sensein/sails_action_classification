"""Statistical-pipeline functions shared exclusively between crawling.py and running.py (verified byte-identical, including their cohen_d/fdr_annotate/_use_random_slope dependencies -- see refactor plan for verification method)."""

import re

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from sailsprep.analysis.common.effect_size import cohen_d_v1 as cohen_d
from sailsprep.analysis.common.significance import fdr_annotate_v1 as fdr_annotate
from sailsprep.analysis.common.mixed_models import _use_random_slope
from sailsprep.analysis.common.keypoints import AGE_BANDS

N_PERM = 5000

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    _lme4     = importr('lme4')
    _lmerTest = importr('lmerTest')
    _RPY2_OK  = True
    print("[rpy2] lme4 + lmerTest available — Kenward-Roger enabled")
except Exception:
    _RPY2_OK  = False
    print("[rpy2] NOT available — falling back to statsmodels LME")

try:
    from wildboottest.wildboottest import WildboottestHC
    _WBT_OK = True
    print("[wildboottest] available — CR2 enabled")
except Exception:
    _WBT_OK = False
    print("[wildboottest] NOT available — skipping CR2")


def bootstrap_ci_d(a, b, n_boot=500, seed=42):
    rng=np.random.default_rng(seed)
    boot=[cohen_d(rng.choice(a,len(a),replace=True),
                  rng.choice(b,len(b),replace=True))
          for _ in range(n_boot)]
    return float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))


def run_consistency_gate_bands(child_df, feat_cols, sig_feats):
    band_mwu={}
    for band in AGE_BANDS.keys():
        sub=child_df[child_df['age_band']==band]
        asd_n=sub[sub['Group']=='ASD']['pid'].nunique()
        nan_n=sub[sub['Group']=='Non-ASD']['pid'].nunique()
        if asd_n<3 or nan_n<3: continue
        recs=[]
        for feat in feat_cols:
            av=sub[sub['Group']=='ASD'][feat].dropna().values
            nv=sub[sub['Group']=='Non-ASD'][feat].dropna().values
            if len(av)<3 or len(nv)<3: continue
            _,p=stats.mannwhitneyu(av,nv,alternative='two-sided')
            recs.append({'feature':feat,'cohens_d':cohen_d(av,nv),'p_raw':p,'age_band':band})
        if recs: band_mwu[band]=pd.DataFrame(recs)
    cons_recs=[]; consistent_feats=[]
    band_all=pd.concat(band_mwu.values(),ignore_index=True) if band_mwu else pd.DataFrame()
    for feat in sig_feats:
        if len(band_all)==0: break
        sub=band_all[band_all['feature']==feat]
        if len(sub)<2: continue
        signs=np.sign(sub['cohens_d'].values)
        n_same=int((signs==signs[0]).sum()); passed=(n_same==len(sub))
        cons_recs.append({'feature':feat,'n_bands_tested':len(sub),
                          'n_same_direction':n_same,'consistent':passed})
        if passed: consistent_feats.append(feat)
    return pd.DataFrame(cons_recs),consistent_feats


def run_child_permutation(child_df, feat_cols, n_perm=N_PERM, subset_label='ALL'):
    rng=np.random.default_rng(42); records=[]
    for feat in feat_cols:
        sub=child_df[['pid','Group',feat]].dropna()
        if sub['Group'].nunique()<2: continue
        av=sub[sub['Group']=='ASD'][feat].values
        nv=sub[sub['Group']=='Non-ASD'][feat].values
        if len(av)<3 or len(nv)<3: continue
        obs_stat=abs(np.mean(av)-np.mean(nv)); n_asd=len(av)
        vals_arr=sub.set_index('pid')[feat].to_numpy(); n_total=len(vals_arr)
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


def run_lme_kr(clip_df, feat_cols, subset_label='ALL',
               covariates='age_mo_c', interaction=True, allow_random_slope=True):
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    use_slope=allow_random_slope and _use_random_slope(df_use)

    for feat in feat_cols:
        sub=df_use[['pid','session',feat,'Group_bin','age_mo','age_mo_c']].dropna()
        if sub['Group_bin'].nunique()<2: continue
        if sub.groupby('Group_bin')['pid'].nunique().min()<3: continue
        av=sub[sub['Group_bin']==1][feat].values
        nv=sub[sub['Group_bin']==0][feat].values
        d=cohen_d(av,nv); ci_lo,ci_hi=bootstrap_ci_d(av,nv,n_boot=500)
        p_val=np.nan; coef=np.nan; se=np.nan; inter_p=np.nan
        method_used='none'; converged=False; slope_used=False

        if _RPY2_OK:
            try:
                safe=re.sub(r'[^A-Za-z0-9_]','_',feat)
                sub2=sub.rename(columns={feat:safe})
                inter_term='+Group_bin:age_mo_c' if interaction else ''
                if use_slope and sub.groupby('pid').size().median()>=2:
                    formula=f'{safe} ~ Group_bin+{covariates}{inter_term}+(1+age_mo_c|pid)'
                else:
                    formula=f'{safe} ~ Group_bin+{covariates}{inter_term}+(1|pid)'
                r_df=pandas2ri.py2rpy(sub2)
                ro.globalenv['r_df']=r_df
                ro.r(f'fit <- lmerTest::lmer({formula}, data=r_df, REML=TRUE)')
                summ=ro.r('as.data.frame(coef(summary(fit, ddf="Kenward-Roger")))')
                summ_pd=pandas2ri.rpy2py(summ)
                if 'Group_bin' in summ_pd.index:
                    coef =float(summ_pd.loc['Group_bin','Estimate'])
                    se   =float(summ_pd.loc['Group_bin','Std. Error'])
                    p_val=float(summ_pd.loc['Group_bin','Pr(>|t|)'])
                    method_used='LME_KR'; converged=True; slope_used='age_mo_c|pid' in formula
                    inter_key='Group_bin:age_mo_c'
                    inter_p=float(summ_pd.loc[inter_key,'Pr(>|t|)']) if inter_key in summ_pd.index else np.nan
            except: pass

        if method_used=='none':
            try:
                inter_term='+Group_bin:age_mo_c' if interaction else ''
                formula_sm=f'{feat} ~ Group_bin+{covariates}{inter_term}'
                fitted=None
                if use_slope and sub.groupby('pid').size().median()>=2:
                    try:
                        fitted=smf.mixedlm(formula_sm,sub,groups=sub['pid'],
                                           re_formula='~age_mo_c').fit(
                            method=['lbfgs','bfgs'],reml=True,maxiter=300)
                        if not fitted.converged: fitted=None
                        else: slope_used=True
                    except: fitted=None
                if fitted is None:
                    fitted=smf.mixedlm(formula_sm,sub,groups=sub['pid']).fit(
                        method=['lbfgs','bfgs'],reml=True,maxiter=300)
                coef =float(fitted.params.get('Group_bin',np.nan))
                se   =float(fitted.bse.get('Group_bin',np.nan))
                p_val=float(fitted.pvalues.get('Group_bin',np.nan))
                inter_p=float(fitted.pvalues.get('Group_bin:age_mo_c',np.nan))
                method_used='LME_noKR'; converged=bool(fitted.converged)
            except: pass

        records.append({'feature':feat,'subset':subset_label,'method':method_used,
                        'coef_ASD':coef,'se':se,'p_raw':p_val,'interaction_p':inter_p,
                        'cohens_d':d,'d_ci_lo':ci_lo,'d_ci_hi':ci_hi,
                        'converged':converged,'random_slope_used':slope_used,
                        'n_asd':sub[sub['Group_bin']==1]['pid'].nunique(),
                        'n_nasd':sub[sub['Group_bin']==0]['pid'].nunique(),
                        'n_clips':len(sub)})

    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')


def run_cr2(clip_df, feat_cols, subset_label='ALL'):
    if not _WBT_OK:
        print("  [CR2] wildboottest not available"); return pd.DataFrame()
    records=[]
    df_use=clip_df.copy()
    df_use['Group_bin']=(df_use['Group']=='ASD').astype(float)
    df_use['age_mo_c']=df_use['age_mo']-df_use['age_mo'].mean()
    for feat in feat_cols:
        sub=df_use[['pid','Group_bin','age_mo_c',feat]].dropna(subset=['pid','Group_bin',feat])
        if sub['Group_bin'].nunique()<2 or len(sub)<10: continue
        X=sub[['Group_bin','age_mo_c']].values.astype(float)
        y=sub[feat].values.astype(float); clusters=sub['pid'].values
        try:
            wbt=WildboottestHC(X=X,y=y,cluster=clusters,
                               R=np.eye(2)[[0],:],B=999,bootstrap_type='WCR11')
            wbt.get_wildboottest()
            records.append({'feature':feat,'subset':subset_label,'method':'CR2',
                            'p_raw':float(wbt.pvalue),'n_clips':len(sub)})
        except: continue
    if not records: return pd.DataFrame()
    return fdr_annotate(pd.DataFrame(records),'p_raw').sort_values('p_raw')
