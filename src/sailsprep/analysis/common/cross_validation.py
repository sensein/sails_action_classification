"""Shared LOSO-CV helper (exact duplicate: crawling, running, walking)."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def run_loso_child(cdf, feat_cols, clf_name='LR', n_perm=500, seed=42, subset_name=''):
    df_=cdf.copy()
    df_['y']=(df_['Group']=='ASD').astype(int)
    if df_['y'].sum()<4 or (1-df_['y']).sum()<4: return None
    usable=[f for f in feat_cols if f in df_.columns and df_[f].notna().mean()>0.5]
    if len(usable)<2: return None
    df_[usable]=df_[usable].fillna(df_[usable].median())
    if clf_name=='LR':
        clf=LogisticRegression(max_iter=2000,C=0.1,class_weight='balanced',random_state=seed)
    elif clf_name=='SVM':
        clf=SVC(kernel='rbf',class_weight='balanced',probability=True,random_state=seed)
    else:
        clf=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=seed,n_jobs=-1)
    pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('clf',clf)])
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
    perm_aucs=[roc_auc_score(rng.permuted(np.array(y_true)),y_score) for _ in range(n_perm)]
    p_perm=float((np.array(perm_aucs)>=auc).mean())
    print(f"  [{subset_name} {clf_name}] AUC={auc:.3f}  AP={ap:.3f}  p_perm={p_perm:.4f}  n_feat={len(usable)}")
    fi_df=pd.DataFrame()
    if clf_name=='RF':
        rf_m=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=seed)
        rf_m.fit(SimpleImputer(strategy='median').fit_transform(df_[usable]),df_['y'])
        fi_df=pd.DataFrame({'feature':usable,'importance':rf_m.feature_importances_}
                           ).sort_values('importance',ascending=False)
    return {'auc':auc,'ap':ap,'perm_p':p_perm,'n_features':len(usable),
            'n_subjects':df_['pid'].nunique(),'y_true':y_true,'y_score':y_score,
            'perm_aucs':perm_aucs,'clf':clf_name,'feature_importance':fi_df}
