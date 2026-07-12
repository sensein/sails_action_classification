"""Shared FDR-correction / significance-annotation helpers (exact duplicates across analysis modules)."""

from statsmodels.stats.multitest import multipletests


def fdr_annotate_v1(df_res, p_col):
    if len(df_res)>1:
        _,p_fdr,_,_=multipletests(df_res[p_col].fillna(1),method='fdr_bh')
        df_res=df_res.copy(); df_res['p_fdr']=p_fdr
    else:
        df_res=df_res.copy(); df_res['p_fdr']=df_res[p_col]
    df_res['sig_fdr05']=df_res['p_fdr']<0.05
    df_res['sig_raw05']=df_res[p_col]<0.05
    return df_res


def fdr_annotate_v2(df_res, p_col):
    if len(df_res) > 1:
        _, p_fdr, _, _ = multipletests(df_res[p_col].fillna(1), method='fdr_bh')
        df_res = df_res.copy(); df_res['p_fdr'] = p_fdr
    else:
        df_res = df_res.copy(); df_res['p_fdr'] = df_res[p_col]
    df_res['sig_fdr05'] = df_res['p_fdr'] < 0.05
    df_res['sig_raw05'] = df_res[p_col] < 0.05
    return df_res


def add_sig_bar_v1(ax, x1, x2, y, p, h=0.02):
    label='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col='#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],lw=1.2,color='black')
    ax.text((x1+x2)/2,y+h*1.05,label,ha='center',va='bottom',fontsize=10,color=col,fontweight='bold')


def add_sig_bar_v2(ax, x1, x2, y, p, h=0.02):
    label = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    col   = '#cc0000' if p<0.001 else '#e06600' if p<0.01 else '#888800' if p<0.05 else '#888888'
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw=1.2, color='black')
    ax.text((x1+x2)/2, y+h*1.05, label, ha='center', va='bottom',
            fontsize=10, color=col, fontweight='bold')
