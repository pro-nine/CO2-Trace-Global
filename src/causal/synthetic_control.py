"""Synthetic control with pre-period holdout and in-space placebo inference.

This is an exploratory attribution design, not proof of policy causality. Donor
pool and intervention year must be selected before examining post-treatment gaps.
Weights are nonnegative and sum to one; nearest donor screening and a held-out
pre-period prevent the near-perfect in-sample fit common with many donors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def simplex_least_squares(X,y,iters=2000,tol=1e-12):
    n=X.shape[1]; w=np.full(n,1/n)
    for _ in range(iters):
        grad=2*X.T@(X@w-y); j=int(np.argmin(grad)); direction=-w.copy(); direction[j]+=1
        xd=X@direction; den=float(xd@xd)
        if den<=tol: break
        step=float(np.clip(-((X@w-y)@xd)/den,0,1))
        if step<=tol: break
        w+=step*direction
    return w

def outcome_matrix(panel,column,log=True):
    m=panel.pivot(index="year",columns="iso_code",values=column).dropna(axis=1)
    if log: m=m.loc[:,(m>0).all()]; return np.log(m)
    return m

def _nearest(outcomes,treated,mask,candidates,k):
    dist=np.sqrt(outcomes.loc[mask,candidates].sub(outcomes.loc[mask,treated],axis=0).pow(2).mean())
    return list(dist.sort_values().index[:k])

class SyntheticFit:
    def __init__(self,treated,treatment_year,weights,gap,pre_rmspe,post_rmspe):
        self.treated=treated; self.treatment_year=treatment_year; self.weights=weights; self.gap=gap; self.pre_rmspe=pre_rmspe; self.post_rmspe=post_rmspe
    @property
    def ratio(self): return self.post_rmspe/max(self.pre_rmspe,1e-12)
    @property
    def mean_post_gap(self): return float(self.gap[self.gap.index>=self.treatment_year].mean())

def fit_synthetic(outcomes,treated,treatment_year,donors=None,k=15,holdout_years=5):
    candidates=[c for c in (donors if donors is not None else outcomes.columns) if c!=treated]
    pre_years=outcomes.index[outcomes.index<treatment_year]
    if len(pre_years)<=holdout_years+2: raise ValueError("Not enough pre-treatment years for holdout")
    train_years=pre_years[:-holdout_years]; hold_years=pre_years[-holdout_years:]
    chosen=_nearest(outcomes,treated,outcomes.index.isin(train_years),candidates,min(k,len(candidates)))
    w0=simplex_least_squares(outcomes.loc[train_years,chosen].to_numpy(),outcomes.loc[train_years,treated].to_numpy())
    hold_gap=outcomes.loc[hold_years,treated].to_numpy()-outcomes.loc[hold_years,chosen].to_numpy()@w0
    w=simplex_least_squares(outcomes.loc[pre_years,chosen].to_numpy(),outcomes.loc[pre_years,treated].to_numpy())
    gap=pd.Series(outcomes[treated].to_numpy()-outcomes[chosen].to_numpy()@w,index=outcomes.index)
    return SyntheticFit(treated,treatment_year,pd.Series(w,index=chosen),gap,float(np.sqrt(np.mean(hold_gap**2))),float(np.sqrt(np.mean(gap[outcomes.index<treatment_year]**2))))

def placebo_test(outcomes,treated,treatment_year,max_pre_ratio=2.,k=15):
    main=fit_synthetic(outcomes,treated,treatment_year,k=k); ratios={}; std={}; donors=[c for c in outcomes.columns if c!=treated]
    for donor in donors:
        f=fit_synthetic(outcomes,donor,treatment_year,donors=donors,k=min(k,len(donors)-1))
        if f.pre_rmspe<=max_pre_ratio*max(main.pre_rmspe,1e-9): ratios[donor]=f.ratio; std[donor]=f.mean_post_gap/max(f.pre_rmspe,1e-9)
    a=np.asarray(list(ratios.values())); b=np.abs(np.asarray(list(std.values()))); stat=main.mean_post_gap/max(main.pre_rmspe,1e-9)
    return {"fit":main,"placebo_ratios":ratios,"n_placebos":len(a),"p_value":float((1+(a>=main.ratio).sum())/(1+len(a))),"standardised_gap":float(stat),"p_value_gap":float((1+(b>=abs(stat)).sum())/(1+len(b)))}
