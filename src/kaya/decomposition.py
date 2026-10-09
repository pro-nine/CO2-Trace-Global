"""Path-aware Kaya LMDI and measurement-noise sensitivity.

The additive Kaya identity decomposes CO2 into population, affluence,
energy-intensity and carbon-intensity effects. `chain_lmdi` decomposes each
adjacent year and sums it, preserving path-dependent log-mean weights.
`driver_stability` perturbs the observed endpoint inputs on the log scale; its
percentiles are a sensitivity analysis, not a sampling confidence interval.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
FACTORS=["population","gdp_per_capita","energy_per_gdp","co2_per_energy"]
EFFECTS=["delta_population","delta_affluence","delta_intensity","delta_carbon_mix"]
REDUCERS=["delta_intensity","delta_carbon_mix"]

def log_mean(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.where(np.isclose(a,b,rtol=1e-12,atol=0),a,(a-b)/(np.log(a)-np.log(b)))

def lmdi_effects(c0,c1,f0,f1):
    effects=log_mean(c1,c0)[:,None]*(np.log(f1)-np.log(f0))
    return effects,(c1-c0)-effects.sum(axis=1)

def eligible_countries(panel,year_start,year_end):
    sub=panel[panel.year.between(year_start,year_end)]; cols=["co2",*FACTORS]
    good=sub.groupby("iso_code")[cols].apply(lambda g: bool((g>0).all().all()))
    n=sub.groupby("iso_code").year.nunique()
    return sorted(good[good&(n==year_end-year_start+1)].index)

def chain_lmdi(panel,year_start=1990,year_end=2022):
    sub=panel[panel.iso_code.isin(eligible_countries(panel,year_start,year_end))&panel.year.between(year_start,year_end)]
    rows=[]
    for iso,g in sub.sort_values("year").groupby("iso_code"):
        c=g.co2.to_numpy(float); f=g[FACTORS].to_numpy(float); e,r=lmdi_effects(c[:-1],c[1:],f[:-1],f[1:])
        x=pd.DataFrame(e,columns=EFFECTS); x.insert(0,"year",g.year.to_numpy()[1:]); x.insert(0,"iso_code",iso)
        x["delta_co2"]=c[1:]-c[:-1]; x["residual"]=r; rows.append(x)
    return pd.concat(rows,ignore_index=True)

def endpoint_lmdi(panel,year_start=1990,year_end=2022):
    a=panel[panel.year==year_start].set_index("iso_code"); b=panel[panel.year==year_end].set_index("iso_code")
    ix=sorted(set(a.index)&set(b.index)); cols=["co2",*FACTORS]; a,b=a.loc[ix],b.loc[ix]
    ok=((a[cols]>0)&(b[cols]>0)).all(axis=1)&a[cols].notna().all(axis=1)&b[cols].notna().all(axis=1); a,b=a[ok],b[ok]
    e,r=lmdi_effects(a.co2.to_numpy(float),b.co2.to_numpy(float),a[FACTORS].to_numpy(float),b[FACTORS].to_numpy(float))
    out=pd.DataFrame(e,columns=EFFECTS,index=a.index); out.insert(0,"country",a.country)
    out["co2_start"],out["co2_end"]=a.co2,b.co2; out["co2_change_mt"]=b.co2-a.co2
    out["co2_pct_change"]=out.co2_change_mt/a.co2*100; out["residual"]=r
    return out.reset_index()

def dominant(df,columns=EFFECTS): return df[list(columns)].abs().idxmax(axis=1)
def main_reducer(df): return df[REDUCERS].idxmin(axis=1)

def driver_stability(panel,year_start=1990,year_end=2022,n_draws=300,noise_sd=.03,seed=0):
    base=endpoint_lmdi(panel,year_start,year_end); cols=["co2",*FACTORS]
    a=panel[panel.year==year_start].set_index("iso_code").loc[base.iso_code]; b=panel[panel.year==year_end].set_index("iso_code").loc[base.iso_code]
    rng=np.random.default_rng(seed); n=len(base); d0,red0=dominant(base),main_reducer(base); kd=np.zeros(n); kr=np.zeros(n); draws=np.zeros((n_draws,n,4))
    for i in range(n_draws):
        f0=a[cols].to_numpy(float)*np.exp(rng.normal(0,noise_sd,(n,5))); f1=b[cols].to_numpy(float)*np.exp(rng.normal(0,noise_sd,(n,5)))
        e,_=lmdi_effects(f0[:,0],f1[:,0],f0[:,1:],f1[:,1:]); draws[i]=e; tmp=pd.DataFrame(e,columns=EFFECTS)
        kd+=dominant(tmp).to_numpy()==d0.to_numpy(); kr+=main_reducer(tmp).to_numpy()==red0.to_numpy()
    out=base[["iso_code","country","co2_pct_change"]].copy(); out["dominant_driver"],out["main_reducer"]=d0,red0
    out["p_dominant_holds"],out["p_main_reducer_holds"]=kd/n_draws,kr/n_draws
    for j,name in enumerate(EFFECTS): out[f"{name}_p05"]=np.percentile(draws[:,:,j],5,axis=0); out[f"{name}_p95"]=np.percentile(draws[:,:,j],95,axis=0)
    return out

def efficiency_led_share(panel,n_draws=300,noise_sd=.03,seed=1):
    base=endpoint_lmdi(panel); red=base[base.co2_change_mt<0]; cols=["co2",*FACTORS]
    a=panel[panel.year==1990].set_index("iso_code").loc[red.iso_code]; b=panel[panel.year==2022].set_index("iso_code").loc[red.iso_code]
    rng=np.random.default_rng(seed); shares=[]
    for _ in range(n_draws):
        f0=a[cols].to_numpy(float)*np.exp(rng.normal(0,noise_sd,(len(red),5))); f1=b[cols].to_numpy(float)*np.exp(rng.normal(0,noise_sd,(len(red),5)))
        e,_=lmdi_effects(f0[:,0],f1[:,0],f0[:,1:],f1[:,1:]); still=(f1[:,0]-f0[:,0])<0; shares.append(float((e[still,2]<e[still,3]).mean()))
    return {"n_reducers":len(red),"efficiency_led_share":float((main_reducer(red)=="delta_intensity").mean()),"p05":float(np.percentile(shares,5)),"p95":float(np.percentile(shares,95))}
