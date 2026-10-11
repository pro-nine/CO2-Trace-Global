"""Tapio decoupling elasticity, spell episodes and censored survival analysis.

The four-state sign rule remains as a compatibility layer. New reporting uses
Tapio's elasticity classes. Strong decoupling is a positive GDP/cap change and
negative CO2/cap change. Spell duration is censored if the spell reaches the
last observed year; Kaplan-Meier/log-rank avoid treating those spells as ended.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass

class DecouplingState(Enum):
    ABSOLUTE = "absolute_decoupling"
    RELATIVE = "relative_decoupling"
    NO_DECOUPLING = "no_decoupling"
    DEGROWTH = "degrowth_reduction"
    UNCLEAR = "unclear"

TAPIO_STATES = ["strong_decoupling","weak_decoupling","expansive_coupling","expansive_negative_decoupling",
"strong_negative_decoupling","weak_negative_decoupling","recessive_coupling","recessive_decoupling","flat"]

def tapio_state(d_co2,d_gdp,flat_tol=1e-4):
    if not np.isfinite(d_co2) or not np.isfinite(d_gdp) or abs(d_gdp)<flat_tol: return "flat"
    e=d_co2/d_gdp
    if d_gdp>0:
        if d_co2<0: return "strong_decoupling"
        return "weak_decoupling" if e<.8 else ("expansive_coupling" if e<=1.2 else "expansive_negative_decoupling")
    if d_co2>0: return "strong_negative_decoupling"
    return "weak_negative_decoupling" if e<.8 else ("recessive_coupling" if e<=1.2 else "recessive_decoupling")

def annual_states(panel):
    p=panel.sort_values(["iso_code","year"]).copy(); g=p.groupby("iso_code")
    p["d_co2"]=g["co2_per_capita"].pct_change(fill_method=None); p["d_gdp"]=g["gdp_per_capita"].pct_change(fill_method=None)
    consecutive=g["year"].diff()==1
    p["tapio_state"]=[tapio_state(a,b) if ok else "flat" for a,b,ok in zip(p.d_co2,p.d_gdp,consecutive)]
    return p[["iso_code","country","year","d_co2","d_gdp","tapio_state"]]

def find_spells(states,target="strong_decoupling",gap_tolerance=0,last_year=None):
    last_year=last_year or int(states.year.max()); rows=[]
    state_col="tapio_state" if "tapio_state" in states else "state"
    for iso,g in states.sort_values("year").groupby("iso_code"):
        years=g.loc[g[state_col]==target,"year"].to_numpy()
        if not len(years): continue
        start=prev=years[0]; count=1
        for y in years[1:]:
            if y-prev<=gap_tolerance+1: prev=y; count+=1
            else:
                rows.append((iso,g.country.iloc[0],int(start),int(prev),count,int(prev)==last_year)); start=prev=y; count=1
        rows.append((iso,g.country.iloc[0],int(start),int(prev),count,int(prev)==last_year))
    out=pd.DataFrame(rows,columns=["iso_code","country","start","end","years_in_state","ongoing"])
    if out.empty: out["length"]=pd.Series(dtype=int); return out
    out["length"]=out.end-out.start+1
    return out

def kaplan_meier(durations,observed):
    """Kaplan-Meier curve. Censored durations stay in the risk set and appear as zero-event rows."""
    d,o=np.asarray(durations,int),np.asarray(observed,int); s=1.; rows=[(0,1.,len(d),0)]
    for t in np.unique(d):
        risk=int((d>=t).sum()); events=int(((d==t)&(o==1)).sum()); s*=1-events/risk; rows.append((int(t),s,risk,events))
    return pd.DataFrame(rows,columns=["t","survival","at_risk","events"])

def km_median(curve):
    below=curve[curve.survival<=.5]
    return float(below.t.iloc[0]) if len(below) else float("nan")

def logrank(d1,o1,d2,o2):
    from scipy.stats import chi2
    d=np.r_[np.asarray(d1,int),np.asarray(d2,int)]; o=np.r_[np.asarray(o1,int),np.asarray(o2,int)]
    group=np.r_[np.zeros(len(d1),int),np.ones(len(d2),int)]; oe=var=0.
    for t in np.unique(d[o==1]):
        at=d>=t; risk=int(at.sum()); r1=int((at&(group==0)).sum()); ev=int(((d==t)&(o==1)).sum()); ev1=int(((d==t)&(o==1)&(group==0)).sum())
        oe+=ev1-ev*r1/risk
        if risk>1: var+=ev*(r1/risk)*(1-r1/risk)*(risk-ev)/(risk-1)
    stat=oe**2/var if var>0 else 0.
    return {"chi2":float(stat),"p_value":float(chi2.sf(stat,1))}

def income_groups(panel,year=1990):
    x=panel[panel.year==year].set_index("iso_code").gdp_per_capita
    return pd.qcut(x,3,labels=["low","middle","high"]).astype(str)

def spell_survival(spells,groups):
    x=spells.assign(group=spells.iso_code.map(groups),event=(~spells.ongoing).astype(int)).dropna(subset=["group"])
    curves={g:kaplan_meier(v.length,v.event) for g,v in x.groupby("group")}; med={g:km_median(c) for g,c in curves.items()}
    hi,lo=x[x.group=="high"],x[x.group=="low"]
    test=logrank(hi.length,hi.event,lo.length,lo.event) if len(hi) and len(lo) else {}
    return {"curves":curves,"medians":med,"logrank_high_vs_low":test,"n_spells":len(x)}

# Legacy API retained so existing external scripts do not break.
@dataclass
class CountryDecoupling:
    iso_code:str; country:str; state_by_decade:dict; years_absolute:int; longest_abs_run:int; currently_absolute:bool
    suspect_offshoring:bool; suspect_deindustrialisation:bool; gdp_per_cap_growth_pct_1990_2022:float; co2_per_cap_change_pct_1990_2022:float

def _classify_window(gdp_growth,co2_growth,threshold=.5):
    if not np.isfinite(gdp_growth) or not np.isfinite(co2_growth): return DecouplingState.UNCLEAR
    if gdp_growth>threshold:
        if co2_growth<0:return DecouplingState.ABSOLUTE
        return DecouplingState.RELATIVE if co2_growth<gdp_growth else DecouplingState.NO_DECOUPLING
    return DecouplingState.DEGROWTH if gdp_growth< -threshold and co2_growth<0 else DecouplingState.UNCLEAR

def classify_country(df,iso):
    s=df[df.iso_code==iso].sort_values("year").dropna(subset=["gdp_per_capita","co2_per_capita"])
    if len(s)<10:return None
    dg=s.gdp_per_capita.pct_change(fill_method=None); dc=s.co2_per_capita.pct_change(fill_method=None)
    flag=(dg>0)&(dc<0)&(s.year.diff()==1); years=s.year.to_numpy(); longest=run=0; prev=None
    for y,f in zip(years,flag.fillna(False)):
        if bool(f) and (prev is None or y==prev+1): run+=1
        else: run=int(bool(f))
        longest=max(longest,run); prev=y
    recent=s[s.year>=2018]; current=False
    if len(recent)>=3: current=recent.gdp_per_capita.iloc[-1]>recent.gdp_per_capita.iloc[0] and recent.co2_per_capita.iloc[-1]<recent.co2_per_capita.iloc[0]
    states={}
    for label,y0,y1 in [("1990-2000",1990,2000),("2000-2010",2000,2010),("2010-2022",2010,2022)]:
        a=s[s.year==y0]; b=s[s.year==y1]
        states[label]=_classify_window(np.nan,np.nan).value if a.empty or b.empty else _classify_window(((b.gdp_per_capita.iloc[0]/a.gdp_per_capita.iloc[0])**(1/(y1-y0))-1)*100,((b.co2_per_capita.iloc[0]/a.co2_per_capita.iloc[0])**(1/(y1-y0))-1)*100).value
    e=s.energy_per_gdp.dropna(); c=s.co2_per_energy.dropna(); ed=e.iloc[-1]/e.iloc[0]-1 if len(e)>1 and e.iloc[0]>0 else 0; cd=c.iloc[-1]/c.iloc[0]-1 if len(c)>1 and c.iloc[0]>0 else 0
    first,last=s.iloc[0],s.iloc[-1]
    return CountryDecoupling(iso,first.country,states,int(flag.sum()),longest,current,ed<-.3 and cd>-.1,ed<-.4 and abs(cd)<.1,(last.gdp_per_capita/first.gdp_per_capita-1)*100,(last.co2_per_capita/first.co2_per_capita-1)*100)

def classify_all(df):
    rows=[]
    for iso in df.iso_code.unique():
        r=classify_country(df,iso)
        if r: rows.append({"iso_code":r.iso_code,"country":r.country,"years_absolute_decoupling":r.years_absolute,"longest_consecutive_abs_run":r.longest_abs_run,"currently_absolute_2018_2022":r.currently_absolute,"decade_1990_2000":r.state_by_decade.get("1990-2000","unclear"),"decade_2000_2010":r.state_by_decade.get("2000-2010","unclear"),"decade_2010_2022":r.state_by_decade.get("2010-2022","unclear"),"suspect_offshoring":r.suspect_offshoring,"suspect_deindustrialisation":r.suspect_deindustrialisation,"gdp_per_cap_growth_pct":r.gdp_per_cap_growth_pct_1990_2022,"co2_per_cap_change_pct":r.co2_per_cap_change_pct_1990_2022})
    return pd.DataFrame(rows)
