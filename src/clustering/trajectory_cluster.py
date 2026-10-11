"""Robust log-trajectory clustering and co-assignment stability.

Countries are compared by RMS Euclidean distance on log(CO2pc / base-year CO2pc).
Near-zero base-year emitters are excluded explicitly because their ratios are
not meaningful. Hierarchical clustering is accompanied by moving-block bootstrap
co-assignment: the latter shows which pairwise stories persist when years change.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score

MIN_START_CO2_PC=.1

def log_trajectories(panel,base_year=1990,min_start=MIN_START_CO2_PC):
    wide=panel.pivot(index="iso_code",columns="year",values="co2_per_capita").sort_index(axis=1)
    complete=wide.dropna(); start=complete[base_year]; keep=(start>=min_start)&(complete>0).all(axis=1)
    excluded=sorted(set(wide.index)-set(complete.index[keep])); return np.log(complete[keep].div(start[keep],axis=0)),excluded

def rms_distance(L):
    d=squareform(pdist(L.to_numpy(),"euclidean"))/np.sqrt(L.shape[1]); return pd.DataFrame(d,index=L.index,columns=L.index)

def cluster_labels(D,k): return fcluster(linkage(squareform(D,checks=False),method="average"),k,criterion="maxclust")-1

def choose_k(D,k_range=(3,8)):
    scores={k:float(silhouette_score(D,cluster_labels(D,k),metric="precomputed")) for k in range(k_range[0],k_range[1]+1)}
    best=max(scores,key=scores.get); return best,scores

def _block_cols(n,block,rng):
    starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block))); return np.concatenate([np.arange(s,s+block) for s in starts])[:n]

def consensus(L,k,n_boot=200,block=5,seed=0):
    rng=np.random.default_rng(seed); X=L.to_numpy(); n=len(L); co=np.zeros((n,n))
    for _ in range(n_boot):
        cols=_block_cols(X.shape[1],min(block,X.shape[1]),rng); D=squareform(pdist(X[:,cols]))/np.sqrt(len(cols)); lab=cluster_labels(D,k); co+=(lab[:,None]==lab[None,:])
    return pd.DataFrame(co/n_boot,index=L.index,columns=L.index)

def compare(focal,peers,D,C):
    rows=[{"peer":p,"distance":float(D.loc[focal,p]),"co_assignment":float(C.loc[focal,p])} for p in peers if p in D.index and p!=focal]
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)

def boundary_countries(D,labels,index,threshold=.1):
    s=silhouette_samples(D,labels,metric="precomputed"); return [i for i,v in zip(index,s) if v<threshold]

def cluster_summary(L,labels,years=(2000,2010,2022)):
    med=np.exp(L.groupby(labels).median()); out=med[list(years)].copy(); out["n"]=pd.Series(labels).value_counts().sort_index().values
    def shape(row):
        end,mid=row[years[-1]],row[years[1]]
        if end>1.25:return "growth"
        if end<.75 and row[years[0]]<.8:return "early drop"
        if end<.75:return "late decline"
        return "plateau" if mid>.9 else "decline then plateau"
    out["shape"]=out.apply(shape,axis=1); return out
