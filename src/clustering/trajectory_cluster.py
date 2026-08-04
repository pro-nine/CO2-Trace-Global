"""
Country CO2 Trajectory Clustering
=====================================
K-means clustering on per-capita CO2 trajectories (1990-2022), normalized
to each country's 1990 baseline (index = 1.0 in 1990). This clusters
countries by the SHAPE of their emissions path — not by income group,
region, or political bloc, which is the conventional grouping.

The key question: do countries' actual trajectories cluster in ways that
are more or less informative than the usual political/income groupings?

Pre-processing:
  - Interpolate minor missing years (≤3 consecutive) within each series
  - Drop countries with >5 missing years after interpolation
  - Normalize to 1990=1.0 so clustering is on trajectory SHAPE,
    not on absolute emission level (otherwise China's high absolute
    level would dominate purely on scale)

Cluster labeling: after fitting, each cluster is described by its
median trajectory and labeled by the dominant pattern:
  "Sustained reducers"   -- consistently below 1990 and trending down
  "Peak and decline"     -- rose then fell, now below or near 1990
  "Growth then plateau"  -- rose fast, now leveling off
  "Rapid growth"         -- strong continuous increase throughout
  "Collapse-recover"     -- sharp dip (often post-Soviet) then recovery
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

YEAR_COLS = list(range(1990, 2023))
N_CLUSTERS_DEFAULT = 5
RANDOM_STATE = 42


def build_trajectory_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a wide-format DataFrame: rows = countries, columns = years.
    Values are the per-capita CO2 INDEX relative to 1990 (1990=1.0).
    """
    wide = panel.pivot_table(index="iso_code", columns="year",
                              values="co2_per_capita", aggfunc="mean")
    available_years = [y for y in YEAR_COLS if y in wide.columns]
    wide = wide[available_years]

    # Normalize to 1990 baseline
    if 1990 in wide.columns:
        wide = wide.div(wide[1990], axis=0)

    # Interpolate within-series gaps (forward then backward for edge years)
    wide = wide.interpolate(axis=1, limit=3, limit_direction="both")

    # Drop countries with too many missing years
    wide = wide.dropna(thresh=len(available_years) - 5)
    return wide


def find_optimal_k(X: np.ndarray, k_range=(3, 8)) -> int:
    """Find k with best silhouette score."""
    best_k, best_score = k_range[0], -1
    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_trajectories(panel: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    wide = build_trajectory_matrix(panel)
    X = wide.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.T).T   # scale each year-feature, not each country

    if n_clusters is None:
        n_clusters = find_optimal_k(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)

    result = pd.DataFrame({"iso_code": wide.index, "cluster": labels})

    # Add country names from panel
    names = panel[["iso_code", "country"]].drop_duplicates()
    result = result.merge(names, on="iso_code", how="left")

    # Add median trajectory per cluster for labeling
    wide_labeled = wide.copy()
    wide_labeled["cluster"] = labels
    cluster_medians = wide_labeled.groupby("cluster").median()

    # Add some characterization metrics
    result["co2_index_2000"] = wide.reindex(result["iso_code"])[[2000]].values.flatten() if 2000 in wide.columns else np.nan
    result["co2_index_2010"] = wide.reindex(result["iso_code"])[[2010]].values.flatten() if 2010 in wide.columns else np.nan
    result["co2_index_2022"] = wide.reindex(result["iso_code"])[[2022]].values.flatten() if 2022 in wide.columns else np.nan

    return result, cluster_medians, n_clusters


if __name__ == "__main__":
    from src.data.loader import build_panel, load_raw
    co2_raw, energy_raw = load_raw()
    panel = build_panel(co2_raw, energy_raw)

    result, medians, n_k = cluster_trajectories(panel)
    print(f"Optimal k: {n_k}  |  Countries clustered: {len(result)}")
    print("\nCluster median CO2 index (1990=1.0):")
    print(medians[[1990, 2000, 2010, 2022]].round(2).to_string())
    print("\nCluster membership samples:")
    for c in sorted(result["cluster"].unique()):
        members = result[result["cluster"] == c]["country"].tolist()
        print(f"\nCluster {c} ({len(members)} countries):")
        print("  " + ", ".join(members[:12]) + ("..." if len(members) > 12 else ""))
