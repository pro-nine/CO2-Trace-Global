"""Claim audit: every sentence of the v1 report that contained a number is recomputed.

v1 wrote its findings as fixed text. This module turns each of them into a
function of the data that returns evidence and a verdict, so the report cannot
drift away from what the code finds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.clustering.trajectory_cluster import choose_k, cluster_labels, compare, consensus, log_trajectories, rms_distance
from src.decoupling.classifier import annual_states, classify_all, find_spells
from src.kaya.decomposition import chain_lmdi, efficiency_led_share, eligible_countries, endpoint_lmdi

WINSOR = float(np.log(4.0))  # trajectories beyond a 4x change are capped before distances are taken


def _v1_kmeans_margin(panel: pd.DataFrame, focal="DEU", rival="UKR", other="DNK") -> dict:
    """Re-run the v1 method (K-means, k=3, years standardised) and measure how firm Germany's cluster was."""
    wide = panel.pivot_table(index="iso_code", columns="year", values="co2_per_capita")
    wide = wide.div(wide[1990], axis=0).interpolate(axis=1, limit=3, limit_direction="both").dropna(thresh=wide.shape[1] - 5)
    X = StandardScaler().fit_transform(wide.values.T).T
    km = KMeans(3, random_state=42, n_init=20).fit(X)
    dist = km.transform(X)
    idx = list(wide.index)
    d = np.sort(dist[idx.index(focal)])
    lab = km.labels_
    return {"v1_distance_to_nearest_centre": float(d[0]), "v1_distance_to_second_centre": float(d[1]),
            "v1_centre_margin_pct": float((d[1] - d[0]) / d[0] * 100),
            "v1_same_cluster_as_ukraine": bool(lab[idx.index(focal)] == lab[idx.index(rival)]),
            "v1_same_cluster_as_denmark": bool(lab[idx.index(focal)] == lab[idx.index(other)])}


def claim_germany_ukraine(panel: pd.DataFrame, n_boot: int = 200) -> dict:
    L, excluded = log_trajectories(panel)
    L = L.clip(-WINSOR, WINSOR)
    D = rms_distance(L)
    k, _ = choose_k(D.to_numpy())
    C = consensus(L, k, n_boot=n_boot)
    table = compare("DEU", ["UKR", "DNK", "SWE", "GBR", "FRA", "AUT", "POL", "CZE"], D, C)
    row = table.set_index("peer")
    refuted = bool(row.loc["UKR", "distance"] > max(row.loc["DNK", "distance"], row.loc["SWE", "distance"])
                   and row.loc["UKR", "co_assignment"] < min(row.loc["DNK", "co_assignment"], row.loc["SWE", "co_assignment"]))
    ev = {"k": int(k), "n_countries": int(len(L)), "n_excluded_low_base": len(excluded),
          "distance_ukraine": float(row.loc["UKR", "distance"]), "distance_denmark": float(row.loc["DNK", "distance"]),
          "distance_sweden": float(row.loc["SWE", "distance"]), "co_assign_ukraine": float(row.loc["UKR", "co_assignment"]),
          "co_assign_denmark": float(row.loc["DNK", "co_assignment"]), "co_assign_sweden": float(row.loc["SWE", "co_assignment"]),
          **_v1_kmeans_margin(panel)}
    return {"id": "C1", "v1_claim": "Germany clusters with Ukraine, not with Denmark and Sweden.",
            "verdict": "not supported" if refuted else "supported", "evidence": ev, "table": table}


def claim_efficiency_led(panel: pd.DataFrame) -> dict:
    r = efficiency_led_share(panel)
    return {"id": "C2", "v1_claim": "Most countries that cut CO2 did so mainly through efficiency, not fuel switching.",
            "verdict": "supported" if r["p05"] > 0.5 else "not supported", "evidence": r}


def claim_sustained_decoupling(panel: pd.DataFrame) -> dict:
    states = annual_states(panel)
    strict = find_spells(states, gap_tolerance=0)
    tolerant = find_spells(states, gap_tolerance=1)
    ev = {"strict_longest_years": int(strict["length"].max()), "tolerant_longest_years": int(tolerant["length"].max()),
          "strict_spells": int(len(strict)), "strict_median_length": float(strict["length"].median()),
          "strict_share_one_year": float((strict["length"] == 1).mean()),
          "ongoing_spells_2022": int(strict["ongoing"].sum())}
    verdict = "definition-sensitive" if ev["tolerant_longest_years"] > 2 * ev["strict_longest_years"] else "supported"
    return {"id": "C3", "v1_claim": "Absolute decoupling is real but never sustained (longest run 7 years).",
            "verdict": verdict, "evidence": ev}


def claim_offshoring_flag(panel: pd.DataFrame) -> dict:
    dc = classify_all(panel)
    flagged = dc[dc["suspect_offshoring"]]
    rising = int((flagged["co2_per_cap_change_pct"] > 0).sum())
    ev = {"n_flagged": int(len(flagged)), "n_countries": int(len(dc)), "n_flagged_with_rising_co2_per_capita": rising,
          "share_rising": float(rising / max(len(flagged), 1)),
          "median_gdp_per_capita_growth_pct_flagged": float(flagged["gdp_per_cap_growth_pct"].median())}
    return {"id": "C4", "v1_claim": "41 countries show the signature of offshoring.",
            "verdict": "misleading" if ev["share_rising"] > 0.25 else "supported", "evidence": ev}


def claim_coverage(panel: pd.DataFrame) -> dict:
    ev = {"countries_in_panel": int(panel["iso_code"].nunique()), "countries_in_endpoint_decomposition": int(len(endpoint_lmdi(panel))),
          "countries_in_chain_decomposition": int(len(eligible_countries(panel, 1990, 2022)))}
    consistent = ev["countries_in_panel"] == ev["countries_in_endpoint_decomposition"]
    return {"id": "C5", "v1_claim": "The decomposition covers all 162 countries.",
            "verdict": "supported" if consistent else "denominator differs", "evidence": ev}


def efficiency_front_load(panel: pd.DataFrame, cutoff_year: int = 1998, min_effect_mt: float = 100.0) -> pd.DataFrame:
    """Share of each large reducer's cumulative efficiency effect that fell in 1991..cutoff_year."""
    chain = chain_lmdi(panel)
    ep = endpoint_lmdi(panel)
    reducers = set(ep.loc[ep["co2_change_mt"] < 0, "iso_code"])
    c = chain[chain["iso_code"].isin(reducers)]
    total = c.groupby("iso_code")["delta_intensity"].sum()
    early = c[c["year"] <= cutoff_year].groupby("iso_code")["delta_intensity"].sum()
    out = pd.DataFrame({"efficiency_effect_mt": total, "early_share": early / total}).dropna()
    names = panel[["iso_code", "country"]].drop_duplicates().set_index("iso_code")["country"]
    out = out[out["efficiency_effect_mt"] <= -min_effect_mt].join(names)
    return out.sort_values("early_share", ascending=False).reset_index()


def run_audit(panel: pd.DataFrame, n_boot: int = 200) -> list:
    return [claim_germany_ukraine(panel, n_boot), claim_efficiency_led(panel), claim_sustained_decoupling(panel),
            claim_offshoring_flag(panel), claim_coverage(panel)]
