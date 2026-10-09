import numpy as np
import pandas as pd
import pytest

from src.kaya.decomposition import EFFECTS, chain_lmdi, endpoint_lmdi
from src.decoupling.classifier import annual_states, find_spells, kaplan_meier, tapio_state, logrank
from src.clustering.trajectory_cluster import log_trajectories, rms_distance, cluster_labels, consensus, compare
from src.causal.synthetic_control import simplex_least_squares, fit_synthetic


def test_lmdi_is_exact_and_chain_links_to_endpoint():
    years = np.arange(2000, 2006)
    rows = []
    for iso, mult in [("A", 1.0), ("B", 1.4)]:
        for i, y in enumerate(years):
            p = 10 * mult * (1.01 ** i)
            aff = 1000 * (1.02 ** i)
            intensity = 0.5 * (0.98 ** i)
            carbon = 0.06 * (0.99 ** i)
            rows.append({"iso_code": iso, "country": iso, "year": y,
                         "co2": p * aff * intensity * carbon,
                         "population": p, "gdp_per_capita": aff,
                         "energy_per_gdp": intensity, "co2_per_energy": carbon})
    panel = pd.DataFrame(rows)
    ep = endpoint_lmdi(panel, 2000, 2005).set_index("iso_code")
    chain = chain_lmdi(panel, 2000, 2005).groupby("iso_code")[EFFECTS + ["residual"]].sum()
    for iso in ep.index:
        assert abs(ep.loc[iso, "residual"]) < 1e-10
        for col in EFFECTS:
            assert chain.loc[iso, col] == pytest.approx(ep.loc[iso, col], rel=1e-10, abs=1e-10)
        assert abs(chain.loc[iso, "residual"]) < 1e-10


def test_tapio_eight_states_and_flat_case():
    cases = [
        ("strong_decoupling", -0.02, 0.03),
        ("weak_decoupling", 0.02, 0.04),
        ("expansive_coupling", 0.03, 0.03),
        ("expansive_negative_decoupling", 0.06, 0.03),
        ("strong_negative_decoupling", 0.02, -0.03),
        ("weak_negative_decoupling", -0.02, -0.04),
        ("recessive_coupling", -0.03, -0.03),
        ("recessive_decoupling", -0.06, -0.03),
        ("flat", 0.01, 0.0),
    ]
    for expected, dc, dg in cases:
        assert tapio_state(dc, dg) == expected


def test_spell_boundaries_and_right_censoring():
    states = pd.DataFrame({"iso_code": ["A"] * 8, "country": ["A"] * 8,
                           "year": range(2000, 2008),
                           "tapio_state": ["strong_decoupling"] * 3 + ["weak_decoupling"] + ["strong_decoupling"] * 4})
    spells = find_spells(states, last_year=2007)
    assert list(spells["length"]) == [3, 4]
    assert list(spells["ongoing"]) == [False, True]
    bridged = find_spells(states, gap_tolerance=1, last_year=2007)
    assert list(bridged["length"]) == [8]
    assert bool(bridged["ongoing"].iloc[0])


def test_kaplan_meier_keeps_censored_observations_in_risk_set():
    km = kaplan_meier([1, 2, 3], [1, 1, 0])
    assert km.loc[km.t == 1, "survival"].iloc[0] == pytest.approx(2 / 3)
    assert km.loc[km.t == 2, "survival"].iloc[0] == pytest.approx(1 / 3)
    assert km.loc[km.t == 3, "events"].iloc[0] == 0
    assert logrank([1, 2, 3], [1, 1, 0], [2, 3, 4], [1, 1, 1])["p_value"] < 1


def test_log_trajectory_distance_is_symmetric_and_consensus_reports_boundary():
    panel = pd.DataFrame({"iso_code": np.repeat(["A", "B", "C"], 3),
                          "year": list(range(2000, 2003)) * 3,
                          "co2_per_capita": [1, 2, 4, 1, 1.5, 2, 1, 0.5, 0.25]})
    L, excluded = log_trajectories(panel, base_year=2000, min_start=0.1)
    assert not excluded
    D = rms_distance(L)
    assert D.loc["A", "B"] == pytest.approx(D.loc["B", "A"])
    assert D.loc["A", "B"] > D.loc["A", "A"]
    co = consensus(L, 2, n_boot=30, block=1, seed=4)
    assert np.allclose(np.diag(co), 1.0)
    assert compare("A", ["B", "C"], D, co).iloc[0].peer == "B"


def test_simplex_synthetic_control_recovers_treated_series():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4))
    w0 = np.array([0.1, 0.2, 0.3, 0.4])
    y = X @ w0
    w = simplex_least_squares(X, y)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    assert np.sqrt(np.mean((X @ w - y) ** 2)) < 1e-4


def test_synthetic_control_pre_fit_uses_held_out_pre_years():
    rng = np.random.default_rng(9)
    years = np.arange(2000, 2020)
    donor = rng.normal(0, 0.01, (20, 6)).cumsum(axis=0)
    weights = np.array([0.1, 0.2, 0.15, 0.25, 0.1, 0.2])
    treated = donor @ weights
    treated[15:] -= 0.2
    frame = pd.DataFrame(np.column_stack([treated, donor]), index=years,
                         columns=["T", *[f"D{i}" for i in range(6)]])
    fit = fit_synthetic(frame, "T", 2015, k=6, holdout_years=5)
    assert fit.pre_rmspe < 0.05
    assert fit.mean_post_gap < -0.1
    assert fit.weights.sum() == pytest.approx(1.0)
