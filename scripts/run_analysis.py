"""Run the full analysis offline and write the report, tables and figures.

    python -m scripts.run_analysis              # full run, about a minute
    python -m scripts.run_analysis --fast       # skips placebo inference
    python -m scripts.run_analysis --refresh    # re-download OWID first (explicit)

Every number in outputs/analysis_report.md is computed in this run.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.audit.claims import efficiency_front_load, run_audit  # noqa: E402
from src.causal.synthetic_control import fit_synthetic, outcome_matrix, placebo_test  # noqa: E402
from src.data.loader import MANIFEST_PATH, load_panel, refresh_from_owid  # noqa: E402
from src.decoupling.classifier import annual_states, find_spells, income_groups, spell_survival  # noqa: E402
from src.kaya.decomposition import EFFECTS, chain_lmdi, driver_stability, endpoint_lmdi  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "figures"
# Well-known policy dates picked by the analyst. Japan 2011 is a positive control: a large known shock.
CASES = [("GBR", 2013, "UK Carbon Price Support (Apr 2013)"), ("AUS", 2012, "Australia carbon price (Jul 2012)"),
         ("IRL", 2010, "Ireland carbon tax (2010)"), ("JPN", 2011, "Japan nuclear shutdowns after Fukushima (2011, positive control)")]


def synthetic_results(panel: pd.DataFrame, fast: bool) -> list:
    Y = outcome_matrix(panel, "co2_per_energy")
    rows = []
    for iso, year, label in CASES:
        if fast:
            fit = fit_synthetic(Y, iso, year)
            res = {"fit": fit, "p_value_gap": float("nan"), "n_placebos": 0, "standardised_gap": fit.mean_post_gap / max(fit.pre_rmspe, 1e-9)}
        else:
            res = placebo_test(Y, iso, year)
        rows.append({"iso": iso, "year": year, "label": label, "effect_pct": float(np.expm1(res["fit"].mean_post_gap) * 100),
                     "pre_rmspe": res["fit"].pre_rmspe, "std_gap": res["standardised_gap"], "p_value": res["p_value_gap"],
                     "n_placebos": res["n_placebos"], "fit": res["fit"]})
    return rows


def build_report(r: dict) -> str:
    c = {x["id"]: x for x in r["claims"]}
    c1, c2, c3, c4, c5 = (c[k]["evidence"] for k in ["C1", "C2", "C3", "C4", "C5"])
    surv = r["survival"]
    L = ["# CO2 Trace: audited carbon accounting, 1990-2022", "",
         f"Data: committed OWID snapshot, {c5['countries_in_panel']} countries (manifest sha256 `{r['sha256'][:12]}`). "
         "Every number below is computed by `python -m scripts.run_analysis`; none is typed by hand.", "",
         "## 1. Audit of the v1 report", "", "| # | v1 claim | verdict |", "|---|---|---|"]
    L += [f"| {x['id']} | {x['v1_claim']} | **{x['verdict']}** |" for x in r["claims"]]
    L += ["", "### C1: Germany and Ukraine",
          f"On the RMS distance between log-trajectories (1990 = 1), Germany is {c1['distance_ukraine']:.2f} from Ukraine, "
          f"{c1['distance_denmark']:.2f} from Denmark and {c1['distance_sweden']:.2f} from Sweden. Under a moving-block bootstrap over years "
          f"(k = {c1['k']}), Germany is co-clustered with Denmark in {c1['co_assign_denmark']:.0%} of draws, with Sweden in "
          f"{c1['co_assign_sweden']:.0%}, and with Ukraine in {c1['co_assign_ukraine']:.0%}. The v1 K-means run did put Germany and Ukraine "
          f"together, but Germany's nearest centre was only {c1['v1_centre_margin_pct']:.0f}% closer than the next, so the assignment was a boundary case, "
          f"and {c1['n_excluded_low_base']} near-zero emitters were excluded here because their 1990-based ratios are not meaningful.", "",
          "### C2: efficiency versus fuel switching",
          f"Of {c2['n_reducers']} countries whose CO2 fell, {c2['efficiency_led_share']:.0%} had efficiency as their larger reducing effect; "
          f"{c2['p05']:.0%} to {c2['p95']:.0%} under 3% log-normal noise on all inputs. This is a sensitivity range, not a sampling confidence interval.", "",
          "### C3: how long does absolute decoupling last?",
          f"Strictly consecutive strong-decoupling spells: {c3['strict_spells']}, median length {c3['strict_median_length']:.0f} year, "
          f"{c3['strict_share_one_year']:.0%} last a single year, longest {c3['strict_longest_years']} years. Allowing one interrupted year, the longest "
          f"spell becomes {c3['tolerant_longest_years']} years. The claim holds only under the strict definition.", "",
          "### C4: the offshoring flag",
          f"{c4['n_flagged']} of {c4['n_countries']} countries meet the v1 rule (energy intensity down more than 30%, carbon intensity of energy down less than 10%). "
          f"{c4['n_flagged_with_rising_co2_per_capita']} of them ({c4['share_rising']:.0%}) have rising CO2 per capita and a median GDP per capita growth of "
          f"{c4['median_gdp_per_capita_growth_pct_flagged']:.0f}%: fast growth lowers energy per GDP mechanically. The flag does not isolate offshoring and is not used here.", "",
          "### C5: coverage",
          f"The panel has {c5['countries_in_panel']} countries; {c5['countries_in_endpoint_decomposition']} have positive values at both endpoints and "
          f"{c5['countries_in_chain_decomposition']} in every year, so the chain-linked decomposition covers the smaller set.", "",
          "## 2. Where emissions changed: chain-linked LMDI", "",
          "Year-by-year LMDI closes exactly (largest residual "
          f"{r['max_residual']:.1e} Mt). Its totals differ from the two-endpoint version because each year is weighted by that year's emissions. The eight largest differences:", "",
          "| Country | Efficiency, endpoint (Mt) | Efficiency, chain (Mt) | Difference (Mt) |", "|---|---|---|---|"]
    L += [f"| {x.country} | {x.endpoint:+.0f} | {x.chain:+.0f} | {x.diff:+.0f} |" for x in r["path_gap"].head(8).itertuples()]
    L += ["", f"For large reducers, the share of the cumulative efficiency effect that fell in 1991-1998: "
          + ", ".join(f"{x.country} {x.early_share:.0%}" for x in r["front"].head(5).itertuples())
          + f"; for comparison United Kingdom {r['front'].set_index('iso_code').loc['GBR','early_share']:.0%}. "
          "A front-loaded efficiency effect is consistent with transition-era restructuring, but this panel has no sector split and cannot separate it from technology.", "",
          "## 3. Do decoupling spells last? Survival analysis", "",
          f"{surv['n_spells']} strong-decoupling spells, right-censored where a spell was still running in 2022. "
          f"Kaplan-Meier survival past one year: high-income {r['km1']['high']:.0%}, middle {r['km1']['middle']:.0%}, low {r['km1']['low']:.0%} (1990 GDP per capita tertiles). "
          f"Log-rank high vs low: chi2 = {surv['logrank_high_vs_low']['chi2']:.1f}, p = {surv['logrank_high_vs_low']['p_value']:.4f}.", "",
          "## 4. Synthetic control: what can and cannot be said", "",
          "Outcome: log carbon intensity of energy. Donors: the 15 countries closest before the policy, weights on the probability simplex, fit scored on the last five pre-policy years. "
          "Cases and years are well-known policy dates picked by the analyst; point estimates were seen before this report was written, so inference is exploratory. "
          "p-values rank the standardised post-policy gap among in-space placebos.", "",
          "| Case | Effect vs synthetic | Held-out pre-fit RMSPE | Placebo p |", "|---|---|---|---|"]
    L += [f"| {x['label']} | {x['effect_pct']:+.1f}% | {x['pre_rmspe']:.3f} | {('n/a' if np.isnan(x['p_value']) else format(x['p_value'], '.2f'))} |" for x in r["synthetic"]]
    L += ["", "Japan 2011 is a positive control: a large, known increase in carbon intensity. If the test cannot flag that at conventional levels, it has low power here, "
          "and a non-significant UK, Australian or Irish estimate is not evidence of no effect.", "",
          "## Limitations", "",
          "- CO2 is territorial; consumption-based emissions and trade flows are not in the panel.",
          "- Kaya LMDI is accounting. Efficiency mixes technology, structure and offshoring.",
          "- Spell and distance results depend on stated definitions (strict spells, 4x winsorisation, k chosen by silhouette).",
          "- Synthetic control is exploratory: few treated units, short pre-periods, and EU donors share common policy shocks.", ""]
    return "\n".join(L)


def make_figures(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    t = r["claims"][0]["table"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(t["peer"], t["distance"], color="#264653"); ax[0].set_title("Distance from Germany (log-trajectory RMS)"); ax[0].invert_yaxis()
    ax[1].barh(t["peer"], t["co_assignment"], color="#2a9d8f"); ax[1].set_xlim(0, 1.05); ax[1].set_title("Bootstrap co-assignment with Germany"); ax[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(FIG / "claim_germany_ukraine.png", dpi=140); plt.close(fig)

    g = r["path_gap"].head(8)
    fig, ax = plt.subplots(figsize=(8, 4.5)); y = np.arange(len(g))
    ax.barh(y - 0.2, g["endpoint"], 0.4, label="endpoint", color="#e76f51"); ax.barh(y + 0.2, g["chain"], 0.4, label="chain-linked", color="#264653")
    ax.set_yticks(y); ax.set_yticklabels(g["country"]); ax.invert_yaxis(); ax.set_xlabel("Efficiency effect, Mt CO2"); ax.legend(); ax.set_title("Endpoint LMDI hides path dependence")
    fig.tight_layout(); fig.savefig(FIG / "chain_vs_endpoint.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for grp, color in [("high", "#264653"), ("middle", "#e9c46a"), ("low", "#e76f51")]:
        c = r["survival"]["curves"][grp]; ax.step(c["t"], c["survival"], where="post", label=f"{grp} income", color=color)
    ax.set_xlabel("Years in strong decoupling"); ax.set_ylabel("Share of spells surviving"); ax.legend(); ax.set_title("Decoupling spells are short (Kaplan-Meier, censored)")
    fig.tight_layout(); fig.savefig(FIG / "spell_survival.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7)); Y = r["outcomes"]
    for ax, x in zip(axes.ravel(), r["synthetic"]):
        f = x["fit"]; actual = np.exp(Y[x["iso"]]); synth = np.exp(Y[x["iso"]] - f.gap)
        ax.plot(actual.index, actual, color="#264653", label=x["iso"]); ax.plot(synth.index, synth, "--", color="#e76f51", label="synthetic")
        ax.axvline(x["year"], color="grey", lw=0.8); ax.set_title(x["label"], fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("Carbon intensity of energy (t CO2 per MWh)"); fig.tight_layout(); fig.savefig(FIG / "synthetic_control.png", dpi=140); plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download OWID data and rewrite the manifest")
    ap.add_argument("--fast", action="store_true", help="skip placebo inference and use fewer draws")
    args = ap.parse_args(argv)

    panel = refresh_from_owid() if args.refresh else load_panel()
    OUT.mkdir(exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
    sha = json.loads(MANIFEST_PATH.read_text())["sha256"]

    claims = run_audit(panel, n_boot=60 if args.fast else 200)
    chain, endpoint = chain_lmdi(panel), endpoint_lmdi(panel)
    tot = chain.groupby("iso_code")[EFFECTS].sum()
    gap = endpoint.set_index("iso_code").join(tot, rsuffix="_chain", how="inner")
    names = panel[["iso_code", "country"]].drop_duplicates().set_index("iso_code")["country"]
    gap = pd.DataFrame({"country": names.reindex(gap.index), "endpoint": gap["delta_intensity"], "chain": gap["delta_intensity_chain"]})
    gap["diff"] = gap["chain"] - gap["endpoint"]
    gap = gap.reindex(gap["diff"].abs().sort_values(ascending=False).index).reset_index()

    spells = find_spells(annual_states(panel))
    survival = spell_survival(spells, income_groups(panel))
    km1 = {g: float(c.loc[c["t"] == 1, "survival"].iloc[0]) for g, c in survival["curves"].items()}
    stab = driver_stability(panel, n_draws=60 if args.fast else 300)
    front = efficiency_front_load(panel)
    synthetic = synthetic_results(panel, args.fast)
    results = {"claims": claims, "sha256": sha, "path_gap": gap, "front": front, "survival": survival, "km1": km1,
               "synthetic": synthetic, "outcomes": outcome_matrix(panel, "co2_per_energy"),
               "max_residual": float(max(chain["residual"].abs().max(), endpoint["residual"].abs().max()))}

    (OUT / "analysis_report.md").write_text(build_report(results), encoding="utf-8")
    (OUT / "claims.json").write_text(json.dumps([{k: v for k, v in x.items() if k != "table"} for x in claims], indent=2, default=float) + "\n")
    stab.to_csv(OUT / "tables" / "kaya_driver_stability.csv", index=False); gap.to_csv(OUT / "tables" / "kaya_chain_vs_endpoint.csv", index=False)
    front.to_csv(OUT / "tables" / "efficiency_front_load.csv", index=False); spells.to_csv(OUT / "tables" / "decoupling_spells.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "fit"} for s in synthetic]).to_csv(OUT / "tables" / "synthetic_control.csv", index=False)
    make_figures(results)
    print("Wrote", OUT / "analysis_report.md")


if __name__ == "__main__":
    main()
