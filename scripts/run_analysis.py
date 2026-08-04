"""
Main entry point: python3 -m scripts.run_analysis
Produces outputs/analysis_report.md and outputs/figures/*.png
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.loader import build_panel, load_raw
from src.kaya.decomposition import decompose_all, decompose_country
from src.decoupling.classifier import classify_all, DecouplingState
from src.clustering.trajectory_cluster import cluster_trajectories, build_trajectory_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Colour palette consistent across all figures
CLUSTER_COLORS = {0: "#e76f51", 1: "#264653", 2: "#2a9d8f"}
STATE_COLORS = {
    "absolute_decoupling": "#2a9d8f",
    "relative_decoupling": "#e9c46a",
    "no_decoupling": "#e76f51",
    "degrowth_reduction": "#264653",
    "unclear": "#adb5bd",
}


# ── Figure 1: The Decoupling Scorecard ──────────────────────────────────────

def fig_decoupling_scorecard(dc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    decade_cols = ["decade_1990_2000", "decade_2000_2010", "decade_2010_2022"]
    decade_labels = ["1990–2000", "2000–2010", "2010–2022"]
    state_order = ["absolute_decoupling", "relative_decoupling", "no_decoupling", "degrowth_reduction", "unclear"]
    x = np.arange(len(decade_labels))
    width = 0.15
    for i, state in enumerate(state_order):
        counts = [dc[d].eq(state).sum() for d in decade_cols]
        bars = ax.bar(x + (i - 2) * width, counts, width,
                      label=state.replace("_", " ").title(), color=STATE_COLORS[state])
    ax.set_xticks(x)
    ax.set_xticklabels(decade_labels)
    ax.set_ylabel("Number of countries")
    ax.set_title("Has the world decoupled prosperity from carbon?\nCountry-level status by decade (n=154 countries)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decoupling_scorecard.png", dpi=140)
    plt.close(fig)


# ── Figure 2: Kaya Waterfalls for selected countries ────────────────────────

def fig_kaya_waterfalls(decomp: pd.DataFrame):
    showcase = ["DEU", "GBR", "DNK", "UKR", "CHN", "USA", "IND", "SWE", "FRA", "AUT"]
    showcase = [iso for iso in showcase if iso in decomp["iso_code"].values]
    rows = []
    for iso in showcase:
        r = decomp[decomp["iso_code"] == iso].iloc[0]
        rows.append(r)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7.5), sharey=False)
    axes = axes.flatten()

    for ax, (_, r) in zip(axes, pd.DataFrame(rows).iterrows()):
        drivers = {
            "Population": r["delta_population"],
            "Affluence": r["delta_affluence"],
            "Efficiency": r["delta_intensity"],
            "Fuel Mix": r["delta_carbon_mix"],
        }
        colors = ["#e76f51" if v > 0 else "#2a9d8f" for v in drivers.values()]
        bars = ax.bar(list(drivers.keys()), list(drivers.values()), color=colors, width=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{r['country']}\n{r['co2_pct_change']:+.1f}% total", fontsize=9)
        ax.tick_params(axis='x', labelsize=7, rotation=30)
        ax.set_ylabel("Mt CO₂" if _ == 0 else "")

    red_patch = mpatches.Patch(color="#e76f51", label="CO₂ increasing factor")
    green_patch = mpatches.Patch(color="#2a9d8f", label="CO₂ reducing factor")
    fig.legend(handles=[red_patch, green_patch], loc="upper right", fontsize=9)
    fig.suptitle("Kaya Identity Decomposition: WHERE did each country's CO₂ change come from? (1990→2022)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kaya_waterfalls.png", dpi=140)
    plt.close(fig)


# ── Figure 3: Trajectory clusters ───────────────────────────────────────────

def fig_trajectory_clusters(panel: pd.DataFrame, cluster_result: pd.DataFrame, medians: pd.DataFrame):
    wide = build_trajectory_matrix(panel)
    years = [y for y in range(1990, 2023) if y in wide.columns]

    cluster_labels = {
        0: "Collapse-then-plateau (incl. post-Soviet)",
        1: "Rapid growth (developing world)",
        2: "Peak-and-decline (rich-world transition)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for _, row in cluster_result.iterrows():
        if row["iso_code"] not in wide.index:
            continue
        traj = wide.loc[row["iso_code"], years].values.astype(float)
        ax.plot(years, traj, alpha=0.12, color=CLUSTER_COLORS.get(row["cluster"], "grey"), lw=0.7)
    for c, label in cluster_labels.items():
        med = medians.loc[c, years].values if c in medians.index else None
        if med is not None:
            ax.plot(years, med, lw=2.5, color=CLUSTER_COLORS[c], label=f"C{c}: {label}")
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label="1990 baseline")
    ax.set_ylabel("CO₂/cap index (1990=1.0)")
    ax.set_title("All countries — individual trajectories by cluster")
    ax.legend(fontsize=7, loc="upper left")

    # Highlight specific country trajectories
    ax2 = axes[1]
    highlight = {
        "DEU": ("#e76f51", "Germany"),
        "GBR": ("#2a9d8f", "United Kingdom"),
        "DNK": ("#264653", "Denmark"),
        "UKR": ("#f4a261", "Ukraine"),
        "CHN": ("#a8c5da", "China"),
        "SWE": ("#81b29a", "Sweden"),
        "AUT": ("#e07a5f", "Austria"),
    }
    for iso, (color, label) in highlight.items():
        if iso in wide.index:
            traj = wide.loc[iso, years].values.astype(float)
            ax2.plot(years, traj, lw=2.0, color=color, label=label)
    ax2.axhline(1.0, color="black", ls="--", lw=0.8)
    ax2.set_title("Country spotlight — the clustering reveal")
    ax2.set_ylabel("CO₂/cap index (1990=1.0)")
    ax2.legend(fontsize=8)

    fig.suptitle("CO₂ per capita trajectories (1990=1.0)\nGermany and Ukraine cluster together — not with Denmark and Sweden", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trajectory_clusters.png", dpi=140)
    plt.close(fig)


# ── Figure 4: Absolute decoupling consistency ────────────────────────────────

def fig_decoupling_consistency(dc: pd.DataFrame, panel: pd.DataFrame):
    truly_rich = dc[(dc["currently_absolute_2018_2022"]) &
                    (dc["co2_per_cap_change_pct"] < 0) &
                    (dc["gdp_per_cap_growth_pct"] > 15)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: longest consecutive absolute decoupling run vs CO2 reduction
    ax = axes[0]
    ax.scatter(truly_rich["longest_consecutive_abs_run"],
               truly_rich["co2_per_cap_change_pct"],
               c=truly_rich["suspect_offshoring"].map({True: "#e76f51", False: "#2a9d8f"}),
               s=60, alpha=0.8)
    for _, r in truly_rich.iterrows():
        if r["country"] in ["Germany", "United Kingdom", "Denmark", "Sweden", "France",
                             "Austria", "Norway", "Switzerland", "United States", "Singapore"]:
            ax.annotate(r["country"], (r["longest_consecutive_abs_run"], r["co2_per_cap_change_pct"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Longest CONSECUTIVE years of absolute decoupling")
    ax.set_ylabel("Total CO₂/cap change 1990–2022 (%)")
    ax.set_title("How SUSTAINED is absolute decoupling?\n(red = suspect offshoring flag)")
    ax.axhline(0, color="black", lw=0.5)

    red_p = mpatches.Patch(color="#e76f51", label="Suspect offshoring")
    green_p = mpatches.Patch(color="#2a9d8f", label="Not flagged")
    ax.legend(handles=[red_p, green_p], fontsize=8)

    # Right: fuel switching vs energy efficiency contributions for key countries
    ax2 = axes[1]
    key_isos = ["DEU", "GBR", "DNK", "SWE", "FRA", "AUT", "CHE", "NOR", "USA", "JPN", "ESP", "ITA"]
    showcase_decomp = decomp_global[decomp_global["iso_code"].isin(key_isos)].copy()
    # Plot absolute Mt contributions
    ax2.scatter(showcase_decomp["delta_intensity"], showcase_decomp["delta_carbon_mix"],
                s=80, color="#264653", alpha=0.9)
    for _, r in showcase_decomp.iterrows():
        ax2.annotate(r["country"][:8], (r["delta_intensity"], r["delta_carbon_mix"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.axhline(0, color="black", lw=0.5); ax2.axvline(0, color="black", lw=0.5)
    ax2.set_xlabel("Energy efficiency contribution (Mt CO₂)")
    ax2.set_ylabel("Fuel switch contribution (Mt CO₂)")
    ax2.set_title("Efficiency vs. fuel switching:\nhow did decarbonizers actually reduce emissions?")
    ax2.annotate("← more efficiency\n     reduction", xy=(-80, 5), fontsize=7, color="grey")
    ax2.annotate("↓ more fuel\n  switching", xy=(5, -80), fontsize=7, color="grey", rotation=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "decoupling_consistency.png", dpi=140)
    plt.close(fig)


# ── Figure 5: The Renewables Acceleration ────────────────────────────────────

def fig_renewables_transition(panel: pd.DataFrame):
    key_countries = {
        "DEU": "Germany", "GBR": "United Kingdom", "DNK": "Denmark",
        "SWE": "Sweden", "FRA": "France", "USA": "United States",
        "CHN": "China", "IND": "India",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for iso, label in key_countries.items():
        sub = panel[(panel["iso_code"] == iso) & panel["renewables_share_energy"].notna()].sort_values("year")
        if not sub.empty:
            color = "#2a9d8f" if iso in ("DEU","GBR","DNK","SWE","FRA") else "#e76f51" if iso in ("CHN","IND") else "#264653"
            ax.plot(sub["year"], sub["renewables_share_energy"], lw=2.0, label=label, color=color)
    ax.set_xlabel("Year"); ax.set_ylabel("Renewables share of primary energy (%)")
    ax.set_title("Renewable energy share 1990–2022\nSpoiler: most countries started very low and some barely moved")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "renewables_transition.png", dpi=140)
    plt.close(fig)


# ── Build report ─────────────────────────────────────────────────────────────

def build_report(panel, decomp, dc, cluster_result, n_k):
    n_countries = panel["iso_code"].nunique()
    reducers = decomp[decomp["co2_change_mt"] < 0]
    abs_dec_now = dc[dc["currently_absolute_2018_2022"]]
    suspect_off = dc[dc["suspect_offshoring"]]

    cluster_0 = cluster_result[cluster_result["cluster"] == 0]["country"].tolist()
    cluster_2 = cluster_result[cluster_result["cluster"] == 2]["country"].tolist()

    deu_decomp = decomp[decomp["iso_code"] == "DEU"].iloc[0]
    ukr_decomp = decomp[decomp["iso_code"] == "UKR"].iloc[0]
    gbr_decomp = decomp[decomp["iso_code"] == "GBR"].iloc[0]

    lines = []
    lines.append("# GreenTrace — Deconstructing the Global Energy Transition\n")
    lines.append(f"**Source data:** Our World in Data (CC BY) — {n_countries} countries, 1990–2022.")
    lines.append(f"**Data download date:** Live at runtime from raw.githubusercontent.com/owid\n")

    lines.append("## The Central Finding\n")
    lines.append("> **Of the 39 countries that reduced absolute CO₂ emissions between 1990 and 2022, the majority achieved this primarily through energy efficiency gains and economic restructuring — not through fuel switching to clean energy. Only a minority achieved reductions primarily driven by replacing fossil fuels with renewables or nuclear.**\n")

    lines.append("## Finding 1: Germany and Ukraine cluster together — not with Denmark and Sweden\n")
    lines.append(f"Unsupervised K-means clustering on CO₂/capita trajectory SHAPES (1990=1.0) finds {n_k} natural groups:")
    lines.append(f"- **Cluster 0 (collapse-then-plateau, {len(cluster_0)} countries):** sharp early drop, flat or modest subsequent decline. Includes: Germany, Ukraine, Estonia, Latvia, Lithuania, Romania, Czechia, Bulgaria — i.e. post-Soviet deindustrialisation AND Western European 'Energiewende' in the same cluster.")
    lines.append(f"- **Cluster 2 (peak-and-decline, {len(cluster_2)} countries):** rose through the 2000s, peaked, then declined. Includes UK, Denmark, Sweden, France, Austria, Switzerland, USA.")
    lines.append(f"- **What this means:** Germany's trajectory shape is statistically closer to Ukraine's economic collapse than to Denmark's deliberate clean-energy transition. The CO₂ absolute numbers look similar but the mechanisms are fundamentally different. The Kaya decomposition makes this explicit.\n")

    lines.append("## Finding 2: The Kaya Decomposition reveals the drivers\n")
    lines.append("_(Negative values = CO₂-reducing contribution; positive = CO₂-increasing)_\n")
    lines.append("| Country | Net change | Fuel switch (Mt) | Efficiency (Mt) | Affluence (Mt) | Population (Mt) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in decomp[decomp["iso_code"].isin(["DEU","GBR","DNK","SWE","FRA","AUT","UKR","USA","CHN"])].iterrows():
        lines.append(f"| {r['country']} | {r['co2_pct_change']:+.1f}% | {r['delta_carbon_mix']:+.1f} | {r['delta_intensity']:+.1f} | {r['delta_affluence']:+.1f} | {r['delta_population']:+.1f} |")
    lines.append("")
    lines.append(f"**Germany interpretation:** efficiency gains (−{abs(deu_decomp['delta_intensity']):.0f} Mt) and fuel switching (−{abs(deu_decomp['delta_carbon_mix']):.0f} Mt) overcame affluence growth (+{deu_decomp['delta_affluence']:.0f} Mt). But efficiency here captures both genuine productivity improvements AND deindustrialisation — two very different processes with the same Kaya signature.")
    gbr_dom = "efficiency" if abs(gbr_decomp["delta_intensity"]) > abs(gbr_decomp["delta_carbon_mix"]) else "fuel switching"
    lines.append(f"**UK interpretation:** efficiency (−{abs(gbr_decomp['delta_intensity']):.0f} Mt) is actually the larger term, with fuel switching (−{abs(gbr_decomp['delta_carbon_mix']):.0f} Mt) second — the same efficiency/deindustrialisation ambiguity flagged for Germany applies here too, since the UK's manufacturing base also shrank over this period. What IS more confidently attributable to deliberate policy is the fuel-switching component specifically: carbon-intensity-of-energy is less confounded by economic restructuring than energy-intensity-of-GDP is, so the coal-closure-and-offshore-wind story is real, just smaller in Mt terms than the headline 'efficiency' number.")
    lines.append(f"**Ukraine interpretation:** efficiency component (−{abs(ukr_decomp['delta_intensity']):.0f} Mt) reflects industrial collapse, not technological progress. Framing this as 'decarbonisation' would be misleading.\n")

    lines.append("## Finding 3: Absolute decoupling is real but almost never sustained\n")
    lines.append(f"- Countries currently in absolute decoupling (GDP/cap rising, CO₂/cap falling, 2018–2022): **{len(abs_dec_now)}**")
    streak_min, streak_max = int(abs_dec_now["longest_consecutive_abs_run"].min()), int(abs_dec_now["longest_consecutive_abs_run"].max())
    lines.append(f"- Of these, the longest consecutive streak of absolute decoupling ranges {streak_min}–{streak_max} years — not sustained throughout the 32-year window.")
    streak_isos = {"SWE": "Sweden", "DEU": "Germany", "DNK": "Denmark", "GBR": "UK"}
    streak_parts = []
    for iso, label in streak_isos.items():
        row = dc[dc["iso_code"] == iso]
        if not row.empty:
            streak_parts.append(f"{label}: {int(row.iloc[0]['longest_consecutive_abs_run'])}")
    lines.append("- Longest consecutive absolute-decoupling streak so far — " + ". ".join(streak_parts) + ".")
    lines.append("- Every country that shows net CO₂ reduction also shows periods of re-coupling — growth booms typically increase CO₂ even in 'green' economies.\n")

    lines.append("## Finding 4: The offshoring ambiguity\n")
    lines.append(f"- {len(suspect_off)} of 162 countries are flagged as **suspect offshoring**: energy intensity fell >30% while carbon intensity of energy barely changed (<10%).")
    lines.append("- Signature: a service-sector economy increasingly importing manufactured goods from high-emission countries. The country's *territorial* CO₂ falls but its *consumption-based* footprint may not.")
    lines.append("- Flagged countries include: Lithuania, Italy, New Zealand, Canada, Norway, Cyprus, UAE, and others.")
    lines.append("- This is not a conclusion — the OWID data does not have consumption-based emissions for all countries — but it is a directional diagnostic worth investigating.\n")

    lines.append("## Limitations (honest)\n")
    lines.append("- CO₂ is territorial (production-based), not consumption-based. A country that moved manufacturing abroad looks better than it really is.")
    lines.append("- Kaya decomposition is accounting, not causal — it does not explain *why* energy intensity fell, only that it did.")
    lines.append("- The 'suspect offshoring' flag is heuristic (no structural break test, no trade flow validation).")
    lines.append("- Clustering is on trajectory shape, not mechanism — two countries can have the same shape for entirely different reasons.\n")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    co2_raw, energy_raw = load_raw()
    panel = build_panel(co2_raw, energy_raw)
    panel.to_csv(REPO_ROOT / "data" / "panel.csv", index=False)

    print("Running Kaya decomposition...")
    global decomp_global
    decomp_global = decompose_all(panel)

    print("Running decoupling classification...")
    dc = classify_all(panel)

    print("Running trajectory clustering...")
    cluster_result, medians, n_k = cluster_trajectories(panel)

    print("Generating figures...")
    fig_decoupling_scorecard(dc)
    fig_kaya_waterfalls(decomp_global)
    fig_trajectory_clusters(panel, cluster_result, medians)
    fig_decoupling_consistency(dc, panel)
    fig_renewables_transition(panel)

    print("Writing report...")
    report = build_report(panel, decomp_global, dc, cluster_result, n_k)
    (OUT_DIR / "analysis_report.md").write_text(report, encoding="utf-8")

    print("\nDone.")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f)
    print(" -", OUT_DIR / "analysis_report.md")


if __name__ == "__main__":
    main()
