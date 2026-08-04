"""
Kaya Identity Decomposition
==============================
The Kaya Identity (Yoichi Kaya, 1990) expresses CO2 emissions as a product
of four factors:

    CO2 = P * (G/P) * (E/G) * (C/E)

Where:
    P    = Population
    G/P  = GDP per capita (affluence / prosperity)
    E/G  = Energy intensity of GDP (how much energy per unit of economic output)
    C/E  = Carbon intensity of energy (how dirty is the energy mix)

This is an accounting identity (true by definition, not a behavioural claim),
which means any change in CO2 can be EXACTLY decomposed into contributions
from each factor. It does NOT tell you WHY each factor changed (that requires
causal analysis), but it tells you WHERE change came from -- which is often
deeply counterintuitive.

DECOMPOSITION METHOD: Log-Mean Divisia Index (LMDI), additive form.
  - Chosen over simple ratio decomposition because LMDI produces no residual
    term (the decomposition is complete: all effects sum to the total CO2
    change) and handles zero/negative values more gracefully.
  - LMDI additive decomposition (Ang, 2005):
    ΔTCO2 = Δ_pop + Δ_aff + Δ_int + Δ_mix
    where each term is:
    Δ_factor = [L(C_T, C_0)] * ln(factor_T / factor_0)
    and L(x,y) = (x-y) / (ln(x) - ln(y)) is the log-mean weight.

WHY THIS MATTERS (the humanly interesting question):
When we say "Germany reduced its CO2 by X% since 1990," how much of that is:
  (a) actual clean energy transition (C/E fell)
  (b) becoming a more energy-efficient economy (E/G fell)
  (c) a declining manufacturing base / economic structure shift (G/P grew less
      or manufacturing share fell -- captured partially in E/G)
  (d) demographic stagnation (population barely grew, so P effect is small)

A country can achieve massive "CO2 reduction" entirely from (b), (c), (d)
while actually INCREASING its per-unit-energy carbon intensity or barely
building any renewables. The headline CO2 number masks all of this.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class KayaDecomp:
    iso_code: str
    country: str
    year_start: int
    year_end: int
    co2_start: float
    co2_end: float
    co2_change: float          # absolute change Mt CO2
    co2_pct_change: float      # percentage change
    delta_population: float    # Mt CO2 change attributable to population growth
    delta_affluence: float     # ... to GDP/capita growth (income effect)
    delta_intensity: float     # ... to energy intensity of GDP (efficiency)
    delta_carbon_mix: float    # ... to carbon intensity of energy (fuel switching)
    check_residual: float      # should be near zero (LMDI completeness check)

    @property
    def dominant_driver(self) -> str:
        contributions = {
            "population": abs(self.delta_population),
            "affluence": abs(self.delta_affluence),
            "energy_efficiency": abs(self.delta_intensity),
            "fuel_switch": abs(self.delta_carbon_mix),
        }
        return max(contributions, key=contributions.get)

    @property
    def reduction_breakdown_pct(self) -> dict:
        """
        For countries with any net CO2 change: signed contribution of each
        factor as a % of the total ABSOLUTE change, signed so negative =
        CO2-reducing contribution and positive = CO2-increasing contribution.
        Uses the total ABSOLUTE change as denominator so values can exceed
        100% in magnitude (when opposing forces more than cancel).

        Interpretation example for Germany (-36.7% net):
          affluence = +150% means income growth ADDED 1.5x the net reduction
          efficiency = -187% means efficiency REMOVED 1.87x the net reduction
          Net = -37% -> efficiency and fuel switching overcame affluence uplift
        """
        if abs(self.co2_change) < 1e-6:
            return {}
        denom = abs(self.co2_change)
        return {
            "fuel_switch": round(self.delta_carbon_mix / denom * 100, 1),
            "energy_efficiency": round(self.delta_intensity / denom * 100, 1),
            "affluence": round(self.delta_affluence / denom * 100, 1),
            "population": round(self.delta_population / denom * 100, 1),
        }


def _log_mean(x: float, y: float) -> float:
    """Log-mean weight for LMDI. Handles x==y exactly."""
    if abs(x - y) < 1e-12:
        return x
    if x <= 0 or y <= 0:
        return np.nan
    return (x - y) / (np.log(x) - np.log(y))


def _lmdi_additive(co2_0: float, co2_T: float,
                    p_0: float, p_T: float,
                    gpop_0: float, gpop_T: float,
                    egdp_0: float, egdp_T: float,
                    ce_0: float, ce_T: float) -> tuple:
    """
    LMDI additive decomposition.
    All inputs must be positive; NaN propagates.
    Returns (delta_pop, delta_aff, delta_int, delta_mix, residual).
    """
    if any(v is None or np.isnan(v) or v <= 0 for v in
           [co2_0, co2_T, p_0, p_T, gpop_0, gpop_T, egdp_0, egdp_T, ce_0, ce_T]):
        return (np.nan,) * 5

    L = _log_mean(co2_T, co2_0)
    d_pop = L * np.log(p_T / p_0)
    d_aff = L * np.log(gpop_T / gpop_0)
    d_int = L * np.log(egdp_T / egdp_0)
    d_mix = L * np.log(ce_T / ce_0)
    residual = (co2_T - co2_0) - (d_pop + d_aff + d_int + d_mix)
    return d_pop, d_aff, d_int, d_mix, residual


def decompose_country(df: pd.DataFrame, iso: str,
                       year_start: int = 1990, year_end: int = 2022) -> Optional[KayaDecomp]:
    sub = df[(df["iso_code"] == iso) & (df["year"].isin([year_start, year_end]))].copy()
    if len(sub) < 2:
        return None

    r0 = sub[sub["year"] == year_start].iloc[0]
    rT = sub[sub["year"] == year_end].iloc[0]

    needed = ["co2", "population", "gdp_per_capita", "energy_per_gdp", "co2_per_energy"]
    if any(pd.isna(r0[c]) or pd.isna(rT[c]) or r0[c] <= 0 or rT[c] <= 0 for c in needed):
        return None

    d_pop, d_aff, d_int, d_mix, resid = _lmdi_additive(
        r0["co2"], rT["co2"],
        r0["population"], rT["population"],
        r0["gdp_per_capita"], rT["gdp_per_capita"],
        r0["energy_per_gdp"], rT["energy_per_gdp"],
        r0["co2_per_energy"], rT["co2_per_energy"],
    )
    return KayaDecomp(
        iso_code=iso, country=r0["country"],
        year_start=year_start, year_end=year_end,
        co2_start=r0["co2"], co2_end=rT["co2"],
        co2_change=rT["co2"] - r0["co2"],
        co2_pct_change=(rT["co2"] - r0["co2"]) / r0["co2"] * 100,
        delta_population=d_pop, delta_affluence=d_aff,
        delta_intensity=d_int, delta_carbon_mix=d_mix,
        check_residual=resid,
    )


def decompose_all(df: pd.DataFrame, year_start: int = 1990,
                   year_end: int = 2022) -> pd.DataFrame:
    results = []
    for iso in df["iso_code"].unique():
        d = decompose_country(df, iso, year_start, year_end)
        if d is not None:
            results.append({
                "iso_code": d.iso_code, "country": d.country,
                "co2_start": d.co2_start, "co2_end": d.co2_end,
                "co2_change_mt": d.co2_change, "co2_pct_change": d.co2_pct_change,
                "delta_population": d.delta_population, "delta_affluence": d.delta_affluence,
                "delta_intensity": d.delta_intensity, "delta_carbon_mix": d.delta_carbon_mix,
                "check_residual": d.check_residual, "dominant_driver": d.dominant_driver,
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.loader import build_panel, load_raw
    co2_raw, energy_raw = load_raw()
    panel = build_panel(co2_raw, energy_raw)
    decomp = decompose_all(panel)

    print(f"Decomposed {len(decomp)} countries.")
    print(f"Max LMDI residual (should be ~0): {decomp['check_residual'].abs().max():.4f}")

    reducers = decomp[decomp["co2_change_mt"] < 0].sort_values("co2_pct_change")
    print(f"\nCountries that REDUCED emissions 1990-2022: {len(reducers)}")
    print("\nTop 10 reducers (% change):")
    for _, row in reducers.head(10).iterrows():
        d = decompose_country(panel, row["iso_code"])
        bkd = d.reduction_breakdown_pct
        print(f"  {row['country']:25s} {row['co2_pct_change']:+6.1f}%  "
              f"fuel_switch={bkd.get('fuel_switch',0):.0f}%  "
              f"efficiency={bkd.get('energy_efficiency',0):.0f}%  "
              f"affluence={bkd.get('affluence',0):.0f}%")
