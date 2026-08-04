"""
Decoupling Classification
============================
The UNEP/academic literature distinguishes:

  1. NO DECOUPLING:       Both GDP per capita AND CO2 per capita increasing
  2. RELATIVE DECOUPLING: CO2/cap increasing, but slower than GDP/cap
                           (OR CO2/cap falling while GDP/cap also falling,
                           i.e. everything is shrinking -- contested whether
                           this counts as "decoupling" at all)
  3. ABSOLUTE DECOUPLING: GDP/cap increasing AND CO2/cap decreasing
                           This is the only state consistent with meeting
                           climate targets while maintaining prosperity.
  4. DEGROWTH REDUCTION:  CO2/cap falling but so is GDP/cap (contraction)
                           Emissions fell but not because of clean technology.

The "decoupling question" is: how many countries have achieved ABSOLUTE
decoupling, for how long, and was it sustained?

This is where the genuinely counterintuitive findings are:
- Many celebrated success stories are SHORT SPURTS of absolute decoupling
  followed by reversals (often during economic booms when cheap fossil
  energy is used to sustain growth)
- Some apparent absolute decouplements are actually emissions OFFSHORING:
  the country moved manufacturing abroad, so territorial CO2 fell while
  consumption-based emissions stayed flat or rose. We flag this using the
  ratio of CO2 per capita to energy intensity -- if energy intensity falls
  fast without corresponding fuel switching, that is more consistent with
  structural economic change than genuine decarbonization.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import pandas as pd


class DecouplingState(Enum):
    ABSOLUTE = "absolute_decoupling"     # GDP/cap up, CO2/cap down
    RELATIVE = "relative_decoupling"     # both up, but CO2 grows slower
    NO_DECOUPLING = "no_decoupling"      # CO2 grows at least as fast as GDP
    DEGROWTH = "degrowth_reduction"      # both down (recession-driven)
    UNCLEAR = "unclear"                  # GDP flat or insufficient data


@dataclass
class CountryDecoupling:
    iso_code: str
    country: str
    # 10-year rolling window classification
    state_by_decade: dict   # e.g. {"1990-2000": DecouplingState.ABSOLUTE, ...}
    # Sustained absolute decoupling: consecutive years where both conditions hold
    years_absolute: int
    longest_abs_run: int
    currently_absolute: bool   # 2018-2022 (latest 5 years)
    # Flags
    suspect_offshoring: bool   # efficiency improved fast but fuel mix barely changed
    suspect_deindustrialisation: bool   # energy intensity fell faster than CO2 intensity
    # Terminal values
    gdp_per_cap_growth_pct_1990_2022: float
    co2_per_cap_change_pct_1990_2022: float


def _classify_window(gdp_growth: float, co2_growth: float,
                      threshold: float = 0.5) -> DecouplingState:
    """
    Classify a time window based on per-capita GDP and CO2 growth rates.
    `threshold`: minimum GDP growth to distinguish relative decoupling from
    degrowth (default 0.5% annualised over the window).
    """
    if np.isnan(gdp_growth) or np.isnan(co2_growth):
        return DecouplingState.UNCLEAR
    if gdp_growth > threshold:
        if co2_growth < 0:
            return DecouplingState.ABSOLUTE
        elif co2_growth < gdp_growth:
            return DecouplingState.RELATIVE
        else:
            return DecouplingState.NO_DECOUPLING
    elif gdp_growth < -threshold:
        return DecouplingState.DEGROWTH
    else:
        return DecouplingState.UNCLEAR


def classify_country(df: pd.DataFrame, iso: str) -> CountryDecoupling:
    sub = df[df["iso_code"] == iso].sort_values("year")
    if len(sub) < 10:
        return None
    country_name = sub["country"].iloc[0]

    sub = sub.dropna(subset=["gdp_per_capita", "co2_per_capita"])
    if len(sub) < 10:
        return None

    sub = sub.set_index("year")

    # Per-year absolute decoupling flag
    gdp_growth_rate = sub["gdp_per_capita"].pct_change()
    co2_growth_rate = sub["co2_per_capita"].pct_change()
    abs_flag = (gdp_growth_rate > 0) & (co2_growth_rate < 0)

    years_absolute = int(abs_flag.sum())
    # Longest consecutive absolute decoupling run
    longest = 0
    current_run = 0
    for v in abs_flag:
        if v:
            current_run += 1
            longest = max(longest, current_run)
        else:
            current_run = 0

    # Is the country currently in absolute decoupling? (last 5 years as a period)
    recent = sub.loc[sub.index >= 2018] if 2018 in sub.index or sub.index.max() >= 2018 else pd.DataFrame()
    currently_abs = False
    if len(recent) >= 3:
        g_gdp = (recent["gdp_per_capita"].iloc[-1] / recent["gdp_per_capita"].iloc[0]) - 1
        g_co2 = (recent["co2_per_capita"].iloc[-1] / recent["co2_per_capita"].iloc[0]) - 1
        currently_abs = (g_gdp > 0) and (g_co2 < 0)

    # Decade-by-decade classification
    decades = [
        ("1990-2000", 1990, 2000),
        ("2000-2010", 2000, 2010),
        ("2010-2022", 2010, 2022),
    ]
    state_by_decade = {}
    for label, y0, y1 in decades:
        if y0 in sub.index and y1 in sub.index:
            g_gdp = (sub.loc[y1, "gdp_per_capita"] / sub.loc[y0, "gdp_per_capita"]) ** (1 / (y1 - y0)) - 1
            g_co2 = (sub.loc[y1, "co2_per_capita"] / sub.loc[y0, "co2_per_capita"]) ** (1 / (y1 - y0)) - 1
            state_by_decade[label] = _classify_window(g_gdp * 100, g_co2 * 100)
        else:
            closest0 = sub.loc[sub.index <= y0 + 2].index.max() if any(sub.index <= y0 + 2) else None
            closest1 = sub.loc[sub.index >= y1 - 2].index.min() if any(sub.index >= y1 - 2) else None
            if closest0 and closest1 and closest0 < closest1:
                g_gdp = (sub.loc[closest1, "gdp_per_capita"] / sub.loc[closest0, "gdp_per_capita"]) ** (1 / (closest1 - closest0)) - 1
                g_co2 = (sub.loc[closest1, "co2_per_capita"] / sub.loc[closest0, "co2_per_capita"]) ** (1 / (closest1 - closest0)) - 1
                state_by_decade[label] = _classify_window(g_gdp * 100, g_co2 * 100)
            else:
                state_by_decade[label] = DecouplingState.UNCLEAR

    # Offshoring/deindustrialisation suspects:
    # If energy intensity fell much faster than CO2 intensity, the drop in emissions
    # is likely structural (manufacturing moved) rather than clean-energy driven
    e_cols = sub["energy_per_gdp"].dropna()
    c_cols = sub["co2_per_energy"].dropna()
    if len(e_cols) >= 2 and len(c_cols) >= 2:
        e_drop = (e_cols.iloc[-1] / e_cols.iloc[0]) - 1
        c_drop = (c_cols.iloc[-1] / c_cols.iloc[0]) - 1
        suspect_offshoring = (e_drop < -0.30) and (c_drop > -0.10)
        suspect_deindustrialisation = (e_drop < -0.40) and (e_drop / (c_drop + 0.001) > 3.0)
    else:
        suspect_offshoring = False
        suspect_deindustrialisation = False

    s0 = sub.iloc[0]
    sT = sub.iloc[-1]
    gdp_change = (sT["gdp_per_capita"] / s0["gdp_per_capita"] - 1) * 100 if s0["gdp_per_capita"] > 0 else np.nan
    co2_change = (sT["co2_per_capita"] / s0["co2_per_capita"] - 1) * 100 if s0["co2_per_capita"] > 0 else np.nan

    return CountryDecoupling(
        iso_code=iso, country=country_name, state_by_decade=state_by_decade,
        years_absolute=years_absolute, longest_abs_run=longest,
        currently_absolute=currently_abs, suspect_offshoring=suspect_offshoring,
        suspect_deindustrialisation=suspect_deindustrialisation,
        gdp_per_cap_growth_pct_1990_2022=gdp_change,
        co2_per_cap_change_pct_1990_2022=co2_change,
    )


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for iso in df["iso_code"].unique():
        r = classify_country(df, iso)
        if r is None:
            continue
        records.append({
            "iso_code": r.iso_code, "country": r.country,
            "years_absolute_decoupling": r.years_absolute,
            "longest_consecutive_abs_run": r.longest_abs_run,
            "currently_absolute_2018_2022": r.currently_absolute,
            "decade_1990_2000": r.state_by_decade.get("1990-2000", DecouplingState.UNCLEAR).value,
            "decade_2000_2010": r.state_by_decade.get("2000-2010", DecouplingState.UNCLEAR).value,
            "decade_2010_2022": r.state_by_decade.get("2010-2022", DecouplingState.UNCLEAR).value,
            "suspect_offshoring": r.suspect_offshoring,
            "suspect_deindustrialisation": r.suspect_deindustrialisation,
            "gdp_per_cap_growth_pct": r.gdp_per_cap_growth_pct_1990_2022,
            "co2_per_cap_change_pct": r.co2_per_cap_change_pct_1990_2022,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    from src.data.loader import build_panel, load_raw
    co2_raw, energy_raw = load_raw()
    panel = build_panel(co2_raw, energy_raw)
    dc = classify_all(panel)

    print("=== Decoupling Status (2010-2022) across all countries ===")
    print(dc["decade_2010_2022"].value_counts().to_string())

    truly_abs = dc[dc["currently_absolute_2018_2022"] & (dc["co2_per_cap_change_pct"] < 0) & (dc["gdp_per_cap_growth_pct"] > 10)]
    print(f"\nCurrently in absolute decoupling AND grew richer AND reduced CO2/cap: {len(truly_abs)} countries")
    print(truly_abs[["country", "gdp_per_cap_growth_pct", "co2_per_cap_change_pct",
                       "suspect_offshoring", "longest_consecutive_abs_run"]].sort_values("co2_per_cap_change_pct").to_string(index=False))

    suspect = dc[dc["suspect_offshoring"]]
    print(f"\nCountries flagged as SUSPECT OFFSHORING ({len(suspect)}):")
    print(suspect[["country", "gdp_per_cap_growth_pct", "co2_per_cap_change_pct"]].sort_values("co2_per_cap_change_pct").to_string(index=False))
