"""
Data Loader: OWID CO2 + Energy Datasets
==========================================
Downloads the two Our World in Data (OWID) open datasets directly from their
GitHub repository at runtime. No API key or authentication required.

Sources:
  https://github.com/owid/co2-data       (CO2, GDP, population, energy intensity)
  https://github.com/owid/energy-data    (energy mix detail, renewables, by source)

Both datasets are published under CC BY license. The data covers 1750–2024
for up to 270+ countries/regions. We restrict to:
  - ISO-coded sovereign countries only (removes regional aggregates like
    "Africa", "World", "High-income countries" which contaminate country-
    level analysis)
  - Years 1990–2022 (the policy-relevant modern era; pre-1990 data has
    significant quality variation across countries)
  - Countries with data on ALL four Kaya variables (co2, gdp, population,
    primary_energy_consumption) for >= 25 of the 32 years in scope

Quality notes on the underlying data:
  - GDP is in constant 2017 PPP international dollars (Penn World Tables
    lineage, corrected via World Bank)
  - CO2 is fossil fuel combustion + industrial processes; excludes LULUCF
    (land use change) unless suffixed _including_luc
  - energy_per_capita (kWh per person) is derived from primary energy
    consumption (total, all sources) divided by population
"""
import urllib.request
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

YEAR_MIN, YEAR_MAX = 1990, 2022
MIN_VALID_YEARS = 25

KAYA_COLS = ["co2", "gdp", "population", "primary_energy_consumption"]
ENERGY_EXTRA_COLS = [
    "renewables_share_energy", "fossil_share_energy",
    "solar_share_energy", "wind_share_energy",
    "coal_share_energy", "gas_share_energy", "oil_share_energy",
    "low_carbon_share_energy",
    "renewables_electricity", "electricity_generation",
]


def _download(url: str) -> pd.DataFrame:
    print(f"  Downloading {url.split('/')[-1]} ...")
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(content))


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    co2 = _download(CO2_URL)
    energy = _download(ENERGY_URL)
    return co2, energy


def build_panel(co2_raw: pd.DataFrame, energy_raw: pd.DataFrame) -> pd.DataFrame:
    # Filter to ISO-coded sovereign countries (drop regional aggregates)
    co2 = co2_raw.dropna(subset=["iso_code"]).copy()
    co2 = co2[~co2["iso_code"].str.startswith("OWID", na=False)]

    energy = energy_raw.dropna(subset=["iso_code"]).copy()
    energy = energy[~energy["iso_code"].str.startswith("OWID", na=False)]

    # Restrict to analysis window
    co2 = co2[(co2["year"] >= YEAR_MIN) & (co2["year"] <= YEAR_MAX)]
    energy = energy[(energy["year"] >= YEAR_MIN) & (energy["year"] <= YEAR_MAX)]

    # Select and merge
    co2_sel = co2[["iso_code", "country", "year"] + KAYA_COLS +
                   ["co2_per_capita", "co2_per_gdp", "energy_per_capita",
                    "share_global_co2", "cumulative_co2"]].copy()

    en_cols = ["iso_code", "year"] + [c for c in ENERGY_EXTRA_COLS if c in energy.columns]
    energy_sel = energy[en_cols].copy()

    panel = co2_sel.merge(energy_sel, on=["iso_code", "year"], how="left")

    # Compute energy intensity of GDP: primary energy (TWh) / GDP (2017 PPP $bn)
    # Primary energy in OWID is in TWh; GDP in 2017 international $ (not bn)
    # energy_per_gdp_kwh_per_dollar = primary_energy_consumption*1e9 / gdp
    panel["energy_per_gdp"] = (
        panel["primary_energy_consumption"] * 1e9 / panel["gdp"]
    )  # kWh per 2017 PPP $

    # Carbon intensity of energy: co2 (Mt) / primary energy (TWh) -> tonne CO2 / MWh
    # 1 Mt CO2 = 1e6 t, 1 TWh = 1e6 MWh
    panel["co2_per_energy"] = (
        panel["co2"] * 1e6 / (panel["primary_energy_consumption"] * 1e6)
    )  # t CO2 per MWh

    # GDP per capita (2017 PPP $)
    panel["gdp_per_capita"] = panel["gdp"] / panel["population"]

    # Drop countries with too many missing Kaya values
    valid = (
        panel.groupby("iso_code")[KAYA_COLS]
        .apply(lambda g: g.notna().all(axis=1).sum())
    )
    valid_isos = valid[valid >= MIN_VALID_YEARS].index
    panel = panel[panel["iso_code"].isin(valid_isos)].reset_index(drop=True)

    return panel


def save_panel(panel: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    print(f"  Saved panel: {panel.shape[0]} rows, {panel['iso_code'].nunique()} countries, "
          f"years {panel['year'].min()}-{panel['year'].max()}")


if __name__ == "__main__":
    co2_raw, energy_raw = load_raw()
    panel = build_panel(co2_raw, energy_raw)
    repo_root = Path(__file__).resolve().parent.parent.parent
    save_panel(panel, str(repo_root / "data" / "panel.csv"))
    print("\nSample:")
    deu = panel[panel["iso_code"] == "DEU"][["year", "co2", "gdp", "primary_energy_consumption",
                                              "co2_per_energy", "renewables_share_energy"]].head(5)
    print(deu.to_string(index=False))
