"""Offline-first panel loading with an integrity manifest and a data contract.

The analysis runs against the committed snapshot `data/panel.csv`. Its SHA-256
(line endings normalised) is recorded in `data/panel.manifest.json`; `load_panel`
refuses a file that does not match. Fetching newer OWID data is an explicit
action (`refresh_from_owid`), never an implicit side effect of a run.

Sources: https://github.com/owid/co2-data and https://github.com/owid/energy-data
(CC BY). CO2 is territorial fossil and industrial CO2, excluding land-use change.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "panel.csv"
MANIFEST_PATH = ROOT / "data" / "panel.manifest.json"
CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
YEAR_MIN, YEAR_MAX, MIN_VALID_YEARS = 1990, 2022, 25
KAYA_COLS = ["co2", "gdp", "population", "primary_energy_consumption"]
REQUIRED = ["iso_code", "country", "year", "co2", "gdp", "population", "primary_energy_consumption",
            "co2_per_capita", "gdp_per_capita", "energy_per_gdp", "co2_per_energy"]
ENERGY_EXTRA = ["renewables_share_energy", "fossil_share_energy", "low_carbon_share_energy",
                "renewables_electricity", "electricity_generation"]


class DataContractError(ValueError):
    """Raised when the panel breaks its documented contract."""


def file_sha256(path) -> str:
    """SHA-256 of the file with CRLF normalised to LF, so checkouts agree across platforms."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_panel(df: pd.DataFrame) -> list:
    """Return a list of contract violations (empty when the panel is valid)."""
    problems = [f"missing column: {c}" for c in REQUIRED if c not in df.columns]
    if problems:
        return problems
    if df.duplicated(["iso_code", "year"]).any():
        problems.append("duplicate (iso_code, year) rows")
    if not df["year"].between(YEAR_MIN, YEAR_MAX).all():
        problems.append(f"years outside {YEAR_MIN}-{YEAR_MAX}")
    if df["iso_code"].str.startswith("OWID", na=False).any() or df["iso_code"].isna().any():
        problems.append("regional aggregate or missing ISO code present")
    if not df["iso_code"].str.fullmatch(r"[A-Z]{3}").all():
        problems.append("ISO codes are not three capital letters")
    counts = df.groupby("iso_code")["year"].nunique()
    if (counts < MIN_VALID_YEARS).any():
        problems.append("country with too few years")
    for col in ["co2", "gdp", "population", "gdp_per_capita", "co2_per_capita"]:
        if (df[col] < 0).any():
            problems.append(f"negative values in {col}")
    return problems


def load_panel(path=PANEL_PATH, verify: bool = True) -> pd.DataFrame:
    path = Path(path)
    if verify and MANIFEST_PATH.exists() and path == PANEL_PATH:
        expected = json.loads(MANIFEST_PATH.read_text())["sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise DataContractError(f"panel.csv hash {actual[:12]} does not match manifest {expected[:12]}")
    df = pd.read_csv(path)
    problems = validate_panel(df)
    if problems:
        raise DataContractError("; ".join(problems))
    return df


def write_manifest(path=PANEL_PATH, manifest=MANIFEST_PATH) -> dict:
    df = pd.read_csv(path)
    info = {"sha256": file_sha256(path), "rows": int(len(df)), "countries": int(df["iso_code"].nunique()),
            "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
            "sources": [CO2_URL, ENERGY_URL], "note": "Snapshot committed to this repository."}
    Path(manifest).write_text(json.dumps(info, indent=2) + "\n")
    return info


def _download(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as resp:
        return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))


def load_raw():
    return _download(CO2_URL), _download(ENERGY_URL)


def build_panel(co2_raw: pd.DataFrame, energy_raw: pd.DataFrame) -> pd.DataFrame:
    def clean(x):
        x = x.dropna(subset=["iso_code"])
        x = x[~x["iso_code"].str.startswith("OWID", na=False)]
        return x[x["year"].between(YEAR_MIN, YEAR_MAX)]
    co2, energy = clean(co2_raw), clean(energy_raw)
    keep = ["iso_code", "country", "year", *KAYA_COLS, "co2_per_capita", "co2_per_gdp", "energy_per_capita",
            "share_global_co2", "cumulative_co2"]
    panel = co2[keep].merge(energy[["iso_code", "year", *[c for c in ENERGY_EXTRA if c in energy]]],
                            on=["iso_code", "year"], how="left")
    panel["energy_per_gdp"] = panel["primary_energy_consumption"] * 1e9 / panel["gdp"]
    panel["co2_per_energy"] = panel["co2"] / panel["primary_energy_consumption"]
    panel["gdp_per_capita"] = panel["gdp"] / panel["population"]
    valid = panel.groupby("iso_code")[KAYA_COLS].apply(lambda g: g.notna().all(axis=1).sum())
    return panel[panel["iso_code"].isin(valid[valid >= MIN_VALID_YEARS].index)].reset_index(drop=True)


def refresh_from_owid(path=PANEL_PATH) -> pd.DataFrame:
    """Explicit network refresh. Rewrites the snapshot and its manifest."""
    panel = build_panel(*load_raw())
    problems = validate_panel(panel)
    if problems:
        raise DataContractError("; ".join(problems))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    write_manifest(path)
    return panel
