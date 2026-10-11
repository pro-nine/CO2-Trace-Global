import numpy as np
import pandas as pd
import pytest

from src.decoupling.classifier import DecouplingState, _classify_window, classify_all, classify_country

S = DecouplingState


def make_country(iso="AAA", years=range(1990, 2023), gdp_rate=0.02, co2_rate=-0.01, energy=None, co2_energy=None):
    years = list(years)
    n = len(years)
    return pd.DataFrame({
        "iso_code": iso, "country": f"Country {iso}", "year": years,
        "gdp_per_capita": 10_000 * (1 + gdp_rate) ** np.arange(n),
        "co2_per_capita": 8.0 * (1 + co2_rate) ** np.arange(n),
        "energy_per_gdp": energy if energy is not None else np.ones(n),
        "co2_per_energy": co2_energy if co2_energy is not None else np.ones(n),
    })


@pytest.mark.parametrize("gdp, co2, expected", [
    (2.0, -1.0, S.ABSOLUTE), (3.0, 1.0, S.RELATIVE), (2.0, 2.0, S.NO_DECOUPLING), (2.0, 5.0, S.NO_DECOUPLING),
    (-2.0, -1.0, S.DEGROWTH), (-2.0, 1.0, S.UNCLEAR), (0.2, -3.0, S.UNCLEAR), (0.5, -3.0, S.UNCLEAR),
    (np.nan, 1.0, S.UNCLEAR), (1.0, np.nan, S.UNCLEAR),
])
def test_legacy_window_states(gdp, co2, expected):
    assert _classify_window(gdp, co2) is expected


def test_steady_absolute_decoupling_and_current_status():
    result = classify_country(make_country(), "AAA")
    assert result.years_absolute == 32
    assert result.longest_abs_run == 32
    assert bool(result.currently_absolute)
    assert set(result.state_by_decade.values()) == {"absolute_decoupling"}


def test_rising_emissions_are_not_decoupled():
    result = classify_country(make_country(co2_rate=0.03), "AAA")
    assert result.years_absolute == 0
    assert not bool(result.currently_absolute)
    assert set(result.state_by_decade.values()) == {"no_decoupling"}


def test_short_series_returns_none_and_classify_all_skips_it():
    short = make_country("BBB", years=range(2015, 2020))
    assert classify_country(short, "BBB") is None
    out = classify_all(pd.concat([make_country("AAA"), short]))
    assert list(out["iso_code"]) == ["AAA"]


def test_missing_years_do_not_create_a_consecutive_run():
    years = [1990, 1991, 1992, 1993, 1994, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
    result = classify_country(make_country(years=years), "AAA")
    assert result.longest_abs_run == 6
    assert result.years_absolute == 10


def test_offshoring_and_deindustrialisation_flags_use_separate_fuel_mix_conditions():
    n = 33
    flat_fuel = classify_country(make_country(energy=np.linspace(1, 0.5, n), co2_energy=np.ones(n)), "AAA")
    cleaner_fuel = classify_country(make_country(energy=np.linspace(1, 0.5, n), co2_energy=np.linspace(1, 0.6, n)), "AAA")
    assert bool(flat_fuel.suspect_offshoring) and bool(flat_fuel.suspect_deindustrialisation)
    assert not bool(cleaner_fuel.suspect_offshoring) and not bool(cleaner_fuel.suspect_deindustrialisation)
