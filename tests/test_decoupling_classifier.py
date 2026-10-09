import numpy as np
import pandas as pd
import pytest

from src.decoupling.classifier import (
    DecouplingState,
    _classify_window,
    classify_all,
    classify_country,
)

S = DecouplingState


def make_country(iso="AAA", years=range(1990, 2023), gdp_rate=0.02, co2_rate=-0.01,
                 energy=None, co2_energy=None):
    years = list(years)
    n = len(years)
    return pd.DataFrame({
        "iso_code": iso,
        "country": f"Country {iso}",
        "year": years,
        "gdp_per_capita": 10_000 * (1 + gdp_rate) ** np.arange(n),
        "co2_per_capita": 8.0 * (1 + co2_rate) ** np.arange(n),
        "energy_per_gdp": energy if energy is not None else np.ones(n),
        "co2_per_energy": co2_energy if co2_energy is not None else np.ones(n),
    })


@pytest.mark.parametrize(
    "gdp, co2, expected",
    [
        (2.0, -1.0, S.ABSOLUTE),
        (3.0, 1.0, S.RELATIVE),
        (2.0, 2.0, S.NO_DECOUPLING),
        (2.0, 5.0, S.NO_DECOUPLING),
        (-2.0, -1.0, S.DEGROWTH),
        (0.2, -3.0, S.UNCLEAR),
        (0.5, -3.0, S.UNCLEAR),
        (np.nan, 1.0, S.UNCLEAR),
        (1.0, np.nan, S.UNCLEAR),
    ],
)
def test_classify_window_states(gdp, co2, expected):
    assert _classify_window(gdp, co2) is expected


def test_classify_window_threshold_is_configurable():
    assert _classify_window(0.4, -1.0, threshold=0.3) is S.ABSOLUTE
    assert _classify_window(0.4, -1.0, threshold=0.5) is S.UNCLEAR


def test_country_with_steady_absolute_decoupling():
    result = classify_country(make_country(), "AAA")
    assert result.years_absolute == 32
    assert result.longest_abs_run == 32
    assert bool(result.currently_absolute) is True
    assert set(result.state_by_decade.values()) == {S.ABSOLUTE}
    assert result.gdp_per_cap_growth_pct_1990_2022 > 0
    assert result.co2_per_cap_change_pct_1990_2022 < 0
    assert not result.suspect_offshoring


def test_country_with_rising_emissions_is_not_decoupled():
    result = classify_country(make_country(gdp_rate=0.02, co2_rate=0.03), "AAA")
    assert result.years_absolute == 0
    assert bool(result.currently_absolute) is False
    assert set(result.state_by_decade.values()) == {S.NO_DECOUPLING}


def test_short_series_returns_none():
    assert classify_country(make_country(years=range(2010, 2018)), "AAA") is None


def test_missing_values_are_dropped_before_the_length_check():
    df = make_country(years=range(2000, 2012))
    df.loc[df.index[:5], "gdp_per_capita"] = np.nan
    assert classify_country(df, "AAA") is None


def test_classify_all_returns_one_row_per_eligible_country():
    panel = pd.concat([make_country("AAA"), make_country("BBB", co2_rate=0.03),
                       make_country("CCC", years=range(2015, 2020))])
    out = classify_all(panel)
    assert sorted(out["iso_code"]) == ["AAA", "BBB"]
    assert {"decade_1990_2000", "longest_consecutive_abs_run",
            "suspect_offshoring", "co2_per_cap_change_pct"} <= set(out.columns)
    row = out.set_index("iso_code")
    assert row.loc["AAA", "decade_2010_2022"] == S.ABSOLUTE.value
    assert row.loc["BBB", "decade_2010_2022"] == S.NO_DECOUPLING.value


def test_offshoring_flag_when_energy_intensity_collapses_without_fuel_switching():
    n = 33
    result = classify_country(
        make_country(energy=np.linspace(1.0, 0.5, n), co2_energy=np.ones(n)), "AAA")
    assert bool(result.suspect_offshoring) is True


def test_offshoring_flag_off_when_fuel_mix_also_improves():
    n = 33
    result = classify_country(
        make_country(energy=np.linspace(1.0, 0.5, n), co2_energy=np.linspace(1.0, 0.6, n)), "AAA")
    assert bool(result.suspect_offshoring) is False


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: e_drop / (c_drop + 0.001) is negative when CO2-per-energy is flat "
    "or rising, so the clearest deindustrialisation cases are never flagged."))
def test_deindustrialisation_flag_when_energy_intensity_halves_and_fuel_mix_is_flat():
    n = 33
    result = classify_country(
        make_country(energy=np.linspace(1.0, 0.5, n), co2_energy=np.ones(n)), "AAA")
    assert bool(result.suspect_deindustrialisation) is True


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: after dropna, pct_change spans multi-year gaps, so non-adjacent "
    "years are counted as one consecutive run."))
def test_longest_run_does_not_bridge_missing_years():
    years = [1990, 1991, 1992, 1993, 1994, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
    result = classify_country(make_country(years=years), "AAA")
    assert result.longest_abs_run <= 6
