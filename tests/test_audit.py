import json

import pandas as pd
import pytest

from src.audit.claims import claim_coverage, claim_efficiency_led, claim_germany_ukraine, claim_offshoring_flag, claim_sustained_decoupling, efficiency_front_load
from src.data import loader


@pytest.fixture(scope="module")
def panel():
    return loader.load_panel()


def test_committed_snapshot_matches_its_manifest_and_contract(panel):
    manifest = json.loads(loader.MANIFEST_PATH.read_text())
    assert loader.file_sha256(loader.PANEL_PATH) == manifest["sha256"]
    assert (len(panel), panel["iso_code"].nunique()) == (manifest["rows"], manifest["countries"])
    assert loader.validate_panel(panel) == []


def test_contract_rejects_duplicates_negatives_aggregates_and_missing_columns(panel):
    dup = pd.concat([panel, panel.iloc[[0]]])
    assert any("duplicate" in p for p in loader.validate_panel(dup))
    neg = panel.copy()
    neg.loc[0, "gdp"] = -1
    assert any("negative" in p for p in loader.validate_panel(neg))
    agg = panel.copy()
    agg.loc[0, "iso_code"] = "OWID_WRL"
    assert any("aggregate" in p for p in loader.validate_panel(agg))
    assert loader.validate_panel(panel.drop(columns=["co2"])) == ["missing column: co2"]


def test_hash_is_stable_across_line_endings(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\n1,2\n")
    b.write_bytes(b"x,y\r\n1,2\r\n")
    assert loader.file_sha256(a) == loader.file_sha256(b)


def test_snapshot_with_a_wrong_hash_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "manifest.json"
    fake.write_text(json.dumps({"sha256": "0" * 64}))
    monkeypatch.setattr(loader, "MANIFEST_PATH", fake)
    with pytest.raises(loader.DataContractError):
        loader.load_panel()


def test_germany_ukraine_claim_is_not_supported(panel):
    c = claim_germany_ukraine(panel, n_boot=60)
    ev = c["evidence"]
    assert c["verdict"] == "not supported"
    assert ev["distance_ukraine"] > ev["distance_denmark"] > ev["distance_sweden"]
    assert ev["co_assign_ukraine"] < 0.6 < 0.9 < ev["co_assign_denmark"]


def test_efficiency_led_claim_is_supported_with_a_stable_range(panel):
    c = claim_efficiency_led(panel)
    assert c["verdict"] == "supported"
    assert c["evidence"]["n_reducers"] == 39
    assert 0.75 < c["evidence"]["p05"] <= c["evidence"]["efficiency_led_share"] <= c["evidence"]["p95"] < 0.9


def test_sustained_decoupling_depends_on_the_spell_definition(panel):
    c = claim_sustained_decoupling(panel)
    assert c["verdict"] == "definition-sensitive"
    assert c["evidence"]["strict_longest_years"] == 7
    assert c["evidence"]["tolerant_longest_years"] > 14
    assert c["evidence"]["strict_share_one_year"] > 0.6


def test_offshoring_flag_is_misleading_because_most_flagged_countries_grew(panel):
    c = claim_offshoring_flag(panel)
    assert c["verdict"] == "misleading"
    assert (c["evidence"]["n_flagged"], c["evidence"]["n_flagged_with_rising_co2_per_capita"]) == (41, 24)


def test_coverage_counts_are_reported_not_assumed(panel):
    ev = claim_coverage(panel)["evidence"]
    assert (ev["countries_in_panel"], ev["countries_in_endpoint_decomposition"], ev["countries_in_chain_decomposition"]) == (162, 154, 153)


def test_germany_has_the_most_front_loaded_efficiency_effect_among_large_reducers(panel):
    front = efficiency_front_load(panel)
    assert front.iloc[0]["iso_code"] == "DEU"
    assert 0.3 < front.iloc[0]["early_share"] < 0.4
