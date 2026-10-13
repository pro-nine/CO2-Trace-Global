from scripts import run_analysis


def test_fast_run_writes_a_report_whose_numbers_come_from_the_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_analysis, "OUT", tmp_path)
    monkeypatch.setattr(run_analysis, "FIG", tmp_path / "figures")
    run_analysis.main(["--fast"])
    report = (tmp_path / "analysis_report.md").read_text(encoding="utf-8")
    assert "**not supported**" in report and "**misleading**" in report
    assert "Germany is 0.53 from Ukraine" in report
    assert report.count("\n| ") < 60
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert (tmp_path / "claims.json").exists()
    assert (tmp_path / "tables" / "synthetic_control.csv").exists()
