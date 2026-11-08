"""Tests for the geo tools. None of these contact Earth Engine."""

from __future__ import annotations

from eudr_agent.config import PLOTS_DIR
from eudr_agent.geo import tools
from eudr_agent.plots import load_plots

SQUARE = [[75.123456, 13.123456], [75.133456, 13.123456], [75.133456, 13.133456],
          [75.123456, 13.133456], [75.123456, 13.123456]]


def polygon(ring):
    return {"type": "Polygon", "coordinates": [ring]}


def reasons(geometry, area_ha=None):
    return tools.validate_geometry(geometry, area_ha)["values"]["reasons"]


def test_real_plots_pass_validation():
    for plot in load_plots().values():
        assert tools.validate_geometry(plot["geometry"], plot["area_ha"])["values"]["valid"]


def test_rejects_point_for_large_plot_but_allows_small():
    point = {"type": "Point", "coordinates": [75.123456, 13.123456]}
    assert "needs a polygon" in reasons(point, 9.0)[0]
    assert reasons(point, 3.0) == []


def test_rejects_too_few_decimals_but_tolerates_trailing_zero():
    rounded = [[round(x, 3), round(y, 3)] for x, y in SQUARE]
    assert "decimal places" in reasons(polygon(rounded))[0]
    one_short = [[75.12, 13.123456]] + SQUARE[1:4] + [[75.12, 13.123456]]
    assert reasons(polygon(one_short)) == []


def test_rejects_self_crossing_and_open_polygons():
    bowtie = [SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3], SQUARE[0]]
    assert "crosses itself" in reasons(polygon(bowtie))[0]
    assert "closed" in reasons(polygon(SQUARE[:-1]))[0]


def test_rejects_malformed_geometry():
    assert not tools.validate_geometry({"type": "Line"})["values"]["valid"]
    assert not tools.validate_geometry({})["values"]["valid"]


def test_area_of_about_one_square_km():
    assert 118 < tools.polygon_area_ha(SQUARE) < 122


def test_read_document_blocks_other_paths():
    assert "permitted" in tools.read_document("../../.env")["error"]
    assert "permitted" in tools.read_document(str(PLOTS_DIR / "labels.csv"))["error"]
    assert tools.read_document(str(PLOTS_DIR / "documents" / "K-01_land_record.md"))["values"]["mocked"] is True


def test_failed_tool_returns_error_and_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "GEO_CACHE_DIR", tmp_path)

    @tools._satellite_tool("test-dataset", "note")
    def broken(geometry):
        raise ValueError("quota exceeded")

    finding = broken(polygon(SQUARE))
    assert "quota exceeded" in finding["error"]
    assert list(tmp_path.iterdir()) == []


def test_live_and_cached_results_have_the_same_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "GEO_CACHE_DIR", tmp_path)

    @tools._satellite_tool("test-dataset", "note")
    def by_year(geometry):
        return {"loss_by_year": {2025: 0.22}}

    live, cached = by_year(polygon(SQUARE)), by_year(polygon(SQUARE))
    assert live["cached"] is False and cached["cached"] is True
    assert live["values"] == cached["values"] == {"loss_by_year": {"2025": 0.22}}


def test_second_call_reads_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "GEO_CACHE_DIR", tmp_path)
    calls = []

    @tools._satellite_tool("test-dataset", "note")
    def counted(geometry):
        calls.append(1)
        return {"ha": 1.5}

    assert counted(polygon(SQUARE))["cached"] is False
    second = counted(polygon(SQUARE))
    assert second["cached"] is True and second["values"] == {"ha": 1.5} and len(calls) == 1


# ---------- the newer satellite tools, checked against each other on the saved real results ----------

from eudr_agent.geo import agent as geo_agent, prompts as geo_prompts  # noqa: E402
from eudr_agent.plots import load_plots  # noqa: E402
from eudr_agent.report import TOOL_LABELS, key_figures  # noqa: E402

NEW = ("locate_loss", "boundary_sensitivity", "ndvi_series", "coffee_probability")


def cached(name, plot_id):
    finding = getattr(tools, name)(load_plots()[plot_id]["geometry"])
    assert finding.get("cached"), f"{name} {plot_id} is not cached; run scripts/run_geo_tools.py"
    return finding["values"]


def test_every_offered_tool_exists_is_labelled_and_explained():
    for name in geo_agent.SATELLITE_TOOLS:
        assert callable(getattr(tools, name)) and name in TOOL_LABELS
    assert all(name in geo_prompts.GEO_SYSTEM for name in NEW)  # each follow-up tool has a rule for when to use it


def test_located_patches_add_up_to_the_loss_totals():
    for plot_id in ("C-01", "C-02", "K-01", "N-01"):
        located, loss = cached("locate_loss", plot_id), cached("loss_after_2020", plot_id)
        union = max(loss["hansen_loss_ha"], loss["tmf_deforestation_ha"])
        assert abs(located["loss_ha"] - union) <= 0.02, plot_id
        assert located["pattern"] == ("none" if union == 0 else "scattered")  # no plot has a 0.5 ha block


def test_boundary_as_drawn_matches_the_loss_tool():
    for plot_id in ("C-01", "C-02", "K-01", "N-01"):
        drawn = cached("boundary_sensitivity", plot_id)["variants"]["as_drawn"]["loss_ha"]
        loss = cached("loss_after_2020", plot_id)
        assert abs(drawn - max(loss["hansen_loss_ha"], loss["tmf_deforestation_ha"])) <= 0.02, plot_id


def test_edge_loss_on_k01_makes_the_boundary_check_unstable():
    k01 = cached("boundary_sensitivity", "K-01")
    assert not k01["stable"] and "within 30 m" in k01["changes"][0]
    assert cached("boundary_sensitivity", "C-02")["stable"]


def test_ndvi_covers_every_finished_dry_season_and_loss_pixels_only_where_there_is_loss():
    seasons = [str(y) for y in range(tools.NDVI_FIRST_YEAR, tools.NDVI_LAST_YEAR + 1)]
    for plot_id in ("C-01", "C-02", "K-01", "N-01"):
        v = cached("ndvi_series", plot_id)
        assert list(v["plot"]) == seasons
        assert bool(v["loss_pixels"]) == (cached("locate_loss", plot_id)["loss_ha"] > 0)


def test_new_findings_render_for_the_report():
    for name in NEW:
        finding = getattr(tools, name)(load_plots()["K-01"]["geometry"])
        assert key_figures(finding) and all(label and value for label, value in key_figures(finding))
