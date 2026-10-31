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
