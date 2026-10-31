"""Tests for the test-plot inputs."""

from __future__ import annotations

import csv
from pathlib import Path

from eudr_agent.config import PLOTS_DIR
from eudr_agent.plots import load_plots


def test_all_plots_load_with_a_closed_polygon_and_labels():
    plots = load_plots()
    labelled = {r["plot_id"] for r in csv.DictReader((PLOTS_DIR / "labels.csv").open())}
    assert set(plots) == labelled
    for plot in plots.values():
        ring = plot["geometry"]["coordinates"][0]
        assert plot["geometry"]["type"] == "Polygon"
        assert ring[0] == ring[-1]
        assert plot["area_ha"] > 0


def test_listed_documents_exist():
    for plot in load_plots().values():
        assert all(Path(doc).exists() for doc in plot["documents"])


def test_agent_input_cannot_reveal_the_label():
    allowed = {"plot_id", "geometry", "area_ha", "declared_crop", "crop_since", "documents"}
    for plot in load_plots().values():
        assert set(plot) <= allowed
