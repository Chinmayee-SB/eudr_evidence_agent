"""Load test plots as agent inputs. Labels are deliberately not loaded here: the agents must not see them."""

from __future__ import annotations

import json

from eudr_agent.config import PLOTS_DIR
from eudr_agent.state import PlotInput


def load_plots() -> dict[str, PlotInput]:
    collection = json.loads((PLOTS_DIR / "plots.geojson").read_text(encoding="utf-8"))
    details = json.loads((PLOTS_DIR / "estate_details.json").read_text(encoding="utf-8"))
    plots: dict[str, PlotInput] = {}
    for feature in collection["features"]:
        plot_id = feature["properties"]["plot_id"]
        extra = details.get(plot_id, {})
        plots[plot_id] = PlotInput(
            plot_id=plot_id,
            geometry=feature["geometry"],
            area_ha=feature["properties"]["area_ha"],
            declared_crop=extra.get("declared_crop", "unknown"),
            documents=[str(PLOTS_DIR / doc) for doc in extra.get("documents", [])],
        )
        if "crop_since" in extra:
            plots[plot_id]["crop_since"] = extra["crop_since"]
    return plots
