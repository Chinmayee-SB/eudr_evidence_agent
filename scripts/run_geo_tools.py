"""Run the geo tools on every test plot and print the results. Fills the per-plot cache.

Usage:  python scripts/run_geo_tools.py
"""

from __future__ import annotations

import json
import time

from eudr_agent.geo import tools
from eudr_agent.plots import load_plots


def main() -> None:
    for plot_id, plot in load_plots().items():
        print(f"\n=== {plot_id}  ({plot['area_ha']} ha, documents: {len(plot['documents'])})")
        print(f"  validate_geometry     {tools.validate_geometry(plot['geometry'], plot['area_ha'])['values']}")
        for tool in (tools.protected_area_check, tools.forest_2020, tools.loss_after_2020):
            start = time.perf_counter()
            finding = tool(plot["geometry"])
            source = "cache" if finding.get("cached") else f"live {time.perf_counter() - start:.1f}s"
            print(f"  {tool.__name__:<21} [{source}] {json.dumps(finding.get('values', finding.get('error')))}")


if __name__ == "__main__":
    main()
