"""Baseline: FAO's Whisp on the test plots, next to our own evidence, for the evaluation.

Whisp is not an agent tool. It runs a fixed set of indicators on every plot and gives one risk class, so it
is the "no agent" comparison: what a compliance officer would get from an existing tool alone.

Usage:  python scripts/run_whisp_baseline.py        (needs EE_PROJECT and  pip install -e ".[geo]")
Writes: eval/whisp_baseline.csv
"""

from __future__ import annotations

import csv
import os
import sys

from eudr_agent.config import PLOTS_DIR, ROOT
from eudr_agent.geo import tools
from eudr_agent.plots import load_plots

OUT = ROOT / "eval" / "whisp_baseline.csv"
INDICATORS = ["Ind_01_treecover", "Ind_02_commodities", "Ind_03_disturbance_before_2020",
              "Ind_04_disturbance_after_2020", "Ind_05_primary_2020", "Ind_06_nat_reg_forest_2020"]


def main() -> None:
    tools._ee()  # our login first, so Whisp reuses it
    import openforis_whisp as whisp

    stats = whisp.whisp_formatted_stats_geojson_to_df(PLOTS_DIR / "plots.geojson", external_id_column="plot_id")
    risk = whisp.whisp_risk(stats)
    labels = {r["plot_id"]: r["label"] for r in csv.DictReader((PLOTS_DIR / "labels.csv").open())}
    plots = load_plots()

    rows = []
    for _, r in risk.iterrows():
        plot_id = r["external_id"]
        geometry = plots[plot_id]["geometry"]
        ours_protected = tools.protected_area_check(geometry).get("values", {})
        ours_forest = tools.forest_2020(geometry).get("values", {})
        ours_loss = tools.locate_loss(geometry).get("values", {})
        rows.append({
            "plot_id": plot_id, "manual_label": labels.get(plot_id, ""),
            "whisp_risk_pcrop": r["risk_pcrop"], **{name: r[name] for name in INDICATORS},
            "whisp_coffee_2020": r.get("Coffee_FDaP"),
            "ours_in_protected_area": ours_protected.get("inside_protected_area"),
            "ours_gfc2020_forest_share": ours_forest.get("gfc2020_forest_share"),
            "ours_hansen_tree_cover_share": ours_forest.get("hansen_tree_cover_share"),
            "ours_loss_after_2020_ha": ours_loss.get("loss_ha"),
        })

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'plot':<6}{'label':<12}{'whisp':<18}{'protected':<11}{'forest (EU/Hansen)':<20}{'loss after 2020'}")
    for row in rows:
        print(f"{row['plot_id']:<6}{row['manual_label']:<12}{row['whisp_risk_pcrop']:<18}"
              f"{str(row['ours_in_protected_area']):<11}"
              f"{row['ours_gfc2020_forest_share']:.0%} / {row['ours_hansen_tree_cover_share']:.0%}{'':<9}"
              f"{row['ours_loss_after_2020_ha']} ha")
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter cleanup; Earth Engine's threads can delay exit
