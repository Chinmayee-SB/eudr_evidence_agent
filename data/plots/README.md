# Test plots

GeoJSON polygons drawn by the team on satellite imagery near known coffee estates, plus one
protected-forest control plot. Approximate and often partial: legal estate boundaries are private.
Labels refer to the land inside each polygon.

Plot names are anonymised. The estate-name mapping and original KML files are kept outside this
repository (`../eudr_private/`) and must never be added here.

- `plots.geojson` - one feature per plot, properties: plot_id, area_ha (computed from the polygon)
- `estate_details.json` - MOCKED declared crop, crop-since year and document list
- `documents/` - MOCKED land records. C-02 and N-01 deliberately have none
- `labels.csv` - manual labels for evaluation. Never passed to the agents, so they cannot see the answer

Current state: 4 plots, one labeller (A). A second labeller is needed before agreement can be measured.
