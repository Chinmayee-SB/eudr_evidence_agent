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

Current state (10 Nov 2026): 7 test plots, kept for testing the agent.

| Group | Plots | Labeller | Expected |
|---|---|---|---|
| coffee | C-01, C-02, K-01 | A (human) | Clear (rule 4), Clear (rule 6), Needs evidence (rule 7) |
| control: protected forest (Karnataka) | N-01, S-01 | A | High risk (rule 1) |
| control: non-coffee, Kerala | T-01 tea (AI label), W-01 rubber (A) | | Blocked by the input check |

- Every plot, the non-coffee controls included, declares coffee in the mocked details: the agent has to check the claim.
- Only C-01 and K-01 have a (mocked) land record.
- 16 more plots (C-03 to C-06, H-01 to H-03, K-02 to K-08, N-02, U-01) are kept outside the repository in
  `../data/<plot>/`, one ready-to-upload folder each (boundary .kml and .geojson, mocked land record, what to
  enter on the form, and the label to compare with). They are submitted on the desk as new plots.
- Labeller "AI" is an AI model's reading of the imagery: a second opinion, not ground truth.
