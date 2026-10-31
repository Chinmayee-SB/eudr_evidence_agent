# Dataset limitation notes

Short notes the rules agent can cite when a conflict comes from a dataset's known weakness.
Each note names where the fact comes from.

## JRC GFC2020 forest map and coffee
The EU's Global Forest Cover 2020 map (JRC) is a 10 m forest / not-forest map for the EUDR cut-off date. The JRC reports that it has no authoritative status and shows high errors for coffee and cocoa, because both grow under a tree canopy that is hard to tell apart from forest in optical imagery. Overlap with forest on this map does not by itself mean non-compliance. Sources: JRC technical report JRC136960; Bourgoin et al., ESSD 2026.

## Hansen Global Forest Change loss is not deforestation
Hansen GFC (University of Maryland) records the year a 30 m pixel lost tree canopy. Loss can come from clearing, pruning, storms, fire or plantation harvest. It does not say whether land was converted to agricultural use, which is what the EUDR definition of deforestation requires. Source: Hansen et al., Global Forest Change v1.13 documentation.

## JRC Tropical Moist Forest degradation versus deforestation
The JRC TMF dataset separates the first year of deforestation (full canopy removal) from the first year of degradation (partial, temporary canopy damage). It only covers tropical moist forest. Degradation in TMF is not the same as forest degradation as defined in EUDR Article 2(7). Source: JRC TMF v1 2025 documentation.

## Whisp risk thresholds
FAO's Whisp turns dataset areas into yes/no indicators using fixed shares of the plot: tree cover above 10 percent, commodity or pre-2020 disturbance above 50 percent, post-2020 disturbance above 10 percent. Loss below 10 percent of a large plot does not raise its post-2020 disturbance indicator. Its coffee probability layer is reported but not used in the perennial-crop risk score. Source: openforis-whisp source code, risk.py and lookup_datasets.csv.

## Coffee probability map accuracy
The Forest Data Partnership coffee model gives a per-pixel probability of coffee at 10 m. Whisp's code notes precision and recall of about 70 percent at its threshold. The model is not peer-reviewed and has little validation in India. Treat it as a hint, not proof of agricultural use. Source: Google Earth Engine catalog, Forest Data Partnership coffee model; Whisp datasets.py.
