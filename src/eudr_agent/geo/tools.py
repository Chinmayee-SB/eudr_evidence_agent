"""Tools for the geo-analyst agent.

Each tool takes a plot geometry and returns numbers in a fixed shape (see state.Finding).
Built: validate_geometry, read_document, protected_area_check, forest_2020, loss_after_2020.
The rest are still planned, with signatures fixed so later work can be written against them.

Earth Engine does the pixel counting on Google's servers, so no satellite files are downloaded.
Results are cached per plot and dataset version, so repeat runs are fast and work during an outage.
A failed query returns a Finding with `error` set, and is not cached.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from shapely.geometry import shape

from eudr_agent import documents
from eudr_agent.config import GEO_CACHE_DIR, PLOTS_DIR, settings
from eudr_agent.state import Finding

Geometry = dict[str, Any]  # GeoJSON

MAX_POINT_AREA_HA = 4.0   # EUDR Art. 2(28): a plot above this must be a polygon
MIN_DECIMALS = 6          # EUDR Art. 2(28)
TREE_COVER_MIN = 10       # percent; the canopy threshold in the EUDR forest definition, Art. 2(4)
MIN_OVERLAP_HA = 0.05     # ignore sliver overlaps from slightly misaligned boundaries
MAX_DOCUMENT_CHARS = 8000 # keeps a long upload from filling the model's context
IN_FORCE = {"Designated", "Inscribed", "Established", "Adopted"}  # WDPA statuses that are legally in effect

HANSEN = "UMD/hansen/global_forest_change_2025_v1_13"
TMF_DEFORESTATION = "projects/JRC/TMF/v1_2025/DeforestationYear"
TMF_DEGRADATION = "projects/JRC/TMF/v1_2025/DegradationYear"
GFC2020 = "JRC/GFC2020/V4"
WDPA = "WCMC/WDPA/current/polygons"
DOCUMENTS_DIR = PLOTS_DIR / "documents"


# ---------- Earth Engine connection and cache ----------

@functools.lru_cache(maxsize=1)
def _ee() -> Any:
    """Connect on first live query only, so cached results need no login or network."""
    import ee

    if not settings.ee_project:
        raise RuntimeError("EE_PROJECT is not set in .env")
    ee.Initialize(project=settings.ee_project)
    return ee


def _cache_path(tool: str, dataset: str, geometry: Geometry) -> Path:
    key = json.dumps([tool, dataset, geometry], sort_keys=True)
    return GEO_CACHE_DIR / f"{tool}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def _satellite_tool(dataset: str, note: str) -> Callable:
    """Wrap a tool that returns plain values: add the cache, the fixed Finding shape and error handling."""
    def wrap(fn: Callable[[Geometry], dict[str, Any]]) -> Callable[[Geometry], Finding]:
        @functools.wraps(fn)
        def tool(geometry: Geometry) -> Finding:
            path = _cache_path(fn.__name__, dataset, geometry)
            if path.exists():
                return Finding(**json.loads(path.read_text(encoding="utf-8")), cached=True)
            try:
                # the JSON round trip makes a live result identical to a cached one, e.g. year keys become strings
                values = json.loads(json.dumps(fn(geometry)))
                finding = Finding(tool=fn.__name__, dataset=dataset, values=values, note=note)
            except Exception as e:  # the agent must report a failed tool, not stop or guess
                return Finding(tool=fn.__name__, dataset=dataset, error=f"{type(e).__name__}: {e}"[:300])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(finding, indent=1), encoding="utf-8")
            return Finding(**finding, cached=False)
        return tool
    return wrap


def _area_ha_by_year(ee: Any, year_image: Any, region: Any) -> dict[int, float]:
    """Hectares per year value inside the region, from an image whose pixels hold a year."""
    stacked = ee.Image.pixelArea().divide(1e4).addBands(year_image.rename("year").toInt())
    groups = stacked.reduceRegion(reducer=ee.Reducer.sum().group(groupField=1, groupName="year"),
                                  geometry=region, scale=30, maxPixels=int(1e9)).getInfo()["groups"]
    return {int(g["year"]): round(g["sum"], 2) for g in groups}


def _area_ha(ee: Any, mask: Any, region: Any, scale: int) -> float:
    total = ee.Image.pixelArea().divide(1e4).updateMask(mask).rename("ha").reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=int(1e9)).getInfo()["ha"]
    return round(total or 0.0, 2)


# ---------- local tools (no Earth Engine) ----------

def polygon_area_ha(ring: list[list[float]]) -> float:
    """Area of a lon/lat ring, projected flat around its mean latitude. Accurate enough at plot scale."""
    mean_lat = math.radians(sum(p[1] for p in ring) / len(ring))
    metres_per_degree = 111_320
    x = [p[0] * metres_per_degree * math.cos(mean_lat) for p in ring]
    y = [p[1] * metres_per_degree for p in ring]
    twice_area = sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i] for i in range(len(x)))
    return abs(twice_area) / 2 / 10_000


def _decimals(value: float) -> int:
    return max(0, -Decimal(str(value)).as_tuple().exponent)


def validate_geometry(geometry: Geometry, area_ha: float | None = None) -> Finding:
    """EUDR geolocation rules (Art. 2(28)): valid shape, >= 6 decimal places, polygon if plot > 4 ha."""
    reasons: list[str] = []
    kind = geometry.get("type") if isinstance(geometry, dict) else None
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    computed_ha = None

    if kind == "Point" and coords:
        points = [coords]
        if area_ha is None:
            reasons.append("a point needs a stated plot area, to check it is 4 ha or less")
        elif area_ha > MAX_POINT_AREA_HA:
            reasons.append(f"plot of {area_ha} ha needs a polygon; a single point is allowed only up to 4 ha")
    elif kind == "Polygon" and coords:
        points = coords[0]
        if len(points) < 4 or points[0] != points[-1]:
            reasons.append("polygon ring must be closed and have at least 3 distinct corners")
        else:
            if not shape(geometry).is_valid:
                reasons.append("polygon boundary crosses itself")
            computed_ha = round(polygon_area_ha(points), 2)
    else:
        return Finding(tool="validate_geometry", dataset="EUDR Art. 2(28)",
                       values={"valid": False, "reasons": ["geometry must be a GeoJSON Point or Polygon"]})

    if not all(len(p) >= 2 and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in points):
        reasons.append("coordinates are outside valid longitude and latitude ranges")
    # JSON drops trailing zeros (75.500000 reads as 75.5), so one short coordinate proves nothing.
    # Reject only when most coordinates have too few decimals, which means the data was rounded.
    values = [v for p in points for v in p[:2]]
    if sum(_decimals(v) >= MIN_DECIMALS for v in values) < len(values) / 2:
        reasons.append(f"coordinates need at least {MIN_DECIMALS} decimal places")

    return Finding(tool="validate_geometry", dataset="EUDR Art. 2(28)",
                   values={"valid": not reasons, "reasons": reasons, "computed_area_ha": computed_ha})


def read_document(path: str) -> Finding:
    """Read one land record. Only files in data/plots/documents can be opened, whatever path is asked for."""
    resolved = Path(path).resolve()
    if DOCUMENTS_DIR.resolve() not in resolved.parents:
        return Finding(tool="read_document", dataset="mocked land records",
                       error="document is outside the permitted folder")
    if not resolved.exists():
        return Finding(tool="read_document", dataset="mocked land records", error="document not found")
    try:
        text = documents.read_text(resolved)
    except ValueError as e:
        return Finding(tool="read_document", dataset="mocked land records", error=str(e))
    return Finding(tool="read_document", dataset="mocked land records",
                   values={"file": resolved.name, "mocked": True, "text": text[:MAX_DOCUMENT_CHARS],
                           "truncated": len(text) > MAX_DOCUMENT_CHARS},
                   note="Mock document for testing. Not a real land record.")


# ---------- satellite tools ----------

@_satellite_tool(WDPA, "Some boundaries are coarse. Proposed areas are listed but are not legally in effect.")
def protected_area_check(geometry: Geometry) -> dict[str, Any]:
    """Overlap (ha) with WDPA protected areas, separating those in force from those only proposed."""
    ee = _ee()
    region = ee.Geometry(geometry)
    overlaps = ee.FeatureCollection(WDPA).filterBounds(region).map(
        lambda f: ee.Feature(None, {
            "name": f.get("NAME"), "designation": f.get("DESIG_ENG"), "status": f.get("STATUS"),
            "overlap_m2": f.geometry().intersection(region, 1).area(1),
        })).getInfo()["features"]

    in_force, proposed = [], []
    for feature in overlaps:
        p = feature["properties"]
        hectares = round(p["overlap_m2"] / 10_000, 2)
        if hectares < MIN_OVERLAP_HA:
            continue
        row = {"name": p["name"], "designation": p["designation"], "status": p["status"], "overlap_ha": hectares}
        (in_force if p["status"] in IN_FORCE else proposed).append(row)
    return {"inside_protected_area": bool(in_force),
            "in_force": sorted(in_force, key=lambda r: -r["overlap_ha"]),
            "proposed": sorted(proposed, key=lambda r: -r["overlap_ha"])}


@_satellite_tool(f"{GFC2020}; {HANSEN}",
                 "GFC2020 has no legal status and high errors for coffee, which looks like forest under shade "
                 "trees. Hansen tree cover is from 2000 with later loss removed, so it is only a proxy for 2020.")
def forest_2020(geometry: Geometry) -> dict[str, Any]:
    """Area of the plot that was forest around 2020, on two maps, so disagreement can be seen."""
    ee = _ee()
    region = ee.Geometry(geometry)
    hansen = ee.Image(HANSEN)
    plot_ha = round(region.area(1).getInfo() / 10_000, 2)
    # lossyear has no value where no loss happened; without unmask, those pixels vanish from the comparison.
    no_loss_by_2020 = hansen.select("lossyear").unmask(0).Not().Or(hansen.select("lossyear").unmask(0).gt(20))
    gfc_ha = _area_ha(ee, ee.Image(GFC2020).select("Map").eq(1), region, 10)
    hansen_ha = _area_ha(ee, hansen.select("treecover2000").gte(TREE_COVER_MIN).And(no_loss_by_2020), region, 30)
    return {"plot_ha": plot_ha,
            "gfc2020_forest_ha": gfc_ha, "gfc2020_forest_share": round(gfc_ha / plot_ha, 2),
            "hansen_tree_cover_ha": hansen_ha, "hansen_tree_cover_share": round(hansen_ha / plot_ha, 2)}


@_satellite_tool(f"{HANSEN}; {TMF_DEFORESTATION}; {TMF_DEGRADATION}",
                 "Hansen loss is any canopy loss, not proof of deforestation. TMF degradation is not the same as "
                 "EUDR forest degradation. Neither dataset can tell pruning from clearing.")
def loss_after_2020(geometry: Geometry) -> dict[str, Any]:
    """Hansen GFC v1.13 loss (2021-2025) and JRC TMF deforestation/degradation after 2020, in ha."""
    ee = _ee()
    region = ee.Geometry(geometry)
    hansen_year = ee.Image(HANSEN).select("lossyear")  # 21 means 2021
    hansen = _area_ha_by_year(ee, hansen_year.updateMask(hansen_year.gt(20)).add(2000), region)

    def tmf(collection: str) -> dict[int, float]:
        year = ee.ImageCollection(collection).mosaic()
        return _area_ha_by_year(ee, year.updateMask(year.gt(2020)), region)

    deforestation, degradation = tmf(TMF_DEFORESTATION), tmf(TMF_DEGRADATION)
    return {"plot_ha": round(region.area(1).getInfo() / 10_000, 2),
            "hansen_loss_ha": round(sum(hansen.values()), 2), "hansen_loss_by_year": hansen,
            "tmf_deforestation_ha": round(sum(deforestation.values()), 2), "tmf_deforestation_by_year": deforestation,
            "tmf_degradation_ha": round(sum(degradation.values()), 2), "tmf_degradation_by_year": degradation}


# ---------- planned ----------

def run_whisp(geometry: Geometry) -> dict[str, Any]:
    """openforis-whisp stats + risk_pcrop for one plot."""
    raise NotImplementedError


def coffee_probability(geometry: Geometry, year: int = 2020) -> dict[str, Any]:
    """Forest Data Partnership coffee model: share of plot above threshold."""
    raise NotImplementedError


def ndvi_series(geometry: Geometry, years: range = range(2019, 2026)) -> dict[str, Any]:
    """Sentinel-2 dry-season (Jan-Mar) median NDVI per year, for the plot or one loss block."""
    raise NotImplementedError


def locate_loss(geometry: Geometry) -> dict[str, Any]:
    """Loss patches after 2020: size, count, distance to plot edge, overlap with 2020 coffee/forest."""
    raise NotImplementedError


def boundary_sensitivity(geometry: Geometry, buffer_m: float = 30) -> dict[str, Any]:
    """Re-run loss checks on the plot grown and shrunk by buffer_m; report whether the result changes."""
    raise NotImplementedError
