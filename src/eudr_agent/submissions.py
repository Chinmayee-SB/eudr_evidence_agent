"""Plots submitted on the desk: a boundary file, the declared details and land records, saved per submission.

    data/submissions/SB-001/
        boundary.geojson     the polygon as uploaded (KML or GeoJSON, converted), coordinates unchanged
        details.json         reference, declared crop, crop-since year, declared area, when it was submitted
    data/plots/documents/uploads/SB-001/
        uploaded_*.pdf|md|txt  land records, saved by documents.py (the folder the input check permits)

Both folders are gitignored: submissions may be real grower data and must never enter the repository.

This module only reads the file and saves it. It does not judge the plot: a polygon outside Karnataka, a
non-coffee crop, a future crop year or a tampered record is saved as submitted and then stopped, with its
reason, by the input check (guardrail.py) before any agent runs. A file that holds no usable shape at all is
refused here, since there is nothing to check.

The reference (the submitter's own name for the plot) stays on this machine: it is not part of the plot the
agents see, so it is never in a prompt or a LangSmith trace.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from eudr_agent import documents
from eudr_agent.config import DATA_DIR
from eudr_agent.geo import tools as geo_tools
from eudr_agent.state import PlotInput

SUBMISSIONS_DIR = DATA_DIR / "submissions"
PREFIX = "SB"
BOUNDARY_SUFFIXES = {".kml", ".geojson", ".json"}
MAX_BOUNDARY_BYTES = 2 * 1024 * 1024
CROPS = ("coffee", "cocoa", "rubber", "tea", "other")  # only coffee is in scope; the rest show the input check


def _ring(text: str) -> list[list[float]]:
    """A KML coordinate string ("lon,lat[,alt] lon,lat ...") as GeoJSON positions, altitude dropped."""
    try:
        return [[float(x) for x in point.split(",")[:2]] for point in text.split()]
    except ValueError as e:
        raise ValueError("the KML coordinates are not numbers") from e


def _from_kml(text: str) -> dict[str, Any]:
    # read with patterns, not an XML parser: nothing in the file is expanded or fetched
    polygons = re.findall(r"<(?:\w+:)?Polygon\b.*?</(?:\w+:)?Polygon>", text, re.S)
    points = re.findall(r"<(?:\w+:)?Point\b.*?</(?:\w+:)?Point>", text, re.S)
    if len(polygons) > 1:
        raise ValueError(f"the file holds {len(polygons)} polygons; submit one plot per file")
    if polygons:
        outer = re.search(r"<(?:\w+:)?outerBoundaryIs>.*?<(?:\w+:)?coordinates>(.*?)</", polygons[0], re.S)
        if not outer:
            raise ValueError("the polygon has no outer boundary")
        return {"type": "Polygon", "coordinates": [_ring(outer.group(1))]}
    if len(points) == 1:
        coords = re.search(r"<(?:\w+:)?coordinates>(.*?)</", points[0], re.S)
        if coords:
            return {"type": "Point", "coordinates": _ring(coords.group(1))[0]}
    raise ValueError("no polygon was found in the KML file")


def _from_geojson(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("the file is not valid GeoJSON") from e
    if not isinstance(data, dict):
        raise ValueError("the file is not valid GeoJSON")
    if data.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") for f in data.get("features", []) if isinstance(f, dict)]
    elif data.get("type") == "Feature":
        geometries = [data.get("geometry")]
    else:
        geometries = [data]
    geometries = [g for g in geometries if isinstance(g, dict)]
    shapes: list[dict[str, Any]] = []
    for g in geometries:
        if g.get("type") == "MultiPolygon":
            shapes += [{"type": "Polygon", "coordinates": part} for part in g.get("coordinates", [])]
        elif g.get("type") in {"Polygon", "Point"}:
            shapes.append(g)
    if not shapes:
        raise ValueError("no polygon was found in the GeoJSON file")
    if len(shapes) > 1:
        raise ValueError(f"the file holds {len(shapes)} shapes; submit one plot per file")
    shape = shapes[0]
    if shape["type"] == "Point":
        return {"type": "Point", "coordinates": [float(x) for x in shape["coordinates"][:2]]}
    return {"type": "Polygon", "coordinates": [[[float(x) for x in p[:2]] for p in ring]
                                               for ring in shape["coordinates"]]}


def parse_boundary(name: str, data: bytes) -> dict[str, Any]:
    """The one plot shape in a KML or GeoJSON file, as GeoJSON. Raises ValueError with a reason to show."""
    suffix = Path(name).suffix.lower()
    if suffix not in BOUNDARY_SUFFIXES:
        raise ValueError("the boundary must be a .kml or .geojson file")
    if len(data) > MAX_BOUNDARY_BYTES:
        raise ValueError(f"the boundary file is over {MAX_BOUNDARY_BYTES // (1024 * 1024)} MB")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("the boundary file is not text (a .kmz must be unzipped to .kml first)") from e
    return _from_kml(text) if suffix == ".kml" else _from_geojson(text)


def _next_id() -> str:
    used = [int(m.group(1)) for p in SUBMISSIONS_DIR.glob(f"{PREFIX}-*")
            if (m := re.fullmatch(rf"{PREFIX}-(\d+)", p.name))] if SUBMISSIONS_DIR.exists() else []
    return f"{PREFIX}-{max(used, default=0) + 1:03d}"


def computed_area_ha(geometry: dict[str, Any]) -> float | None:
    area = geo_tools.validate_geometry(geometry)["values"].get("computed_area_ha")
    return round(area, 2) if area else None


def create(boundary_name: str, boundary: bytes, *, reference: str = "", declared_crop: str = "coffee",
           crop_since: int | None = None, declared_area_ha: float | None = None,
           records: list[tuple[str, bytes]] = ()) -> tuple[str, list[str]]:
    """Save a new submission. Returns its plot id and the land records that were refused, with reasons.
    Raises ValueError, saving nothing, if the boundary file holds no usable shape."""
    geometry = parse_boundary(boundary_name, boundary)
    plot_id = _next_id()
    folder = SUBMISSIONS_DIR / plot_id
    folder.mkdir(parents=True)
    (folder / "boundary.geojson").write_text(json.dumps(
        {"type": "Feature", "properties": {"plot_id": plot_id}, "geometry": geometry}, indent=1) + "\n")
    details = {"plot_id": plot_id, "reference": reference.strip()[:120], "declared_crop": declared_crop,
               "crop_since": crop_since, "declared_area_ha": declared_area_ha,
               "boundary_file": Path(boundary_name).name,
               "submitted": datetime.datetime.now().isoformat(timespec="seconds")}
    (folder / "details.json").write_text(json.dumps(details, indent=1) + "\n")
    refused = []
    for name, data in records:
        try:
            documents.save_upload(plot_id, name, data)
        except ValueError as error:
            refused.append(f"{name}: {error}")
    return plot_id, refused


def details(plot_id: str) -> dict[str, Any]:
    path = SUBMISSIONS_DIR / plot_id / "details.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_submissions() -> dict[str, PlotInput]:
    """Every saved submission as the plot the agents see (no reference). Newest first."""
    plots: dict[str, PlotInput] = {}
    if not SUBMISSIONS_DIR.exists():
        return plots
    for folder in sorted(SUBMISSIONS_DIR.glob(f"{PREFIX}-*"), reverse=True):
        try:
            geometry = json.loads((folder / "boundary.geojson").read_text())["geometry"]
            info = json.loads((folder / "details.json").read_text())
        except (OSError, ValueError, KeyError):
            continue  # a half-written or damaged submission is skipped, not fatal for the desk
        plot = PlotInput(plot_id=folder.name, geometry=geometry,
                         area_ha=info.get("declared_area_ha") or computed_area_ha(geometry),
                         declared_crop=info.get("declared_crop") or "unknown", documents=[])
        if info.get("crop_since") is not None:
            plot["crop_since"] = info["crop_since"]
        plots[folder.name] = plot
    return plots
