"""Mock documents uploaded for a plot: checked, saved per plot and readable by the geo-analyst.

Uploads live under data/plots/documents/uploads/<plot_id>/, which git ignores. Each plot has its own folder,
so one plot's agent can never be offered another plot's file. Names get an "uploaded_" prefix so an upload
can never replace a sample record of the same name.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader

from eudr_agent.config import PLOTS_DIR
from eudr_agent.state import PlotInput

UPLOADS_DIR = PLOTS_DIR / "documents" / "uploads"
ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}
MAX_BYTES = 5 * 1024 * 1024


def text_from_bytes(data: bytes, suffix: str) -> str:
    """The document's text. Raises ValueError if there is none to read, so a bad file is refused, not skipped."""
    if suffix == ".pdf":
        try:
            pages = PdfReader(io.BytesIO(data)).pages
            text = "\n\n".join(page.extract_text() or "" for page in pages).strip()
        except Exception as e:  # pypdf raises many types on damaged or locked files
            raise ValueError("this PDF could not be opened") from e
        if not text:
            raise ValueError("this PDF has no readable text; it may be a scan")
        return text
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as e:
        raise ValueError("this file is not UTF-8 text") from e
    if not text:
        raise ValueError("this file is empty")
    return text


def read_text(path: Path) -> str:
    return text_from_bytes(path.read_bytes(), path.suffix.lower())


def save_upload(plot_id: str, name: str, data: bytes) -> Path:
    """Check one upload and store it for this plot. Uploading the same name again replaces it."""
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)  # drops any folder in the name
    suffix = Path(stem).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"only {', '.join(sorted(ALLOWED_SUFFIXES))} files are accepted")
    if len(data) > MAX_BYTES:
        raise ValueError(f"the limit is {MAX_BYTES // (1024 * 1024)} MB per file")
    text_from_bytes(data, suffix)
    path = UPLOADS_DIR / plot_id / f"uploaded_{stem}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def uploads_for(plot_id: str) -> list[Path]:
    folder = UPLOADS_DIR / plot_id
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES) if folder.exists() else []


def remove_upload(plot_id: str, file_name: str) -> None:
    """Remove by name from this plot's uploads only, so a sample record can never be deleted here."""
    for path in uploads_for(plot_id):
        if path.name == file_name:
            path.unlink()


def with_uploads(plot: PlotInput) -> PlotInput:
    """The same plot, with its uploaded documents added to the ones the agent may open."""
    return PlotInput(**{**plot, "documents": [*plot.get("documents", []),
                                              *(str(p) for p in uploads_for(plot["plot_id"]))]})
