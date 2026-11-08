"""Turn the downloaded texts into citable chunks.

One chunk = one citable unit, so every answer can point to an exact place:
  - Article 2 is split per definition         -> "Art. 2(4)"
  - other articles are split per paragraph    -> "Art. 9(1)"
  - recitals are one chunk each               -> "Recital 38"
  - annexes are one chunk each                -> "Annex I"
  - our own notes                             -> "Note: ..."
  - hand-added PDFs, along their headings     -> "EC FAQ Q4.10", "Karnataka Forest Act s.24" (see pdf_split.py)

Every chunk also has a `kind`, so an answer can show how much weight each citation carries (see KINDS).
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from eudr_agent.config import CHUNKS_PATH, MANUAL_DIR, NOTES_DIR, RAW_DIR
from eudr_agent.rag import pdf_split
from eudr_agent.rag.fetch import SOURCES, source_url

# Source types, and how each is shown next to a citation.
KINDS = {
    "law": "EU law (binding)",
    "recital": "EU recital (explains the law; not binding on its own)",
    "guidance": "Commission guidance (not legally binding)",
    "national_law": "Indian law (country of production)",
    "note": "our dataset note (not a legal source)",
}
KINDS_SHORT = {"law": "binding", "recital": "recital", "guidance": "guidance, not binding",
               "national_law": "Indian law", "note": "our note"}
LEGAL_KINDS = {"law", "recital", "guidance", "national_law"}  # an answer must rest on at least one of these


@dataclass
class Chunk:
    id: str
    ref: str  # human citation, e.g. "EUDR Art. 2(4)"
    title: str
    text: str
    source: str
    url: str
    kind: str = "law"  # one of KINDS


_PAR = "\u0000PAR{}\u0000"


def _clean(fragment: str) -> str:
    fragment = re.sub(r'<p class="modref">.*?</p>', " ", fragment, flags=re.S)
    fragment = re.sub(
        r'<span class="no-parag">\s*(\d+)\.\s*</span>', lambda m: " " + _PAR.format(m.group(1)) + " ", fragment
    )
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"[▼►◄][A-Z]?\d*", " ", text)  # amendment markers like ▼M2
    return re.sub(r"\s+", " ", text).strip()


def _sections(page: str, prefix: str) -> list[tuple[str, str]]:
    """Return (number, html) for each <div class="eli-subdivision" id="{prefix}_N">. N may be an
    inserted article such as 4a, which is a different article from 4 and must keep its own citation."""
    marks = list(re.finditer(rf'<div class="eli-subdivision" id="{prefix}_(\d+[a-z]?)"', page))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(page)
        out.append((m.group(1), page[m.start():end]))
    return out


def _articles(page: str, src_title: str, url: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for num, frag in _sections(page, "art"):
        title_m = re.search(r'class="stitle-article-norm">(.*?)</p>', frag, flags=re.S)
        title = _clean(title_m.group(1)) if title_m else f"Article {num}"
        text = _clean(frag)
        text = re.sub(rf"^Article {num}\s+{re.escape(title)}\s*", "", text)
        text = re.sub(r"\s+CHAPTER \d+\b.*$", "", text)  # the next chapter's heading sits inside this div

        if num == "38":  # the last article runs into the annexes in this HTML
            text, _, annex_text = text.partition("ANNEX I ")
            for k, part in enumerate(re.split(r"\bANNEX (?=I{1,3}\b)", "ANNEX I " + annex_text)):
                if part.strip():
                    name = "Annex " + part.split()[0]
                    chunks.append(Chunk(f"eudr:annex{k}", f"EUDR {name}", name, part.strip()[:6000], src_title, url))

        if num == "2":  # one chunk per definition
            intro, _, body = text.partition("following definitions apply:")
            for m in re.finditer(r"\((\d+)\)\s+(‘.*?)(?=\s\(\d+\)\s+‘|$)", body, flags=re.S):
                chunks.append(Chunk(f"eudr:art2:{m.group(1)}", f"EUDR Art. 2({m.group(1)})",
                                    "Definitions", m.group(2).strip(), src_title, url))
            continue

        parts = re.split(r"\u0000PAR(\d+)\u0000", text)
        if len(parts) == 1:
            chunks.append(Chunk(f"eudr:art{num}", f"EUDR Art. {num}", title, text, src_title, url))
            continue
        lead = parts[0].strip()
        for par_num, par_text in zip(parts[1::2], parts[2::2]):
            body = (lead + " " if lead and par_num == "1" else "") + par_text.strip()
            chunks.append(Chunk(f"eudr:art{num}:{par_num}", f"EUDR Art. {num}({par_num})", title, body, src_title, url))
    return chunks


def _recitals(page: str, key: str, label: str, src_title: str, url: str) -> list[Chunk]:
    chunks = []
    page = page.partition("HAVE ADOPTED THIS REGULATION")[0]  # otherwise the last recital runs to the end
    for num, frag in _sections(page, "rct"):
        text = re.sub(r"^\(\d+\)\s*", "", _clean(frag))
        chunks.append(Chunk(f"{key}:rct{num}", f"{label} Recital {num}", "Recital", text, src_title, url, "recital"))
    return chunks


def _markdown(path: Path, ref_prefix: str, kind: str) -> list[Chunk]:
    chunks = []
    for block in re.split(r"\n(?=## )", path.read_text(encoding="utf-8")):
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("## "):
            continue
        heading = lines[0][3:].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        cid = f"{path.stem}:{re.sub(r'[^a-z0-9]+', '-', heading.lower()).strip('-')}"
        chunks.append(Chunk(cid, f"{ref_prefix}: {heading}", heading, body, path.name, "", kind))
    return chunks


def _pdf(path: Path) -> list[Chunk]:
    """A known document is split along its headings; any other PDF per page. Long sections become parts."""
    doc = pdf_split.MANUAL_DOCS.get(path.stem)
    found = pdf_split.sections(path, doc) if doc else pdf_split.pages(path)
    kind = doc.kind if doc else "guidance"
    chunks = []
    for section in found:
        parts = pdf_split.split_long(re.sub(r"\s+", " ", " ".join(section.lines)).strip())
        for n, text in enumerate(parts, start=1):
            if len(text) < 80:  # a deleted question or a stray heading
                continue
            ref = section.ref + (f" (part {n})" if len(parts) > 1 else "")
            cid = f"{path.stem}:{re.sub(r'[^A-Za-z0-9.]+', '-', ref)}"
            chunks.append(Chunk(cid, ref, section.title, text, path.name, f"{path.name}#page={section.page}", kind))
    return chunks


def build_chunks() -> list[Chunk]:
    by_key = {s.key: s for s in SOURCES}
    read = lambda key: (RAW_DIR / f"{key}.html").read_text(encoding="utf-8")  # noqa: E731

    cons = by_key["eudr_consolidated"]
    chunks = _articles(read(cons.key), cons.title, source_url(cons))
    orig = by_key["eudr_original"]
    chunks += _recitals(read(orig.key), "eudr", "EUDR", orig.title, source_url(orig))
    amend = by_key["eudr_amend_2025"]
    chunks += _recitals(read(amend.key), "amend2025", "Reg. 2025/2650", amend.title, source_url(amend))

    for md in sorted(NOTES_DIR.glob("*.md")):
        chunks += _markdown(md, "Note", "note")
    for f in sorted(MANUAL_DIR.glob("*")):
        if f.suffix.lower() == ".pdf":
            chunks += _pdf(f)
        elif f.suffix.lower() in {".md", ".txt"} and f.name.lower() != "readme.md":
            chunks += _markdown(f, f.stem, "guidance")
    return chunks


def save_chunks(chunks: list[Chunk]) -> Path:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return CHUNKS_PATH


def load_chunks() -> list[Chunk]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("No index yet. Run: python scripts/build_index.py")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
