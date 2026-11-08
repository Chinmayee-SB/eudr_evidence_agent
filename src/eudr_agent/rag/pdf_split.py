"""Split the hand-added PDFs (Commission guidance, FAQ, Indian law) along their own headings.

One page is a poor unit: a page holds parts of two or three answers, the search model reads only about the
first 1,000 characters of a passage, and a citation to "p.31" does not tell the officer which answer was meant.
So each known document is split at its own numbered headings and cited the way a reader would cite it:

    EC FAQ Q4.10                 one FAQ question and its answer
    EC Guidance §11.4(d)         one section of the Commission guidance
    Karnataka Forest Act s.24    one section of the Act

A heading counts only if its number comes next in sequence (4.10 after 4.9, s.24 after s.23). That one rule
keeps out numbered lists inside an answer, cross-references at the start of a line, and amendment footnotes
("3. Inserted by Act 20 of 2001"), which look exactly like headings. Table-of-contents pages are skipped.

A PDF that is not listed in MANUAL_DOCS is split per page, as before, so any document can still be added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1500  # longer sections are cut into parts at sentence ends; see the module docstring


@dataclass(frozen=True)
class Doc:
    label: str   # citation prefix
    kind: str    # source type, see chunk.KINDS
    style: str   # which heading rules to use: faq, guidance, act, pages


MANUAL_DOCS = {
    "EC_FAQ_v5_2026": Doc("EC FAQ", "guidance", "faq"),
    "EC_Guidance_2026": Doc("EC Guidance", "guidance", "guidance"),
    "Karnataka_Forest_Act_1963": Doc("Karnataka Forest Act", "national_law", "act"),
}


@dataclass
class Section:
    ref: str
    title: str
    page: int
    lines: list[str]


# ---------- reading ----------

PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
AMENDMENT_NOTE = re.compile(r"^\s*\d+\.\s*(Subs|Substituted|Inserted|Ins\.|Omitted|Added|Renumbered|Re-numbered|"
                            r"Amended|It has come)", re.I)
TOC_ENTRY = re.compile(r"^\s*\d+[A-Z]?\.(\d+\.?)*\s+\S")
DOTTED_ENTRY = re.compile(r"\.{5,}\s*\d+\s*$")  # "... 12": a contents line (a fee table's dots end in words)


def _is_contents_page(lines: list[str]) -> bool:
    """A table of contents: dotted leaders to page numbers, or a page that is mostly numbered lines."""
    if sum(bool(DOTTED_ENTRY.search(line)) for line in lines) >= 3:
        return True
    entries = [line for line in lines if TOC_ENTRY.match(line)]
    return len(entries) >= 10 and len(entries) >= 0.5 * len(lines)


def read_pages(path: Path) -> list[tuple[int, list[str]]]:
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        lines = [line.strip() for line in (page.extract_text() or "").splitlines()]
        lines = [line for line in lines if line and not PAGE_NUMBER.match(line)]
        if lines and not _is_contents_page(lines):
            pages.append((number, lines))
    return pages


# ---------- heading rules, one per style ----------

def _upper_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


class FAQ:
    """Headings like '4.10. How to apply ...?' or '2.8.1 Are retreaded ...?'; each must come after the last."""
    HEADING = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?\.?\s+([A-Z(“\"‘].*)$")

    def __init__(self, label: str):
        self.label, self.last = label, (0, 0, 0)

    def heading(self, line: str) -> tuple[str, str] | None:
        m = self.HEADING.match(line)
        if not m:
            return None
        num = (int(m[1]), int(m[2]), int(m[3] or 0))
        if not (num > self.last and num[0] - self.last[0] <= 1):
            return None
        self.last = num
        code = ".".join(str(n) for n in num if n) if num[2] else f"{num[0]}.{num[1]}"
        return f"{self.label} Q{code}", m[4]


class Guidance:
    """Three levels: '11. AGRICULTURAL USE' (capitals), then '4. Definition of ...', then 'd) Clarification ...'.
    Each level must be the next number or letter under its parent."""
    TOP = re.compile(r"^(\d{1,2})\.\s+(.+)$")
    LETTER = re.compile(r"^([a-h])\)\s+([A-Z‘'“].*)$")

    def __init__(self, label: str):
        self.label, self.top, self.mid, self.letter = label, 0, 0, ""

    def _ref(self) -> str:
        ref = f"{self.label} §{self.top}" + (f".{self.mid}" if self.mid else "")
        return ref + (f"({self.letter})" if self.letter else "")

    def heading(self, line: str) -> tuple[str, str] | None:
        m = self.TOP.match(line)
        if m and int(m[1]) == self.top + 1 and _upper_share(m[2]) >= 0.8:
            self.top, self.mid, self.letter = int(m[1]), 0, ""
            return self._ref(), m[2]
        if m and self.top and int(m[1]) == self.mid + 1 and m[2][:1].isupper() and _upper_share(m[2]) < 0.8:
            self.mid, self.letter = int(m[1]), ""
            return self._ref(), m[2]
        m = self.LETTER.match(line)
        if m and self.top and ord(m[1]) == ord(self.letter or "`") + 1:
            self.letter = m[1]
            return self._ref(), m[2]
        return None


class Act:
    """Sections like '24. Acts prohibited in reserved forests- Any person who ...'. A section number must be a
    little above the last one (sections can be omitted) or the same number with a later letter (104A).
    Amended sections start with an editor's marker ('1[83. Rights of ...'), and a few use '37-' for '37.'."""
    HEADING = re.compile(r"^(?:\d?\[)?(\d{1,3})([A-Z]?)\s*[.-]\s+([A-Z].*)$")
    TITLE_END = re.compile(r"\s*(?:\s-|-\s|–|—|\.\s)")

    def __init__(self, label: str):
        self.label, self.last = label, (0, "")

    def heading(self, line: str) -> tuple[str, str] | None:
        m = self.HEADING.match(line)
        if not m or AMENDMENT_NOTE.match(line):
            return None
        num, letter = int(m[1]), m[2]
        last_num, last_letter = self.last
        if not ((last_num < num <= last_num + 3) or (num == last_num and letter > last_letter)):
            return None
        self.last = (num, letter)
        title, *rest = self.TITLE_END.split(m[3], maxsplit=1)
        self.rest = rest[0] if rest else ""  # the section text often starts on the heading line
        return f"{self.label} s.{num}{letter}", title[:120]


STYLES = {"faq": FAQ, "guidance": Guidance, "act": Act}


# ---------- splitting ----------

def sections(path: Path, doc: Doc) -> list[Section]:
    rules = STYLES[doc.style](doc.label)
    found: list[Section] = []
    intro = Section(f"{doc.label} introduction", "Introduction", 1, [])
    for page, lines in read_pages(path):
        for line in lines:
            if doc.style == "act" and AMENDMENT_NOTE.match(line):
                continue  # footnotes about amendments interrupt the section text
            head = rules.heading(line)
            if head:
                rest = getattr(rules, "rest", "")
                found.append(Section(head[0], head[1], page, [rest] if rest else []))
            else:
                (found[-1] if found else intro).lines.append(line)
    return ([intro] if len(" ".join(intro.lines)) > 200 else []) + found


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Cut at sentence ends so each part stays under the limit (a single overlong sentence stays whole)."""
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return parts + [current] if current else parts


def pages(path: Path) -> list[Section]:
    """Fallback for a PDF that is not in MANUAL_DOCS: one section per page."""
    return [Section(f"{path.stem} p.{page}", path.stem, page, lines) for page, lines in read_pages(path)]
