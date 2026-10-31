"""Download the official EUDR texts from the EU Publications Office (Cellar).

EUR-Lex's web pages block scripted downloads, but the Publications Office serves
the same Official Journal texts by CELEX number.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

from eudr_agent.config import RAW_DIR

CELLAR = "http://publications.europa.eu/resource/celex/{celex}"


@dataclass(frozen=True)
class Source:
    key: str
    celex: str
    title: str
    use: str  # which parts we index from it


SOURCES = [
    # Consolidated text = Regulation 2023/1115 with all amendments applied (latest: 2025/2650).
    Source("eudr_consolidated", "02023R1115-20251226",
           "Regulation (EU) 2023/1115 (EUDR), consolidated version of 26 Dec 2025", "articles + annexes"),
    # Original OJ text, used only for its recitals (consolidated versions drop recitals).
    Source("eudr_original", "32023R1115",
           "Regulation (EU) 2023/1115 (EUDR), original text of 31 May 2023", "recitals"),
    # The 2025 amending regulation (dates, simplifications).
    Source("eudr_amend_2025", "32025R2650",
           "Regulation (EU) 2025/2650 amending Regulation (EU) 2023/1115", "recitals"),
]


def source_url(src: Source) -> str:
    return CELLAR.format(celex=src.celex)


def fetch_all(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in SOURCES:
        path = RAW_DIR / f"{src.key}.html"
        if path.exists() and not force:
            paths.append(path)
            continue
        req = urllib.request.Request(
            source_url(src),
            headers={"Accept": "application/xhtml+xml", "Accept-Language": "en"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
        print(f"  downloaded {src.celex} -> {path.name} ({path.stat().st_size // 1024} KB)")
        paths.append(path)
    return paths
