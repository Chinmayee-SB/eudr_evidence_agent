"""Download the EUDR texts (if needed), split them into citable chunks and build both indexes.

Usage:  python scripts/build_index.py [--refresh]
"""

from __future__ import annotations

import argparse
from collections import Counter

from eudr_agent.rag.chunk import build_chunks, save_chunks
from eudr_agent.rag.fetch import fetch_all
from eudr_agent.rag.vector import build_vector_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download the texts again")
    args = ap.parse_args()

    print("Fetching EUDR texts from the EU Publications Office ...")
    fetch_all(force=args.refresh)
    chunks = build_chunks()
    path = save_chunks(chunks)

    kinds = Counter(c.id.split(":")[0] for c in chunks)
    print(f"Saved {len(chunks)} chunks to {path}")
    for kind, n in kinds.most_common():
        print(f"  {kind:<16} {n}")

    print("Embedding chunks into the vector index (the first run downloads the model) ...")
    print(f"Saved {build_vector_index(chunks)} vectors")


if __name__ == "__main__":
    main()
