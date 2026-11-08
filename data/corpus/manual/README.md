# Manually added documents

Documents that can't be downloaded by script. After adding or replacing one, rebuild the index:

```bash
.venv/bin/python scripts/build_index.py
```

| File | What it is | Source type | Cited as |
|---|---|---|---|
| `EC_Guidance_2026.pdf` | Commission Guidance Document for the EUDR, C(2026) 4972, 13 July 2026 | Commission guidance (not legally binding) | `EC Guidance §11.4(d)` |
| `EC_FAQ_v5_2026.pdf` | Commission FAQ on EUDR implementation, version 5, April 2026 | Commission guidance (not legally binding) | `EC FAQ Q4.10` |
| `Karnataka_Forest_Act_1963.pdf` | Karnataka Forest Act 1963 (Act 5 of 1964), from India Code | Indian law | `Karnataka Forest Act s.24` |

Where they came from:
- Guidance: green-forum.ec.europa.eu, "Deforestation Regulation implementation"
- FAQ: environment.ec.europa.eu/publications/faq-eudr-implementation_en
- Karnataka Forest Act: indiacode.nic.in, handle 123456789/7005

## How they are split

Each known file is split along its own numbered headings, not per page (see `src/eudr_agent/rag/pdf_split.py`).
That gives one piece per FAQ question, guidance section or Act section. A section over 1,500 characters
becomes parts, e.g. `(part 2)`.

**Adding a new document:**
- **Any other PDF** is split per page and cited as `<file name> p.N`.
- **To split it along its headings instead,** add it to `MANUAL_DOCS` in `pdf_split.py` with a label, a source type and a heading style.
- **Markdown or text files** are split on `## ` headings.

Keep the file names above. The splitter recognises a document by its file name, and the citations depend on it.
