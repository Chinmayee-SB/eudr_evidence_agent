# EUDR Evidence Agent for Karnataka Coffee

Agentic AI (26CD7PEAAI) mini-project, CSE (Data Science), BMSCE.
Team: Chinmayee S B (lead), Taran T, Alvin S T.

The agent answers one question for a coffee exporter's compliance officer:
**"Can coffee from this plot go into an EU-bound lot, and if not, what evidence is missing?"**
It combines satellite checks with the EU Deforestation Regulation (EUDR) text and returns
Clear / Needs evidence / High risk with an evidence file. A person approves every verdict.

## Status (31 Oct 2026)

| Part | Owner | Status |
|---|---|---|
| Rules agent: agentic RAG over EUDR texts | Chinmayee | **Working** (offline and LLM modes), eval in place |
| Geo-analyst agent + satellite tools | Taran | **Working**: 5 of 10 tools (`geo/tools.py`) and the ReAct loop (`geo/agent.py`) |
| Supervisor, reviewer, verdict rules, approval | Alvin | Not built; interfaces defined (`state.py`, `supervisor/verdict_rules.py`) |
| Evidence desk: upload, live trace, PDF evidence pack | Chinmayee | **Working** (`app/streamlit_app.py`, `documents.py`, `report.py`); no verdict yet |

## Layout

```
src/eudr_agent/
  config.py        settings from .env
  tracing.py       per-step JSONL traces (runs/traces/), stand-in for Langfuse
  llm.py           any OpenAI-compatible model (Groq, Gemini, Ollama ...)
  state.py         the plot's case file shared by all agents
  plots.py         loads a test plot as the agent's input
  rag/             rules agent
    fetch.py       downloads the EUDR texts from the EU Publications Office
    chunk.py       splits them into citable chunks (Art. 2(4), Art. 38(2), Recital 46 ...)
    index.py       BM25 + vector search, merged with Reciprocal Rank Fusion
    vector.py      local Chroma vector index (no account needed)
    agent.py       LangGraph loop: retrieve -> grade -> rewrite/retry -> answer -> check citations
    prompts.py
  geo/agent.py     geo-analyst: ReAct loop that picks tools one at a time
  geo/tools.py     satellite and document tools (Hansen, TMF, GFC2020, WDPA); Whisp and Sentinel-2 planned
  supervisor/      planned supervisor, reviewer, verdict rules
data/corpus/notes/ dataset limitation notes the agent can cite
data/corpus/manual/  drop the Commission guidance + FAQ PDFs here
data/plots/        4 anonymised test plots, mocked estate details and land records, manual labels
eval/              rule questions + evaluation script
scripts/           build_index.py, ask.py
tests/
```

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python scripts/check_setup.py          # checks the index, LLM chat and LLM tool calling
.venv/bin/python scripts/build_index.py          # downloads EUDR texts, builds 300+ chunks and both indexes
.venv/bin/python scripts/run_geo_tools.py        # satellite checks on the test plots (needs EE_PROJECT, or the cache)
.venv/bin/python scripts/run_geo_agent.py K-01      # watch the geo-analyst choose tools for one plot
.venv/bin/python scripts/ask.py "Is shade-grown coffee under trees counted as forest?"
.venv/bin/python eval/run_rag_eval.py            # metrics on 14 rule questions
.venv/bin/python -m pytest
.venv/bin/streamlit run app/streamlit_app.py     # the evidence desk, at http://localhost:8501
```

No API key needed: without `LLM_API_KEY` the rules agent runs offline (keyword grading,
glossary query rewrite, extractive answers). Copy `.env.example` to `.env` and add a key
to switch grading, rewriting and answering to the LLM.

## How the rules agent works

1. **retrieve**: hybrid BM25 + vector search over the regulation, its recitals, the 2025 amendment and our dataset notes.
2. **grade**: each passage is judged relevant or not (LLM yes/no, or keyword coverage offline).
3. **rewrite**: if nothing relevant, or the answer fails its check, rewrite the query in the regulation's own terms and retry (max 2).
4. **answer**: answer only from relevant passages, citing each claim.
5. **check_citations**: an answer must cite a passage that was actually retrieved; otherwise retry, then "no rule found".

Every step is written to `runs/traces/<run_id>.jsonl`.

## Data sources

- Regulation (EU) 2023/1115, consolidated text of 26 Dec 2025 (CELEX 02023R1115-20251226)
- Regulation (EU) 2023/1115, original text, for its recitals (CELEX 32023R1115)
- Regulation (EU) 2025/2650, amending regulation, recitals (CELEX 32025R2650)

All from the EU Publications Office. Commission guidance and FAQ are added by hand (see `data/corpus/manual/`).
