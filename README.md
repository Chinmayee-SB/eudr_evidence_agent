# EUDR Evidence Agent for Karnataka Coffee

The agent answers one question for a coffee exporter's compliance officer:
**"Can coffee from this plot go into an EU-bound lot, and if not, what evidence is missing?"**
It combines satellite checks with the EU Deforestation Regulation (EUDR) text and returns
Clear / Needs evidence / High risk with an evidence file. A person approves every verdict.

## Status (12 Nov 2026)

Everything in the plan is built and working (195 tests pass). The officer's desk is called **Canopy Check**.

| Part | Status |
|---|---|
| Input guardrail | **Working** (`guardrail.py`): blocks, drops or warns before any agent runs |
| Geo-analyst agent + satellite tools | **Working**: 8 tools and a ReAct loop that picks only the checks a plot needs; Whisp runs as a separate baseline |
| Supervisor | **Working**: review, routing, budgets, officer pause, saved cases (SQLite or Postgres) |
| Rules agent: agentic RAG over the EUDR, Commission guidance and Karnataka law | **Working**: legal-citation rule, claim checks, reranker; faithfulness 1.00 on 37 questions |
| Verdict rules | **Working** (`supervisor/verdict_rules.py`): 7 fixed rules, each verdict with its reasons, legal basis and caveats |
| Canopy Check desk | **Working**: landing page; *New submission* (upload a KML/GeoJSON boundary and land records), *Submitted plots*, *Demo plots*; each step shown live, a card for every tool, legal answers with their checks, officer approves / overrides / re-checks with new records (once), PDF evidence pack, *How it works* view with the evaluations. Plain wording throughout (`wording.py`) |
| Reliability | **Working**: timeouts and retries on LLM and Earth Engine calls; one case per plot, with at most one re-check |
| Tracing | **Working**: local JSONL per case, and LangSmith when its key is set (`observability.py`) |

**End to end** (`eval/run_end_to_end.py`, 10 Nov 2026, real agents, draft verdicts, the 7 demo plots):

| Plot | Kind | Label | FAO Whisp | Ours | Deciding rule |
|---|---|---|---|---|---|
| C-01 | coffee | Clear | more info needed | **Clear** | 4: no deforestation after 2020 |
| C-02 | coffee | Clear | more info needed | **Clear** | 6: small, stable loss on coffee farmed before 2020 |
| K-01 | coffee | Clear | low | **Needs evidence** | 7: most of the loss disappears when the boundary moves 30 m |
| N-01 | protected forest | High risk | more info needed | **High risk** | 1: inside a reserve forest in force |
| S-01 | protected forest | Clear (imagery only) | not run | **High risk** | 1: inside a protected area in force |
| T-01 | tea, Kerala | Clear | not run | **Stopped** | input check: not coffee, outside Karnataka |
| W-01 | rubber | Clear | not run | **Stopped** | input check: not coffee |

- C-01, C-02 and N-01 agree with the label. K-01 asks for evidence rather than disagreeing.
- S-01's labeller judged the imagery only ("no obvious new clearing"). The plot is inside a protected area in force, which is high risk under Art. 3(b) and 2(40) whatever the imagery shows, so the expected answer in `data/plots/README.md` is High risk.
- Ours catches both protected-area plots. Whisp does not, because its risk score ignores protected areas.
- 7 plots are too few to claim accuracy. This shows the system works end to end and where it is cautious.

**Plots submitted on the desk** (16 more plots, kept outside the repo in `../data/`; three run on 10 Nov 2026):

| Plot | Label | Ours | Why |
|---|---|---|---|
| K-05 | Clear (AI labeller) | **Clear** (rule 4) | no loss on either dataset |
| K-03 | Clear | **Needs evidence** (rule 7) | Hansen 0 ha, TMF 1.43 ha: the datasets disagree |
| C-05 | Needs evidence (AI labeller) | **Clear** (rule 4) | no loss on either dataset |

The AI labeller's reading of the imagery is a second opinion, not ground truth.

## Known limits

- **Coffee map** (`coffee_probability`, Forest Data Partnership): it scored about 0 on every shade-coffee plot tested (C-01, C-02, K-03). It stays as a tool, but the desk marks it "not reliable for shade-grown coffee in India" and **no verdict rule uses it**.
- **Loss datasets cannot tell pruning from clearing.** Where Hansen and TMF disagree, the verdict is Needs evidence and the officer decides (a field visit or recent imagery).
- **Proposed protected areas** are shown on the evidence card but do not change the verdict, since they are not legally in force.
- **Land records are mocked**, and the crop-since years of the test plots are mocked. The input check screens documents for obvious prompt injection only.
- **Hand-drawn boundaries are approximate**, so loss within 30 m of the edge is flagged and the boundary is moved 30 m in and out.
- **The hero photo** (`app/static/hero_coffee.jpg`) is a stock preview, for the demo only.

## Layout

```
app/streamlit_app.py   the Canopy Check desk: landing page, submissions, case view, officer sign-off, PDF
app/static/            landing page photo
src/eudr_agent/
  config.py        settings from .env
  llm.py           any OpenAI-compatible model, with timeouts and retries
  tracing.py       per-step JSONL traces (runs/traces/)
  observability.py LangSmith tracing, when its key is set
  state.py         the plot's case file shared by all agents
  plots.py         loads a demo plot as the agent's input
  submissions.py   saves a plot submitted on the desk (KML or GeoJSON) as SB-001, SB-002 ...
  documents.py     reads uploaded land records (PDF, MD, TXT)
  guardrail.py     input check before any agent runs
  wording.py       plain words for the desk and the PDF (no internal terms)
  report.py        the PDF evidence pack
  rag/             rules agent
    fetch.py       downloads the EUDR texts from the EU Publications Office
    chunk.py       splits them into citable chunks (Art. 2(4), Art. 38(2), Recital 46 ...)
    pdf_split.py   splits the guidance, FAQ and Karnataka Forest Act PDFs
    index.py       BM25 + vector search, merged with Reciprocal Rank Fusion
    vector.py      local Chroma vector index (no account needed)
    rerank.py      cross-encoder reranker (bge-reranker-base)
    agent.py       LangGraph loop: retrieve -> grade -> rewrite/retry -> answer -> check citations -> verify
    verify.py      quote check and claim check against the passages
    cache.py       verified-answer cache
    prompts.py
  geo/agent.py     geo-analyst: ReAct loop that picks tools one at a time
  geo/tools.py     satellite and document tools: protected areas, forest 2020, loss after 2020, loss location,
                   boundary shift, Sentinel-2 NDVI, coffee model (WDPA, GFC2020, Hansen, TMF, S2, FDP)
  supervisor/
    graph.py       guardrail -> geo-analyst <-> supervisor <-> rules agent -> verdict -> officer pause -> pack
    review.py      code review of the evidence: gaps, conflicts, checks not yet possible
    decide.py      the LLM proposes the next step; code enforces budgets and required evidence
    checkpoints.py saved cases: SQLite by default, Postgres if DATABASE_URL is set
    verdict_rules.py  fixed verdict table: 7 rules, first match decides
data/corpus/notes/   dataset limitation notes the agent can cite
data/corpus/manual/  Commission guidance, FAQ and Karnataka Forest Act PDFs
data/plots/          7 anonymised demo plots, mocked estate details and land records, labels
data/cache/geo/      saved satellite results, so the demo runs without Earth Engine
eval/                rule questions, RAG metrics, end-to-end run, Whisp baseline, saved results
scripts/             build_index.py, ask.py, run_case.py ...
tests/
```

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python scripts/check_setup.py          # checks the index, LLM chat and LLM tool calling
.venv/bin/python scripts/build_index.py          # downloads EUDR texts, splits them + the manual PDFs into ~830 citable pieces
.venv/bin/python scripts/run_geo_tools.py        # satellite checks on the test plots (needs EE_PROJECT, or the cache)
.venv/bin/python scripts/run_geo_agent.py K-01      # watch the geo-analyst choose tools for one plot
.venv/bin/python scripts/run_case.py K-01           # full case; stops for the officer and prints the case id
.venv/bin/python scripts/run_case.py --resume <case id> --approve --reason "..."
.venv/bin/python scripts/ask.py "Is shade-grown coffee under trees counted as forest?"
.venv/bin/python eval/run_rag_eval.py            # metrics on 25 rule questions (14 general, 11 as the supervisor asks)
.venv/bin/python scripts/run_whisp_baseline.py   # FAO Whisp on the test plots -> eval/whisp_baseline.csv
.venv/bin/python eval/run_rag_quality.py --answers  # RAG metrics: recall@k, MRR, nDCG, faithfulness ...
.venv/bin/python eval/run_end_to_end.py          # the full case on every plot, vs the labels and Whisp
.venv/bin/python -m pytest
.venv/bin/streamlit run app/streamlit_app.py     # the Canopy Check desk, at http://localhost:8501
```

**Submitting a plot on the desk.** Choose *New submission*, then upload the boundary (.kml from Google Earth, or
.geojson, one polygon), state the declared crop, crop-since year and (optionally) area, add land records (.pdf,
.md, .txt), and click *Submit and run the check*. The case runs straight away. The input check stops plots
outside Karnataka, non-coffee crops, broken shapes and tampered records before any agent sees them. Submissions
are saved in `data/submissions/` (gitignored, may hold real grower data) and listed under *Submitted plots*.
*Demo plots* are the labelled evaluation set. Your own reference for a plot is never sent to the agents.

No API key needed: without `LLM_API_KEY` the rules agent runs offline (keyword grading,
glossary query rewrite, extractive answers). Copy `.env.example` to `.env` and add a key
to switch grading, rewriting and answering to the LLM.

## How the rules agent works

0. **cache** (`rag/cache.py`): a question already answered, with the same texts, code and model, is answered from `data/cache/rules/` (same answer, no LLM calls). Only verified answers are kept.
1. **retrieve**: hybrid BM25 + vector search on two shelves: the regulation (articles, recitals, our notes; 12 candidates) and the supporting documents (Commission guidance, Indian law; 8 candidates). A local **reranker** (`rag/rerank.py`, bge-reranker-base ONNX, about 270 MB, downloaded once) reads the question with each candidate and orders all 20. The top 8 go on, keeping each search method's top 2 and at least 3 regulation passages. If the model can't load, the plain two-shelf order is used.
2. **grade**: one LLM call judges all 8 passages (falls back to one call per passage if the reply can't be read; keyword coverage offline). Binding law is listed first.
3. **rewrite**: if no relevant *legal* passage was found, or the answer fails its check, rewrite the query in the regulation's own terms and retry (max 2).
4. **answer**: answer only from relevant passages, citing each claim; each passage is shown with its source type.
5. **check_citations**: an answer must cite passages that were actually retrieved, **at least one of them a legal source**. Our dataset notes can support an answer but never carry it alone. Otherwise retry, then "no rule found".
6. **verify** (`rag/verify.py`): every quoted phrase must appear in the passages, word for word (code), and an LLM judge marks each sentence as supported, unsupported or making no claim, against the passages only. A failing answer is written again once, told what failed. If it still fails, the unsupported sentences are removed, and the agent abstains only if nothing supported with a legal citation is left. If the judge itself fails, the answer is kept and marked "not verified".

Every citation carries its source type, shown in the CLI, the desk and the PDF pack: EU law (binding), recital, Commission guidance (not legally binding), Indian law, or our note.

Every step is written to `runs/traces/<run_id>.jsonl`.

**Evaluation.** Two scripts:
- `eval/run_rag_eval.py`: did the agent find and cite the expected passage (25 questions)
- `eval/run_rag_quality.py`: standard RAG metrics on 37 questions in 3 sets: original, supervisor-worded, and **held-out**. The held-out questions were written after the search settings were chosen.

Retrieval (exact, first search, top 8, all 32 in-scope questions):

| Search | hit@8 | MRR | nDCG@8 |
|---|---|---|---|
| hybrid, no reranker | 0.97 | 0.53 | 0.57 |
| **hybrid + reranker, keeping each method's top 2 (as built)** | 0.94 | 0.64 | 0.66 |
| hybrid + reranker, reranker order only | 0.88 | 0.63 | 0.64 |

The reranker ranks the right passage higher, most of all on held-out questions (MRR 0.30 → 0.69), but loses one passage that plain search found.

Answers, end to end (LLM judge gpt-5.6-luna, RAGAS metric definitions; RAGAS itself isn't installed because it would downgrade the OpenAI SDK):

| Metric (37 questions) | No reranker | **With reranker (as built)** |
|---|---|---|
| Faithfulness (supported claims / claims) | 0.99 | 1.00 |
| Context precision | 0.68 | **0.83** |
| Citation precision | 0.95 | 0.92 |
| Answer relevance | 0.66 | 0.66 |
| Abstention correct | 37/37 | 37/37 |

Files: `eval/rag_quality_2026-11-09.json` (with reranker), `eval/rag_quality_no_reranker_2026-11-09.json`. On the 25-question eval, with the reranker: 95% cited correctly. The one miss is s08 (a plot in a reserve forest), answered without Art. 2(40).

Limits:
- The judge is the same model as the agent.
- In a hand check, 2/2 "supported" verdicts were right and 5/5 "unsupported" verdicts were wrong. All 5 came from context the judge lacked, and are now fixed.
- LLM grading varies between runs.
- The answer cache matches exact questions only. The LLM supervisor words its questions differently each run, so in live cases the cache rarely hits; it helps the code fallback, the desk and repeated identical questions.

## Tracing

Every case is traced twice:
- **Locally:** one JSONL file per case in `runs/traces/<case_id>.jsonl`. The desk's step-by-step view reads it, and it works offline.
- **In LangSmith**, when `.env` has `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` (see `.env.example`).

Each case is one trace tree:

```
case <id> (start)                                 thread_id = case id
  guardrail
  geo_analyst -> LangGraph -> think (LLM call) -> tool runs: protected_area_check, forest_2020, ...
  supervisor (LLM call)                           routing decision, overruled or not
  rules_agent -> retrieve -> search (retriever: passages, ref, source type, score) -> rerank
              -> grade (LLM) -> answer (LLM) -> check_citations -> verify (LLM)
  verdict                                         rule, reasons, legal basis, caveats
  officer                                         pauses here
case <id> (approve | override | documents_added)  same thread
  officer -> pack
```

- The officer's decision is logged as feedback (`officer_decision`) on the run that produced the draft verdict.
- Tests use a fake LangSmith client and never send anything (`tests/test_observability.py`).
- Sending a trace publishes its content (plot coordinates, document text, prompts) to LangSmith.

## Data sources

- Regulation (EU) 2023/1115, consolidated text of 26 Dec 2025 (CELEX 02023R1115-20251226)
- Regulation (EU) 2023/1115, original text, for its recitals (CELEX 32023R1115)
- Regulation (EU) 2025/2650, amending regulation, recitals (CELEX 32025R2650)

All from the EU Publications Office. Added by hand (see `data/corpus/manual/README.md`):
- Commission Guidance Document for the EUDR, C(2026) 4972, 13 July 2026
- Commission FAQ on EUDR implementation, version 5, April 2026
- Karnataka Forest Act 1963, from India Code (the EUDR's legality test refers to the country's own law, Art. 2(40))
