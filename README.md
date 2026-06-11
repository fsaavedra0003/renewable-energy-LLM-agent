# Horizon AI — Renewable Energy Asset Intelligence

A production-minded RAG + LangGraph agent for answering natural-language questions
about wind turbine and solar plant operations, maintenance, and fault codes.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
python demo.py                # builds vector store on first run (~10s), then 5 questions
python main.py                # interactive chat
python evaluation/eval.py     # 8 ground-truth test cases
```

---

## Data strategy — what gets embedded and why

| File | Rows | Approach | Reason |
|---|---|---|---|
| `manual_excerpts.txt` | ~2,100 words | **ChromaDB** | Only genuinely unstructured prose. 12 fault codes + 4 narrative sections. Semantic search is the correct tool for "what does E-1001 mean?" |
| `assets.csv` | 50 | **In-memory dict** | Pure key-value lookup. 50 rows loaded once at startup. Fuzzy match via Levenshtein handles typos. |
| `telemetry.csv` | 8,844 | **pandas** | Numeric data. Every question requires aggregation, ranking, or trend detection — impossible with semantic search. |
| `maintenance_logs.csv` | 188 | **pandas** | Only 15 unique description templates. All questions are structural (filter by asset, date, cost). `str.contains()` on fault codes is exact and instant. |

**Result: 17 documents embedded instead of 1,505. First-run embedding takes ~10s, not ~60s.**

---

## Architecture

```
User question
    │
    ▼
router_node          — GPT-4o-mini structured output → ToolPlan (ordered list)
    │
    ├─ asset_lookup  — assets registry dict, fuzzy match (Levenshtein ≤ 1 within class)
    ├─ telemetry_tool— pandas on telemetry.csv
    ├─ maintenance_tool pandas on maintenance_logs.csv
    └─ rag_tool      — ChromaDB semantic search on manual_docs
    │
    ▼
synthesiser_node     — merges all tool results → cited answer
    │
    ▼
guardrails_node      — validates citations → confidence score
```

### Why LangGraph over raw ReAct
Explicit `AgentState` TypedDict, built-in conditional routing, inspectable graph
at each node boundary. ReAct requires hand-rolling all of this.

### Embedding model: `text-embedding-3-small`
Best cost/quality ratio for retrieval. Outperforms `ada-002` on MTEB at ~5× lower cost.

### Vector store: ChromaDB
Zero-infra, local persistence, native metadata filtering (fault_code, manufacturer, model).
Runs with `pip install`, no external services.

---

## Project structure

```
horizon-ai/
├── data/                      # raw files — untouched
├── ingestion/
│   └── pipeline.py            # Task 1.1 — load + validate all 4 sources
├── retrieval/
│   └── vectorstore.py         # Task 1.2 — embed manual → ChromaDB
├── agent/
│   ├── rag_chain.py           # Task 2.1 — RAG chain with citations + retry
│   ├── agent.py               # Task 2.2 — LangGraph StateGraph
│   ├── guardrails.py          # Task 2.3 — confidence scoring
│   ├── faithfulness.py        # numeric + semantic faithfulness checks
│   ├── cache.py               # lru_cache singletons for config + pipeline data
│   └── tools/
│       ├── validators.py      # shared input/output validation helpers
│       ├── telemetry.py       # pandas queries on telemetry.csv
│       └── maintenance.py     # pandas queries on maintenance_logs.csv
├── evaluation/
│   └── eval.py                # Task 3.1 — 8 ground-truth test cases
├── notebooks/
│   └── 01_data_exploration.ipynb  # Task 1.3 — EDA + degradation detection
├── config.yaml                # all tuneable parameters
├── demo.py                    # Task 3.2 — 5 preset questions
├── main.py                    # Task 3.2 — interactive CLI
├── requirements.txt
└── .env.example
```

---

## Hidden degradation pattern

**Naive approach fails.** A raw "Jan-Feb avg vs May-Jun avg" comparison flags
~30/30 wind turbines as "degraded" (-44% to -67%) because wind output
naturally falls into summer — and flags **zero** solar assets, since solar
output naturally *rises* into summer. This comparison cannot separate the
deliberate anomaly from ordinary seasonality.

**Fixed approach** (`get_underperforming_assets()` in `agent/tools/telemetry.py`):
for each asset, compute its own seasonal ratio `Mar-Jun avg / Jan-Feb avg`,
then z-score that ratio **within its asset class** (wind vs solar compared
separately, since they move in opposite seasonal directions). Assets with
`z <= -1.5` — a clear statistical outlier relative to same-class peers — are
flagged.

This isolates **PV-005** (z=-2.47, ratio=1.59 vs solar peers averaging ~2.0+)
and **WT-006** (z=-1.62, ratio=0.56 vs wind peers averaging ~0.65+), each
corroborated by relevant fault codes in the telemetry (PV-005: `E-3002`
string underperformance; WT-006: `E-2002` power curve deviation). Demo Q5
exercises this. The EDA notebook (`notebooks/01_data_exploration.ipynb`)
visualises both the naive comparison's failure mode and the fixed result.

---

## Fixes applied (vs original submission)

| # | File | Fix |
|---|---|---|
| 1 | `.env` | Removed real API key from the repo; `.env` is now gitignored. Use `.env.example` as the template. |
| 1b | `agent/tools/telemetry.py`, `evaluation/eval.py`, `notebooks/` | **Fixed degradation detection.** The original Jan-Feb-vs-May-Jun % comparison conflated seasonality with the deliberate anomaly (flagged 30/30 wind turbines, 0 solar assets). Replaced with a seasonality-normalised z-score within asset class — see "Hidden degradation pattern" above. |
| 2 | `agent/rag_chain.py` | Added `tenacity` retry/backoff on LLM calls (RateLimitError, APIConnectionError). |
| 3 | `agent/agent.py` | Fixed dead-code bug in `rag_node`: was reading `source_documents` (never existed); now reads `source_scores` from `run_rag()` so `check_rag_results()` actually fires. |
| 4 | `agent/agent.py` | Removed duplicate `_infer_date_range_from_question`; uses `_infer_date_range` from `telemetry.py` (single source of truth). |
| 5 | `agent/agent.py` | `ChatOpenAI` instances promoted to module-level singletons; no longer re-instantiated on every call. |
| 6 | `agent/agent.py` | Router and synthesiser LLM calls wrapped with tenacity retry. |
| 7 | `agent/tools/validators.py` | Added `asset_ids_from_results()` and `asset_ids_from_text()` as shared helpers. |
| 8 | `agent/tools/telemetry.py` | Removed duplicate `_asset_ids_from_*` helpers; imports from validators. Fixed `get_underperforming_assets()` to use cached registry instead of re-reading `assets.csv`. |
| 9 | `agent/tools/maintenance.py` | Removed duplicate `_asset_ids_from_*` helpers; imports from validators. `_load()` uses pipeline cache. |
| 10 | `ingestion/pipeline.py` | `_load_config()` removed; uses `agent.cache.get_config()`. Added warning for unparseable install_dates. Tightened fuzzy-match threshold (Levenshtein ≤ 1 within same asset class). |
| 11 | `retrieval/vectorstore.py` | `_load_config()` removed; uses `agent.cache.get_config()`. `_batch()` helper inlined. |
| 12 | `agent/guardrails.py` | `BAD_PATTERNS` loaded from `config.yaml` (configurable, not hardcoded). |
| 13 | `agent/faithfulness.py` | Added note in semantic judge prompt about truncated source data. |
| 14 | `evaluation/eval.py` | Fixed degradation detection bug (was always True on non-empty answer). |
| 15 | `notebooks/` | **Added `01_data_exploration.ipynb`** (Task 1.3 — was missing). |
| 16 | `.env.example` | **Added** (was missing — required by submission instructions). |
| 17 | `.gitignore` | **Added** — excludes `.env`, `.chroma_db/`, `__pycache__/`, venvs. |
| 18 | `requirements.txt` | Added `tenacity>=8.2.0` and `matplotlib>=3.8.0`. |
| 19 | `config.yaml` | Added `guardrails.bad_date_patterns` list. |

---

## Trade-offs and what I'd do with more time

- **Streaming**: The CLI outputs the full answer after agent completion. True
  token-level streaming would require refactoring the synthesiser to use
  `ChatOpenAI.stream()` and yield through LangGraph's streaming API.
- **Fault description in telemetry**: The `fault_description` column is free text
  (~12 unique values). A hybrid approach — pandas filter on fault_code + RAG lookup
  for the description — would handle "find all E-1001 events and explain each one"
  in a single turn.
- **Evaluation**: Keyword recall is a proxy. Production would use LLM-as-judge scoring
  with a larger ground-truth set.
- **Date inference**: `_infer_date_range()` uses keyword matching. A production system
  would parse dates properly with `dateparser` or a dedicated extraction step.
- **Thread safety**: `validate._last_checks` is a function attribute (not thread-safe).
  For a multi-user web server, move this to a per-request context variable.
