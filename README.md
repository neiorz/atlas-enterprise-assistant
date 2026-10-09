# Atlas Industries — Enterprise Assistant

**Agentic internal knowledge chatbot** — graduation project for the Sprints AI/ML program.

Employees ask HR, IT, or Finance questions in English or Arabic. A LangGraph workflow
routes the question, retrieves cited context from the official document corpus,
optionally calls typed tools, and answers with sources — or clearly refuses when the
answer isn't in the corpus.

**Stack:** Python · LangGraph · Google Gemini (free tier) · Qdrant · `BAAI/bge-m3`
embeddings · Chainlit · DeepEval — **total cost $0**.

---

## Quick Start (5 commands)

```bash
git clone <this-repo> && cd atlas-enterprise-assistant
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env          # then paste your GOOGLE_API_KEY into .env
python -m src.ingest && python -m src.vectorstore   # build the index (~1 min)
chainlit run app.py           # chat UI at http://localhost:8000
```

Evaluations and logs:

```bash
python tests/evaluate.py      # DeepEval → outputs/eval_report.json
bash scripts/reset.sh         # wipe + rebuild the vector index (FR-B8, ~70s)
grep <run_id> outputs/run_logs.jsonl   # full timeline of one question
```

### Offline tests (no API key needed)

```bash
pytest tests/ -q          # 25 tests, ~13s
```

- `tests/test_graph.py` stubs the three Gemini handles and drives the real
  graph end-to-end: routing + low-confidence fallback, cross-turn memory,
  tool loop, Pydantic validation recovery, the step-limit fallback node, and
  run-log completeness.
- `tests/test_evaluate.py` stubs the graph and judge to verify the DeepEval
  script itself: gold-case loading (FR-J1), gold-context assembly (FR-J2),
  routing accuracy (FR-J4), the full `eval_report.json` shape (FR-J6), both
  CLI modes, and the friendly no-key error path.

Use them to confirm a refactor didn't break the wiring before spending quota.

---

## How It Works

```
employee question
      │
      ▼
┌────────────────────────────────────────────────────────────┐
│  LangGraph (src/graph.py)                                  │
│                                                            │
│  memory_loader ─► router ─► retriever ─► agent ◄─► tools   │
│                                     │            (max 3)   │
│                                     ▼                      │
│                     generate_answer / fallback (max 10)    │
│                                     │                      │
│                                     ▼                      │
│                              memory_writer ─► END          │
└────────────────────────────────────────────────────────────┘
      │                              │
      ▼                              ▼
outputs/run_logs.jsonl        session memory (MemorySaver,
 (one JSON event per step,     keyed by session_id = thread_id)
  all tagged with run_id)
```

| Node | Responsibility |
|---|---|
| `memory_loader` | Reset per-turn state; prior turns stay in `chat_history` via the checkpointer |
| `router` | Gemini structured output → `hr` / `it` / `finance` + confidence. Below 0.55 confidence → falls back to `"multi"` and fans retrieval out across all domains |
| `retriever` | Domain-filtered top-k search (k from `settings.retrieval_top_k`) |
| `agent` | Decides whether a tool call helps; bound by `max_tool_calls = 3` |
| `tools` | Executes typed tools; Pydantic validation errors are fed back to the agent for **one** retry |
| `generate_answer` | Composes a short answer in the user's language, ending with a real `Sources:` list |
| `fallback` | Step-limit escape hatch (max 10 steps) — answers with explicit uncertainty |
| `memory_writer` | Appends the Q/A pair to session memory and closes the run log |

## Stack Choices (and why)

- **LLM — Google Gemini (`gemini-2.0-flash`).** Free tier with generous limits, native
  Arabic comprehension (3/10 eval cases are Arabic, and the judge must handle them),
  and structured output for the router's typed `RouterDecision`.
- **Embeddings — `BAAI/bge-m3` (1024-dim).** Multilingual by design: English and Arabic
  land in one semantic space, so an Arabic question retrieves Arabic documents without
  a second index. Runs locally on CPU.
- **Vector store — Qdrant, local on-disk mode.** Zero-cost, no server to run, keyword
  payload indexes on `domain` / `language` / `source_filename` give free metadata
  filtering for domain-scoped retrieval. A hosted URL is a one-line `.env` switch.
- **Chat UI — Chainlit.** Native step rendering makes the workflow visible (domain,
  retrieved sources, tool calls) per FR-H2, and it handles RTL text correctly.
- **Loaders — `pypdf` for PDFs.** Deliberately *not* pdfplumber/pdfminer: both reverse
  every line of the Arabic PDFs (classic RTL extraction bug). Verified by hand on all
  9 PDFs before locking this in. `python-docx` for Word, plain UTF-8 for Markdown.
- **Chunking — paragraph → sentence → character fallback.** Hand-rolled and inspectable;
  800-char chunks with 150-char overlap. Documented per FR-B4 — on a 30-document corpus,
  retrieval ranking is sensitive to this, so it lives in `settings.py`.
- **Memory — LangGraph `MemorySaver`.** Thread ID == session ID, so a new chat is a new
  thread with empty memory. No external DB needed for in-session memory.
- **Eval — DeepEval with a Gemini judge.** Same single free key as the answer path;
  the judge reads Arabic so the three Arabic cases are scored fairly.

## Results

_Fill in after `python tests/evaluate.py` — paste the summary table from
`outputs/eval_report.json` here._

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | _tbd_ | |
| Answer Relevancy | _tbd_ | |
| Contextual Precision | _tbd_ | |
| Contextual Recall | _tbd_ | |
| Hallucination | _tbd_ | |
| Routing accuracy | _tbd_ | |

## File Tree

```
atlas-enterprise-assistant/
├── app.py                     ← Chainlit UI entry point
├── src/
│   ├── settings.py            ← all tunable values (paths, models, thresholds)
│   ├── prompts.py             ← router / tool / answer prompts
│   ├── prompt_context.py      ← renders state → prompt blocks (history/chunks/tools)
│   ├── state.py               ← typed graph state + RouterDecision
│   ├── graph.py               ← LangGraph assembly + run_turn()
│   ├── graph_nodes.py         ← input side: memory_loader, router, retriever, writer
│   ├── agent_nodes.py         ← agentic side: agent ⇄ tools, answer, fallback
│   ├── llm_setup.py           ← Gemini client + tool binding
│   ├── judge.py               ← DeepEval judge wrapper (Arabic-capable)
│   ├── retriever.py           ← domain-filtered + multi-domain retrieval
│   ├── ingest.py              ← corpus loading, chunking, policy index
│   ├── loaders.py             ← .md / .pdf / .docx (Arabic-safe)
│   ├── chunking.py            ← language detection + splitter
│   ├── embeddings.py          ← bge-m3 wrapper
│   ├── vectorstore.py         ← Qdrant index build + search
│   ├── tools.py               ← 3 typed tools (Pydantic schemas + logic)
│   ├── corpus_facts.py        ← locked corpus numbers (caps, leave types)
│   └── logging_setup.py       ← run_id + structured JSONL events
├── tests/
│   ├── evaluate.py            ← DeepEval script → outputs/eval_report.json
│   ├── test_evaluate.py       ← offline tests for the eval script (FR-J)
│   ├── test_graph.py          ← offline wiring tests (stubbed LLM, no key)
│   └── eval_cases.jsonl       ← 10 gold questions
├── data/
│   ├── CORPUS_README.md       ← locked facts (read before changing prompts)
│   └── docs/{hr,it,finance}/  ← 30 documents (.md / .pdf / .docx)
├── notebooks/
│   ├── 01_eda.ipynb           ← corpus EDA (formats, languages, lengths)
│   ├── 02_retrieval.ipynb     ← retrieval sanity checks (incl. Arabic)
│   └── 03_eval_walkthrough.ipynb ← DeepEval results discussion
├── outputs/                   ← run_logs.jsonl, eval_report.json, final_report.md
├── docs/                      ← architecture.png, demo.gif
└── scripts/
    ├── reset.sh               ← wipe index + rebuild (FR-B8)
    ├── smoke_demo.py          ← the brief's 3-turn demo scenario, checked
    └── mem_check.py           ← NFR-B4 memory measurement
```

**Mapping to the brief's suggested tree** (it says "adapt it to match the
stack you pick"): `router.py` lives in `graph_nodes.py` (`router_node`), and
`memory.py` is the `MemorySaver` checkpointer wired in `graph.py` — memory
was inlined rather than given its own file because LangGraph's checkpointer
already owns the persistence, so a wrapper module would only re-export it.
Everything else matches one-for-one.

## Development

```bash
black .            # formatting (NFR-C4)
ruff check .       # linting (NFR-C6)
python -m src.ingest   # inspect chunk counts after corpus changes
```

## Known Limitations

- **Memory footprint exceeds NFR-B4's 1 GB target (measured, documented).**
  The brief suggests `BAAI/bge-m3` / `multilingual-e5-large` for the mixed
  EN+AR corpus — but bge-m3 alone is 568M params (2.27 GB fp32), so *every*
  model the brief names blows the 1 GB budget. Measured with
  `python scripts/mem_check.py fp32`: **total RSS ~2.0 GB**, of which only
  **689 MB is private (RssAnon)** and **1312 MB is file-backed mmap of the
  model weights** — clean file pages the OS reclaims under pressure. We chose
  to keep bge-m3 because retrieval quality is a graded rubric row (15%) and
  Arabic/cross-language recall is the project's stated difficulty, while
  memory footprint is not. Loading in bfloat16 was tested and is *worse*
  (224 MB more RSS) because conversion materializes every weight into
  private memory. Swapping to a ~118M-param model (e.g.
  `multilingual-e5-small`) would fit the budget at a real cost to Arabic
  retrieval quality.
- **Cloud-only Qdrant payload indexes.** In the recommended local mode,
  Qdrant ignores payload indexes (it filters by scan, instant at 122 points),
  so `create_payload_index` runs only when `QDRANT_URL` is set. If you move to
  a hosted Qdrant with a much larger corpus, this is already handled.
- The corpus is only 30 documents, so retrieval is sensitive to chunk size —
  revisit `CHUNK_SIZE` if Contextual Recall drops.
- Session memory is in-process: restarting the app clears conversations
  (cross-session persistence is an optional bonus, not implemented).
- The Gemini free tier allows ~15 requests/minute; a graph turn makes up to
  three LLM calls, so back-to-back turns can push a turn past NFR-B1's 8-second
  budget while the rate limiter back-pressures (measured query latency itself
  is ~1.6 s).

## License

MIT — see [LICENSE](LICENSE). Atlas Industries is a fictional company; the corpus
contains no real employer data.
