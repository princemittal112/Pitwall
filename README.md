# PitWall — Autonomous Race Strategy & Telemetry Multi-Agent System

PitWall is a local-first, multi-agent system that simulates a Formula 1 pit wall. It ingests a race situation — for example *"Safety car deployed on lap 40 of 57, running on 20-lap-old hard tyres"* — and runs a cyclic, multi-agent debate loop to produce a definitive, **legally compliant**, and **telemetry-grounded** pit stop strategy.

Two specialist agents argue it out: a Tyre Performance Engineer proposes a strategy from real race telemetry, and a Sporting Director checks it against the FIA Sporting Regulations. If the proposal is illegal, the system loops back with concrete constraints until a compliant strategy emerges or a hard iteration cap is reached.

The entire stack runs **locally with zero ongoing cost and no API dependencies** — quantized LLM inference, local embeddings, and a free telemetry source.

---

## What's built

The full multi-agent pipeline is complete and verified end to end: from raw race telemetry and FIA regulation retrieval, through both agents, to a cyclic LangGraph state machine that produces a final strategy plus an agent-by-agent debate log. Constraint steering and the loop cap are both verified, and Node A is conditioned on the natural-language race situation.

---

## Design principles

The architecture is the point. Three principles run through the whole system:

1. **Deterministic logic stays in Python; the LLM only reasons.** Numerical telemetry calculations and the two-compound legality check are computed in pandas and set arithmetic — never delegated to the model, which would hallucinate them.
2. **Hybrid retrieval over structural lookups.** FIA article numbers shift between regulation versions, so the system retrieves regulations semantically (dense vectors) *and* by exact keyword (sparse BM25), rather than relying on brittle article-number matching.
3. **Cyclic agent loops need explicit safeguards.** The graph has a dual exit condition and a hard recursion backstop, so it is guaranteed to terminate even if the two agents never agree.

---

## Architecture

```
                          Race situation input
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      LangGraph StateGraph      │
                    │                                │
   ┌────────────────┴──────┐            ┌────────────┴───────────┐
   │  Node A                │            │  Node B                 │
   │  Tyre Perf. Engineer   │  proposal  │  Sporting Director      │
   │  - reads telemetry     ├───────────▶│  - queries hybrid RAG   │
   │  - proposes strategy   │            │  - checks legality      │
   └────────────────────────┘            └────────────┬───────────┘
            ▲                                          │
            │  active_constraints                      ▼
            │  (deterministic)                 Conditional edge
            │                              is_legal? loop_count?
            └──────────── loop back ◀──────────────┤
                                                    │ legal OR cap hit
                                                    ▼
                                          Final strategy output
```

The two agents communicate exclusively through a shared `TypedDict` state object. Node A writes a structured proposal; Node B writes a legal ruling and, on failure, a deterministically-computed constraint that steers Node A's next attempt.

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Orchestration | LangGraph (`StateGraph`) | Cyclic multi-agent graph with conditional edges |
| Local inference | llama.cpp via `llama-cpp-python` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| Telemetry | FastF1 | Real historical lap, stint, and tyre data |
| RAG (dense) | FAISS + `all-MiniLM-L6-v2` | Local embeddings, semantic retrieval |
| RAG (sparse) | BM25 (`rank_bm25`) | Exact keyword / article matching |
| Hybrid retrieval | LangChain `EnsembleRetriever` | Weighted 0.6 BM25 / 0.4 dense |
| Structured output | Pydantic | Enforced agent output schemas |

Everything runs on CPU. No GPU, no API keys, no cloud services.

---

## The agents

### Node A — Tyre Performance Engineer
Receives the race situation and structured telemetry (per-stint compound, lap count, average pace, degradation). Reasons over tyre wear and laps remaining to propose a pit lap and compound. Returns a Pydantic `StrategyProposal`. When the Sporting Director rejects a prior proposal, Node A receives explicit constraints injected into its prompt and revises accordingly.

### Node B — Sporting Director
Receives Node A's proposal, retrieves relevant FIA regulation clauses via the hybrid retriever, and rules on legality. Returns a Pydantic `LegalRuling` with an `is_legal` flag, violations, and citations.

The mandatory two-compound rule is **pre-computed in Python** (`len(set(compounds_used) | {proposed}) >= 2`) and handed to the model as ground truth — the LLM reports and explains the ruling rather than performing the boolean logic itself.

---

## The state machine

The shared state is a single `TypedDict` flowing through every node, grouped into three categories:

- **Race setup** (immutable): `year`, `round_number`, `driver`, `current_lap`, `total_laps`, `race_situation`, `compounds_used`
- **Working memory** (read + written each turn): `current_proposal`, `latest_ruling`, `regulation_citations`, `active_constraints`, `debate_history`
- **Loop control** (read by the conditional edge): `is_legal`, `loop_count`, `max_loops`

`debate_history` uses an `Annotated[List[str], operator.add]` reducer so each turn appends to the log rather than overwriting it. Every other field uses default overwrite semantics.

### Convergence safeguards
- **Dual exit:** the conditional edge exits on `is_legal == True` (success) **or** `loop_count >= max_loops` (safety).
- **Constraint steering:** on an illegal ruling, the next constraint is computed deterministically (`ALL_DRY_COMPOUNDS - compounds_used`), not taken from the model's free-text suggestion — preventing nonsensical loop-backs.
- **Recursion backstop:** `recursion_limit=25` in the LangGraph config is a second, lower-level guard independent of the application loop counter.

---

## Project structure

```
PitWall/
├── data/                   # FIA regulations PDF lives here
├── cache/                  # cached FastF1 sessions + persisted FAISS index
├── graph.py                # StateGraph assembly + run harness
├── state.py                # PitWallState TypedDict
├── node_a.py               # Tyre Performance Engineer + StrategyProposal schema
├── node_b.py               # Sporting Director + LegalRuling schema
├── telemetry_tool.py       # FastF1 pipeline (deterministic pandas)
├── rag_pipeline.py         # Hybrid BM25 + FAISS retriever
├── test_llm.py             # Standalone llama.cpp inference check
├── utils.py                # Robust JSON extraction (handles truncation)
├── requirements.txt        # pinned dependencies
├── .env                    # local paths / config (gitignored)
└── .gitignore
```

---

## Setup

### 1. Dependencies

`llama-cpp-python` is compiled separately for a CPU build with an OpenBLAS speedup:

```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

Then install the rest:

```bash
pip install -r requirements.txt
```

### 2. Model
Download `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` from
[bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
and update `MODEL_PATH` in `graph.py`.

### 3. Regulations
Download the 2024 FIA Formula 1 Sporting Regulations PDF from the
[FIA regulations page](https://www.fia.com/regulation/category/110) and save it as
`data/fia_regulations.pdf`.

---

## Usage

Run the full graph against a scenario:

```bash
python graph.py
```

Or programmatically:

```python
from graph import PitWallGraph

pitwall = PitWallGraph(llm=llm, retriever=retriever)

result = pitwall.run(
    year=2023,
    round_number=1,            # Bahrain GP
    driver="VER",
    current_lap=40,
    total_laps=57,
    race_situation="Safety car deployed on lap 40 after a Turn 4 crash. Pit lane open.",
    compounds_used=["SOFT"],
)

print(result["current_proposal"])   # final StrategyProposal
print(result["debate_history"])     # full agent-by-agent log
```

The first telemetry and RAG calls build local caches (~1–2 minutes total). Subsequent runs load from cache instantly.

---

## Performance notes

This runs an 8B model quantized to Q4_K_M on CPU. Expect roughly **20–40 seconds per agent call** (8–15 tokens/sec). A single-loop resolution takes ~1 minute; a multi-loop debate takes proportionally longer. This is by design — the system prioritizes data privacy and zero cost over latency. CPU inference latency is treated as a first-class design constraint, not an afterthought.

---

## Design decisions worth defending

- **Why LangGraph over a simple chain?** The core interaction is *cyclic* — propose, evaluate, revise — which a linear chain cannot express. LangGraph's stateful conditional edges model this natively with built-in termination guarantees.
- **Why local GGUF inference?** Data privacy, zero ongoing cost, and full offline operation. The trade-off is latency, which is accounted for in the UX.
- **Why hybrid RAG?** Regulations contain both semantic concepts ("mandatory compound rule") and exact tokens ("Article 30.2"). Dense search handles the former, BM25 the latter; neither alone is sufficient.
- **Why pre-compute legality in Python?** LLMs are unreliable at counting and boolean logic. The model reasons *about* the rule; it never *evaluates* it.
