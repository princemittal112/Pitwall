from typing import TypedDict, List, Optional, Annotated
import operator
from node_a import StrategyProposal
from node_b import LegalRuling


class PitWallState(TypedDict):
    # ── Race setup (immutable — set once at graph entry, never mutated) ──────
    year: int
    round_number: int
    driver: str
    current_lap: int
    total_laps: int
    race_situation: str              # the natural-language event, e.g. "Safety car deployed lap 40"
    compounds_used: List[str]        # compounds already run this race

    # ── Working memory (read + written as the agents debate) ─────────────────
    current_proposal: Optional[StrategyProposal]   # Node A writes this
    latest_ruling: Optional[LegalRuling]           # Node B writes this
    regulation_citations: List[str]                # accumulated citations
    active_constraints: str                        # injected back into Node A on a failed loop
    debate_history: Annotated[List[str], operator.add]   # append-only log of each turn

    # ── Loop control (read by the conditional edge) ──────────────────────────
    is_legal: bool
    loop_count: int
    max_loops: int