from langgraph.graph import StateGraph, END
from langchain_community.llms import LlamaCpp
from state import PitWallState
from node_a import TyrePerformanceEngineer
from node_b import SportingDirector
from rag_pipeline import (
    build_ensemble_retriever,
    get_or_build_faiss_store,
    load_and_chunk_regulations,
)
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
ALL_DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


class PitWallGraph:
    """Builds and compiles the LangGraph state machine wiring Node A and Node B."""

    def __init__(self, llm: LlamaCpp, retriever):
        self.engineer = TyrePerformanceEngineer(llm=llm)
        self.director = SportingDirector(llm=llm, retriever=retriever)
        self.app = self._build_graph()

    # ── Node A wrapper ───────────────────────────────────────────────────────
    def _node_a(self, state: PitWallState) -> dict:
        """Tyre Performance Engineer proposes a strategy."""
        print(f"\n{'='*60}\n[GRAPH] Node A — loop {state['loop_count'] + 1}\n{'='*60}")

        proposal = self.engineer.propose_strategy(
            year=state["year"],
            round_number=state["round_number"],
            driver=state["driver"],
            current_lap=state["current_lap"],
            total_laps=state["total_laps"],
            race_situation=state["race_situation"],          # new
            regulatory_constraints=state["active_constraints"] or None,
        )

        return {
            "current_proposal": proposal,
            "debate_history": [
                f"[Node A | loop {state['loop_count'] + 1}] "
                f"Proposed: pit lap {proposal.pit_lap}, {proposal.proposed_compound}. "
                f"Rationale: {proposal.rationale}"
            ],
        }

    # ── Node B wrapper ───────────────────────────────────────────────────────
    def _node_b(self, state: PitWallState) -> dict:
        """Sporting Director evaluates legality."""
        print(f"\n{'='*60}\n[GRAPH] Node B — evaluating proposal\n{'='*60}")

        proposal = state["current_proposal"]
        ruling = self.director.evaluate_strategy(
            proposal=proposal,
            compounds_used=state["compounds_used"],
        )

        # ── Deterministic constraint steering ───────────────────────────────
        # If illegal due to two-compound rule, override the model's free-text
        # suggestion with a sensible, concrete dry-compound constraint.
        active_constraints = ""
        if not ruling.is_legal:
            used = set(state["compounds_used"]) | {proposal.proposed_compound}
            legal_alternatives = ALL_DRY_COMPOUNDS - set(state["compounds_used"])
            if legal_alternatives:
                alts = ", ".join(sorted(legal_alternatives))
                active_constraints = (
                    f"Your previous proposal of {proposal.proposed_compound} was ruled ILLEGAL. "
                    f"Violations: {'; '.join(ruling.violations)}. "
                    f"You MUST propose a different dry compound from this set to satisfy the "
                    f"two-compound rule: {alts}. Do NOT propose {proposal.proposed_compound} again."
                )
            else:
                active_constraints = (
                    f"Your previous proposal was ruled ILLEGAL. "
                    f"Violations: {'; '.join(ruling.violations)}. "
                    f"Revise the strategy to comply."
                )

        return {
            "latest_ruling": ruling,
            "is_legal": ruling.is_legal,
            "regulation_citations": ruling.citations,
            "active_constraints": active_constraints,
            "loop_count": state["loop_count"] + 1,
            "debate_history": [
                f"[Node B | loop {state['loop_count'] + 1}] "
                f"is_legal={ruling.is_legal}. "
                f"{'Violations: ' + '; '.join(ruling.violations) if ruling.violations else 'Compliant.'}"
            ],
        }

    # ── Conditional edge ─────────────────────────────────────────────────────
    def _should_continue(self, state: PitWallState) -> str:
        """
        Routing logic after Node B.
        Exits to END if legal OR loop cap hit. Otherwise loops back to Node A.
        """
        if state["is_legal"]:
            print(f"\n[GRAPH] ✅ Strategy is LEGAL — exiting to END")
            return "end"

        if state["loop_count"] >= state["max_loops"]:
            print(f"\n[GRAPH] ⛔ Loop cap ({state['max_loops']}) reached — exiting to END")
            return "end"

        print(f"\n[GRAPH] 🔄 Illegal — looping back to Node A "
              f"(loop {state['loop_count']}/{state['max_loops']})")
        return "continue"

    # ── Graph assembly ───────────────────────────────────────────────────────
    def _build_graph(self):
        workflow = StateGraph(PitWallState)

        workflow.add_node("tyre_engineer", self._node_a)
        workflow.add_node("sporting_director", self._node_b)

        workflow.set_entry_point("tyre_engineer")
        workflow.add_edge("tyre_engineer", "sporting_director")

        workflow.add_conditional_edges(
            "sporting_director",
            self._should_continue,
            {
                "continue": "tyre_engineer",   # loop back
                "end": END,
            },
        )

        # recursion_limit is a hard backstop below our own loop cap
        return workflow.compile()

    # ── Public run method ────────────────────────────────────────────────────
    def run(
        self,
        year: int,
        round_number: int,
        driver: str,
        current_lap: int,
        total_laps: int,
        race_situation: str,
        compounds_used: list,
    ) -> PitWallState:
        initial_state: PitWallState = {
            "year": year,
            "round_number": round_number,
            "driver": driver,
            "current_lap": current_lap,
            "total_laps": total_laps,
            "race_situation": race_situation,
            "compounds_used": compounds_used,
            "current_proposal": None,
            "latest_ruling": None,
            "regulation_citations": [],
            "active_constraints": "",
            "debate_history": [],
            "is_legal": False,
            "loop_count": 0,
            "max_loops": 5,
        }

        # recursion_limit must exceed max_loops * 2 (two nodes per loop) + buffer
        final_state = self.app.invoke(
            initial_state,
            config={"recursion_limit": 25},
        )
        return final_state


# ── Test harness ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Initialising LLM...")
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=8,
        temperature=0.1,
        max_tokens=1024,
        stop=["<|eot_id|>"],
        verbose=False,
    )

    print("Loading RAG pipeline...")
    chunks = load_and_chunk_regulations("data/fia_regulations.pdf")
    faiss_store = get_or_build_faiss_store(chunks)
    retriever = build_ensemble_retriever(chunks, faiss_store)

    print("Building graph...")
    pitwall = PitWallGraph(llm=llm, retriever=retriever)

# ── Test harness ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialising LLM...")
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=8,
        temperature=0.1,
        max_tokens=1024,
        stop=["<|eot_id|>"],
        verbose=False,
    )

    print("Loading RAG pipeline...")
    chunks = load_and_chunk_regulations("data/fia_regulations.pdf")
    faiss_store = get_or_build_faiss_store(chunks)
    retriever = build_ensemble_retriever(chunks, faiss_store)

    print("Building graph...")
    pitwall = PitWallGraph(llm=llm, retriever=retriever)
    
    print("\n\n" + "#"*60)
    print("# SCENARIO WITH RACE SITUTAION")
    print("#"*60)

    result2 = pitwall.run(
        year=2023,
        round_number=1,
        driver="VER",
        current_lap=40,
        total_laps=57,
        race_situation=(
            "SAFETY CAR deployed on lap 40 of 57 after a crash at Turn 4. "
            "Pit lane is open. This is a cheap-stop opportunity — track position "
            "loss from pitting is minimal under safety car."
        ),
        compounds_used=["SOFT"],
    )

    print("\n\n" + "="*60)
    print("SCENARIO 2 RESULT")
    print("="*60)
    print(f"Loops run:        {result2['loop_count']}")
    print(f"Final is_legal:   {result2['is_legal']}")
    if result2["current_proposal"]:
        p = result2["current_proposal"]
        print(f"Final strategy:   Pit lap {p.pit_lap}, {p.proposed_compound}")
    print(f"\nDebate history:")
    for entry in result2["debate_history"]:
        print(f"  {entry}")