from rag_pipeline import build_ensemble_retriever, get_or_build_faiss_store, load_and_chunk_regulations
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from rag_pipeline import build_ensemble_retriever, load_faiss_store, load_and_chunk_regulations
from node_a import StrategyProposal
import json
from dotenv import load_dotenv
import os
from utils import extract_first_json
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

# ── 1. Output schema ─────────────────────────────────────────────────────────

class LegalRuling(BaseModel):
    """Structured legal ruling output from Node B."""
    is_legal: bool = Field(description="Whether the proposed strategy is legally compliant")
    violations: List[str] = Field(description="List of regulation violations found. Empty list if legal.")
    citations: List[str] = Field(description="Relevant regulation articles or clauses retrieved")
    constraints_for_node_a: str = Field(
        description="If illegal, specific constraints Node A must respect in the next proposal. Empty string if legal."
    )
    ruling_rationale: str = Field(description="One concise sentence. Max 25 words.")


# ── 2. Node B agent ──────────────────────────────────────────────────────────

class SportingDirector:
    def __init__(self, llm: LlamaCpp, retriever):
        self.llm = llm
        self.retriever = retriever
        self.parser = PydanticOutputParser(pydantic_object=LegalRuling)

        self.prompt = PromptTemplate(
            template="""<|start_header_id|>system<|end_header_id|>

        You are an FIA Sporting Director. Output exactly one valid JSON object and nothing else. No prose before or after. No repetition. No markdown fences.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Evaluate this pit strategy for regulatory compliance.

        PROPOSED STRATEGY:
        - Pit Lap: {pit_lap}
        - Proposed Compound: {proposed_compound}
        - Laps on New Tyre: {laps_remaining_on_new_tyre}
        - Engineer Rationale: {rationale}

        TWO-COMPOUND RULE CHECK (pre-computed, treat as ground truth):
        - Compounds used so far: {compounds_used}
        - Compound proposed now: {proposed_compound}
        - Total distinct compounds after this pit: {distinct_compound_count}
        - TWO-COMPOUND RULE MET: {two_compound_met}

        RETRIEVED REGULATIONS:
        {retrieved_regulations}

        Output one JSON object matching this schema:
        {format_instructions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """,
            input_variables=[
                "pit_lap",
                "proposed_compound",
                "laps_remaining_on_new_tyre",
                "rationale",
                "compounds_used",
                "distinct_compound_count",
                "two_compound_met",
                "retrieved_regulations",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def _retrieve_regulations(self, proposal: StrategyProposal, compounds_used: List[str]) -> str:
        """
        Builds semantic queries from the proposal context and retrieves
        relevant regulation chunks. Semantic queries only — no article numbers.
        """
        queries = [
            "mandatory tyre compound requirement during race",
            f"use of {proposal.proposed_compound.lower()} tyre compound regulations",
            "pit stop tyre change rules dry weather",
        ]

        all_chunks = []
        seen = set()

        for query in queries:
            docs = self.retriever.invoke(query)
            for doc in docs:
                # Deduplicate by content
                content = doc.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    all_chunks.append(content)

        # Cap at 6 chunks to stay within context window
        return "\n\n---\n\n".join(all_chunks[:6])

    def evaluate_strategy(
        self,
        proposal: StrategyProposal,
        compounds_used: list[str],
    ) -> LegalRuling:

        # ── Pre-compute two-compound rule in Python, not by the LLM ─────────
        all_compounds = set(compounds_used) | {proposal.proposed_compound}
        distinct_compound_count = len(all_compounds)
        two_compound_met = distinct_compound_count >= 2

        print(f"\n[Node B] Querying regulations for {proposal.proposed_compound} proposal...")
        retrieved_regs = self._retrieve_regulations(proposal, compounds_used)

        formatted_prompt = self.prompt.format(
            pit_lap=proposal.pit_lap,
            proposed_compound=proposal.proposed_compound,
            laps_remaining_on_new_tyre=proposal.laps_remaining_on_new_tyre,
            rationale=proposal.rationale,
            compounds_used=", ".join(compounds_used) if compounds_used else "None",
            distinct_compounds=", ".join(all_compounds),
            distinct_compound_count=distinct_compound_count,
            two_compound_met="YES — rule satisfied" if two_compound_met else "NO — rule violated",
            retrieved_regulations=retrieved_regs,
        )

        print(f"[Node B] Evaluating legal compliance...")
        raw_output = self.llm.invoke(formatted_prompt)
        print(f"[Node B] Raw output:\n{raw_output}\n")

        # Robustly extract the FIRST complete JSON object
        clean_json = extract_first_json(raw_output)
        ruling = self.parser.parse(clean_json)
        return ruling
# ── 3. Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_community.llms import LlamaCpp
    from node_a import StrategyProposal

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=8,
        temperature=0.1,
        max_tokens=512,
        stop=["<|eot_id|>", "Here is", "The final answer", "\n\n\n"],  # halt after first object
        verbose=False,
    )

    # Load RAG pipeline
    print("Loading RAG pipeline...")
    chunks = load_and_chunk_regulations("data/fia_regulations.pdf")
    faiss_store = get_or_build_faiss_store(chunks)   # builds if cache missing
    retriever = build_ensemble_retriever(chunks, faiss_store)

    director = SportingDirector(llm=llm, retriever=retriever)

    # ── Test A: legal strategy ───────────────────────────────────────────────
    # VER started on Softs, now proposing Mediums — two compounds used, legal
    legal_proposal = StrategyProposal(
        pit_lap=45,
        proposed_compound="MEDIUM",
        laps_remaining_on_new_tyre=12,
        rationale="Degradation trend favours a medium compound for the final stint.",
        confidence="HIGH",
    )

    print("\n" + "=" * 60)
    print("TEST A: Legal strategy (Soft → Medium, two compounds used)")
    print("=" * 60)
    ruling_a = director.evaluate_strategy(
        proposal=legal_proposal,
        compounds_used=["SOFT"],   # already used Softs, now Mediums = two compounds
    )
    print(f"  is_legal:    {ruling_a.is_legal}")
    print(f"  violations:  {ruling_a.violations}")
    print(f"  citations:   {ruling_a.citations}")
    print(f"  rationale:   {ruling_a.ruling_rationale}")

    # ── Test B: illegal strategy ─────────────────────────────────────────────
    # Driver proposing to use only one compound for the entire race
    illegal_proposal = StrategyProposal(
        pit_lap=45,
        proposed_compound="HARD",
        laps_remaining_on_new_tyre=12,
        rationale="Hard tyres will last to the end without degradation issues.",
        confidence="HIGH",
    )

    print("\n" + "=" * 60)
    print("TEST B: Illegal strategy (Hard → Hard, only one compound)")
    print("=" * 60)
    ruling_b = director.evaluate_strategy(
        proposal=illegal_proposal,
        compounds_used=["HARD"],   # only used Hards — violates two-compound rule
    )
    print(f"  is_legal:    {ruling_b.is_legal}")
    print(f"  violations:  {ruling_b.violations}")
    print(f"  citations:   {ruling_b.citations}")
    print(f"  constraints: {ruling_b.constraints_for_node_a}")
    print(f"  rationale:   {ruling_b.ruling_rationale}")