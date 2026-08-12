from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from telemetry_tool import get_race_context
from typing import Optional
from dotenv import load_dotenv
import os
from utils import extract_first_json

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

# ── 1. Output schema ─────────────────────────────────────────────────────────

class StrategyProposal(BaseModel):
    """Structured strategy proposal output from Node A."""
    pit_lap: int = Field(description="The lap number on which to pit")
    proposed_compound: str = Field(description="Tyre compound to fit: SOFT, MEDIUM, or HARD")
    laps_remaining_on_new_tyre: int = Field(description="Laps the new tyre must last to the end of the race")
    rationale: str = Field(description="One concise sentence. Max 25 words.")
    confidence: str = Field(description="Confidence level: HIGH, MEDIUM, or LOW")


# ── 2. Node A agent ──────────────────────────────────────────────────────────

class TyrePerformanceEngineer:
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=StrategyProposal)

        self.prompt = PromptTemplate(
            template="""<|start_header_id|>system<|end_header_id|>

        You are an F1 Tyre Performance Engineer. Output exactly one valid JSON object and nothing else. No prose before or after.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Propose an optimal pit stop strategy for the situation below.

        CURRENT RACE SITUATION:
        {race_situation}

        RACE TELEMETRY:
        - Driver: {driver}
        - Current Lap: {current_lap} of {total_laps}
        - Current Compound: {current_compound}
        - Laps on Current Tyre: {laps_on_current_tyre}
        - Current Stint Number: {current_stint_number}

        STINT HISTORY:
        {stint_summary}

        REGULATORY CONSTRAINTS (must obey if present):
        {regulatory_constraints}

        Factor the race situation into your reasoning — a safety car, weather change, or
        damage materially changes the optimal call. Output one JSON object matching this schema:
        {format_instructions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """,
            input_variables=[
                "race_situation",
                "driver",
                "current_lap",
                "total_laps",
                "current_compound",
                "laps_on_current_tyre",
                "current_stint_number",
                "stint_summary",
                "regulatory_constraints",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def _format_stint_summary(self, stints: list) -> str:
        """Formats stint summary list into a readable string for the prompt."""
        lines = []
        for s in stints:
            lines.append(
                f"  Stint {s['Stint']} | {s['Compound']:<8} | "
                f"{s['lap_count']} laps | "
                f"avg {s['avg_lap_time_s']:.3f}s | "
                f"degradation {s['degradation_s']:+.3f}s"
            )
        return "\n".join(lines)

    def propose_strategy(
        self,
        year: int,
        round_number: int,
        driver: str,
        current_lap: int,
        total_laps: int,
        race_situation: str = "Normal racing conditions.",   # new param
        regulatory_constraints: Optional[str] = None,
    ) -> StrategyProposal:

        context = get_race_context(year, round_number, driver, current_lap)

        if "error" in context:
            raise ValueError(f"Telemetry error: {context['error']}")

        constraints_text = regulatory_constraints or "None — propose optimal strategy freely."

        formatted_prompt = self.prompt.format(
            race_situation=race_situation,           # new
            driver=context["driver"],
            current_lap=context["current_lap"],
            total_laps=total_laps,
            current_compound=context["current_compound"],
            laps_on_current_tyre=context["laps_on_current_tyre"],
            current_stint_number=context["current_stint_number"],
            stint_summary=self._format_stint_summary(context["stint_summary"]),
            regulatory_constraints=constraints_text,
        )

        print(f"\n[Node A] Analysing telemetry for {driver} at lap {current_lap}...")
        raw_output = self.llm.invoke(formatted_prompt)
        print(f"[Node A] Raw output:\n{raw_output}\n")

        clean_json = extract_first_json(raw_output)
        proposal = self.parser.parse(clean_json)
        return proposal


# ── 3. Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_community.llms import LlamaCpp

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        max_tokens=512,
        verbose=False,
    )

    engineer = TyrePerformanceEngineer(llm=llm)

    proposal = engineer.propose_strategy(
        year=2023,
        round_number=1,       # Bahrain GP
        driver="VER",
        current_lap=40,
        total_laps=57,
        regulatory_constraints=None,   # First loop — no constraints yet
    )

    print("=" * 60)
    print("STRATEGY PROPOSAL")
    print("=" * 60)
    print(f"  Pit lap:          {proposal.pit_lap}")
    print(f"  Compound:         {proposal.proposed_compound}")
    print(f"  Laps on new tyre: {proposal.laps_remaining_on_new_tyre}")
    print(f"  Confidence:       {proposal.confidence}")
    print(f"  Rationale:        {proposal.rationale}")