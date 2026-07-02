"""The supervisor: it ROUTES, and nothing else.

DESIGN CHOICE (documented in full in ARCHITECTURE.md):
We use a CUSTOM supervisor that routes with LangGraph's Command(goto=...), not
langgraph-supervisor's create_supervisor. Why:
  - The spec's preferred "workers-as-tools" pattern buys tighter context control,
    and we get that benefit a different way: each specialist is an encapsulated
    node that keeps its own tool traffic to itself and contributes only typed,
    sourced evidence (Claims/TechnicalSignals) to shared state. The supervisor's
    context stays lean — it sees compact status lines + `visited`, not raw output.
  - A custom router keeps the structured-evidence flow explicit and trivially
    testable/deterministic (essential for the offline eval), instead of hiding it
    inside a prebuilt abstraction.

The supervisor NEVER calls research tools. It only decides "who's next",
constrained by guardrails: all three research specialists must contribute before
the writer runs, and a hard turn cap guarantees termination even if the router
misbehaves. The LLM picks the ORDER among the still-pending specialists; the code
guarantees completeness and progress.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from .agents import _parse_json
from .config import RESPONSIBLE_AI_POLICY, Settings
from .llm import build_llm
from .stub_llm import ROUTE_MARKER

# The research specialists that must all contribute before the writer runs. V1
# keeps exactly two (company research + technical signals) -> writer.
# PHASE 2: extend this list with competitor & ESG/technical-risk agents — the
# routing loop and the structured-evidence flow generalise to N specialists
# unchanged. A CrewAI/AutoGen variant of this same team would also slot in behind it.
RESEARCH_SPECIALISTS = ["company_research_agent", "technical_signals_agent"]

# The destinations the supervisor can route to. Declared as a Literal so LangGraph
# can discover the graph's edges from the node's return annotation.
Destination = Literal["company_research_agent", "technical_signals_agent", "writer_agent"]


def _route_with_llm(settings: Settings, company: str, completed: list[str], options: list[str]) -> str:
    """Ask the supervisor LLM to pick the next specialist from `options`.

    The decision is constrained to `options` and validated by the caller, so even
    a wonky model can't break the flow — it can only influence the order. In stub
    mode the deterministic stub returns the first option, making routing testable.

    RESILIENCE: if the LLM errors (rate limit / quota / timeout / network), we do
    NOT crash the run — routing is not worth failing over. We fall back to the
    first pending specialist (deterministic order), so a transient LLM hiccup on
    the ROUTER just costs the model's ordering, not the whole brief.
    """
    system = SystemMessage(
        content=(
            RESPONSIBLE_AI_POLICY
            + "\nYou are the SUPERVISOR of a technical-intelligence team. You only route; "
            "you never call research tools yourself."
        )
    )
    human = HumanMessage(
        content=(
            f"{ROUTE_MARKER}\n"
            f"company: {company}\n"
            f"completed: [{', '.join(completed)}]\n"
            f"options: [{', '.join(options)}]\n"
            'Reply with ONLY JSON: {"next": "<one of options>", "reason": "<short>"}.'
        )
    )
    try:
        parsed = _parse_json(str(build_llm("supervisor", settings).invoke([system, human]).content))
        choice = (parsed or {}).get("next") if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001 - a routing LLM error must not kill the run
        choice = None
    # Validate: the model may only choose a legal option; otherwise (or on error)
    # fall back to the first pending specialist (deterministic, always progresses).
    return choice if choice in options else options[0]


def make_supervisor(settings: Settings):
    """Build the supervisor node, capturing settings (so add_node gets a 1-arg fn).

    Returns a function annotated with its possible Command destinations so
    LangGraph can wire the dynamic edges.
    """

    def supervisor(state) -> Command[Destination]:
        completed = state.get("visited", [])
        turns = state.get("turns", 0) + 1
        pending = [s for s in RESEARCH_SPECIALISTS if s not in completed]

        # Safety valve: a hard cap guarantees the loop terminates even if the
        # router never marks a specialist done. Force the writer to finish up.
        if turns > settings.max_supervisor_turns:
            nxt = "writer_agent"
        elif pending:
            # Real routing decision: the LLM orders the pending specialists.
            nxt = _route_with_llm(settings, state["company_name"], completed, pending)
        else:
            # All research done -> hand off to the writer (the terminal node).
            nxt = "writer_agent"

        return Command(goto=nxt, update={"turns": turns, "route": nxt})

    return supervisor
