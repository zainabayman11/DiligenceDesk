"""Assemble the technical-intelligence team into a compiled LangGraph app.

The shape (supervisor pattern):

                 ┌─────────────┐
        START ──►│ supervisor  │  routes to the next pending specialist...
                 └──────┬──────┘
            ┌───────────┼────────────────┬──────────────┐
            ▼           ▼                ▼              ▼
   company_research  technical_signals               writer ──► END
            └───────────┴────────────────┘ (each returns to the supervisor)

- The supervisor routes with Command(goto=...) (a DYNAMIC edge).
- Each research specialist runs, appends its sourced evidence to shared state,
  and returns to the supervisor via a STATIC edge.
- Once both have contributed, the supervisor routes to the writer, which
  assembles the brief and ends the run.

A MemorySaver checkpointer is attached so a thread_id persists state across calls
(the seam for multi-turn follow-ups). The specialists are bound to ONE toolbox
built from the chosen mode, so the agents never know whether they're on stub,
local, or MCP tools.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import agents
from .config import Settings, get_settings
from .state import DiligenceState
from .supervisor import make_supervisor
from .tools.provider import Toolbox, get_tools


def build_graph(
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: MemorySaver | None = None,
):
    """Build and compile the multi-agent graph.

    Args:
        settings: Config snapshot (providers, models, tool mode, turn cap).
            Defaults to the environment.
        toolbox: Pre-built toolbox (tests inject the stub toolbox). Defaults to
            get_tools(settings.tool_mode).
        checkpointer: State persistence backend. Defaults to in-process MemorySaver.

    Returns:
        A compiled LangGraph app you call with `.invoke(initial_state, config)`.
    """
    settings = settings or get_settings()
    toolbox = toolbox or get_tools(settings.tool_mode, settings)

    graph = StateGraph(DiligenceState)

    # The supervisor only routes (dynamic Command edges, discovered from its
    # return annotation). Built via a factory so the node is a clean 1-arg fn.
    graph.add_node("supervisor", make_supervisor(settings))

    # Specialists are thin wrappers that inject the shared toolbox + settings, so
    # each node matches LangGraph's expected (state) -> update signature while the
    # real logic in agents.py stays easy to unit-test in isolation.
    graph.add_node("company_research_agent", lambda s: agents.company_research_agent(s, toolbox, settings))
    graph.add_node("technical_signals_agent", lambda s: agents.technical_signals_agent(s, toolbox, settings))
    graph.add_node("writer_agent", lambda s: agents.writer_agent(s, settings))

    graph.add_edge(START, "supervisor")
    # Each specialist returns control to the supervisor (static edges). The
    # supervisor's Command(goto) handles the dynamic outbound routing.
    graph.add_edge("company_research_agent", "supervisor")
    graph.add_edge("technical_signals_agent", "supervisor")
    # The writer is terminal: once the brief exists, the run is done.
    graph.add_edge("writer_agent", END)

    checkpointer = checkpointer or MemorySaver()
    # PHASE 2: add `interrupt_before=["writer_agent"]` here so a human can review
    # the collected evidence before the brief is finalised/saved (HITL), using the
    # MemorySaver checkpointer to pause and resume the run.
    return graph.compile(checkpointer=checkpointer)
