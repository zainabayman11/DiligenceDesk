"""Async tests — the compiled graph runs async, and the MCP path is async-native.

WHY async matters here: MCP stdio sessions are asynchronous, so the "real" MCP
learning path is async. The stub/local graph is synchronous, but a LangGraph app
ALSO exposes `.ainvoke()`, so we verify the async entry point works end-to-end.
pytest-asyncio runs these (asyncio_mode=auto in pyproject.toml).
"""

from __future__ import annotations

import os

import pytest

from diligencedesk.graph import build_graph
from diligencedesk.schemas import CompanyTechIntelligenceBrief
from diligencedesk.state import initial_state


async def test_graph_ainvoke_runs_offline():
    # The async entry point must drive the same supervisor -> specialists -> writer
    # flow and produce a valid brief, with no key/network (stub mode).
    app = build_graph()
    config = {"configurable": {"thread_id": "async-1"}}
    await app.ainvoke(initial_state("Acme AI Health"), config=config)
    brief = app.get_state(config).values["brief"]
    assert isinstance(brief, CompanyTechIntelligenceBrief)
    assert any(s.technology == "RAG" for s in brief.ai_and_data_signals)


@pytest.mark.skipif(
    os.getenv("RUN_MCP_TESTS") != "1",
    reason="Needs real MCP servers (uvx + Node). Set RUN_MCP_TESTS=1 to run.",
)
async def test_load_mcp_tools_returns_fetch_and_save():
    # The genuinely-async MCP loader: skipped by default because it spawns the
    # reference fetch (uvx) + filesystem (npx) servers over stdio.
    from diligencedesk.config import get_settings
    from diligencedesk.tools.provider import _load_mcp_tools

    tools = await _load_mcp_tools(get_settings())
    names = {t.name for t in tools}
    assert {"fetch_page", "save_brief"} <= names
