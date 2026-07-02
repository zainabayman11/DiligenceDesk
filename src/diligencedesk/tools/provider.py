"""The tool PROVIDER: one interface, three modes.

`get_tools(mode)` returns a Toolbox — a tiny name->tool registry over plain
LangChain tools. The agents look tools up by NAME (web_search, fetch_page,
save_brief) and never learn which mode produced them. That is the standardization
lesson: swapping stub <-> local <-> mcp changes NO agent code.

The three modes:
  stub  : canned, fully offline, deterministic. Default for tests/eval/demo.
  local : real public data (ddgs/httpx) + a local file write. Works with just an
          LLM key — zero MCP setup.
  mcp   : the MCP learning path. fetch + filesystem come from real MCP servers
          (stdio via uvx/npx); search stays local for now.
          # V1.5: migrate search (+ GitHub/careers) to MCP servers too.

MCP-security posture (see README/ARCHITECTURE): trusted servers only, read-only,
the filesystem server is scoped to ONE directory (outputs/), short timeouts, no
arbitrary shell. If the MCP servers can't be reached we fall back to the local
tools rather than crash — resilience over fragility.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from ..config import Settings, get_settings
from .local_tools import LOCAL_TOOLS
from .stub_tools import STUB_TOOLS

# The canonical tool names every mode must provide. Agents depend on these.
# (financials were removed in the V1 technical-intelligence pivot; find_jobs was
# added for Egypt-scoped hiring signals.)
REQUIRED_TOOLS = ("web_search", "fetch_page", "find_jobs", "save_brief")


class Toolbox:
    """A name -> LangChain tool registry. The uniform interface the agents use."""

    def __init__(self, tools: list[BaseTool]):
        self._by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def get(self, name: str) -> BaseTool:
        if name not in self._by_name:
            raise KeyError(
                f"Tool {name!r} not in this toolbox. Available: {sorted(self._by_name)}"
            )
        return self._by_name[name]

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())


def get_tools(mode: str | None = None, settings: Settings | None = None) -> Toolbox:
    """Return the Toolbox for a tool mode ('stub' | 'local' | 'mcp').

    Args:
        mode: Tool mode. Defaults to settings.tool_mode (env TOOL_MODE).
        settings: Config snapshot (mcp needs it for the outputs path / timeouts).

    Returns:
        A Toolbox exposing web_search, fetch_page, save_brief.
    """
    settings = settings or get_settings()
    mode = (mode or settings.tool_mode).lower()

    if mode == "stub":
        return Toolbox(STUB_TOOLS)
    if mode == "local":
        return Toolbox(LOCAL_TOOLS)
    if mode == "mcp":
        return Toolbox(_mcp_tools(settings))
    raise ValueError(f"Unknown tool mode: {mode!r}. Choose stub | local | mcp.")


# --------------------------------------------------------------------------- #
# MCP mode (the learning path)
# --------------------------------------------------------------------------- #
def _resolve_launcher(name: str, env_override: str) -> str:
    """Find a launcher (uvx / npx) even when it isn't on PATH.

    On Windows, `pip install uv` drops uvx.exe in the Python Scripts folder, which
    is often NOT on PATH — so a bare "uvx" would fail to spawn and MCP would
    silently fall back to local. We resolve it robustly:
      1. an explicit env override (DILIGENCE_UVX / DILIGENCE_NPX),
      2. PATH (shutil.which),
      3. next to the current Python (Scripts/ on Windows, bin/ in a venv),
      4. the bare name as a last resort.
    """
    override = os.getenv(env_override)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    exe_dir = Path(sys.executable).resolve().parent
    for cand in (exe_dir / "Scripts" / f"{name}.exe", exe_dir / f"{name}.exe", exe_dir / name):
        if cand.exists():
            return str(cand)
    return name  # let the OS try; the graceful fallback catches a failure


def _mcp_server_config(settings: Settings) -> dict:
    """stdio launch config for the trusted MCP servers we use.

    - fetch: the reference Python fetch server (read-only web fetch), via uvx.
    - filesystem: the reference Node filesystem server, SCOPED to outputs/ only,
      via npx. Scoping the path is the key safety control: the server cannot
      touch anything outside that one directory.
    """
    outputs = Path(__file__).resolve().parents[3] / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    # TLS behind a corporate proxy / antivirus (e.g. Avast) that MITMs HTTPS: the
    # bundled cert stores don't know the interceptor's root CA, so uvx (PyPI) and
    # npx (npm) downloads fail with UnknownIssuer. We tell each launcher to use the
    # OS certificate store, which DOES trust that CA:
    #   uv  -> UV_SYSTEM_CERTS=1
    #   node-> --use-system-ca (Node 22+)
    # env must be merged with os.environ so PATH etc. survive.
    fetch_env = {**os.environ, "UV_SYSTEM_CERTS": "1"}
    fs_env = {**os.environ, "NODE_OPTIONS": "--use-system-ca"}
    return {
        "fetch": {
            "command": _resolve_launcher("uvx", "DILIGENCE_UVX"),
            "args": ["mcp-server-fetch"],
            "transport": "stdio",
            "env": fetch_env,
        },
        "filesystem": {
            "command": _resolve_launcher("npx", "DILIGENCE_NPX"),
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(outputs)],
            "transport": "stdio",
            "env": fs_env,
        },
    }


async def _load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Connect to the MCP servers and return wrapped fetch_page + save_brief.

    Async because MCP stdio sessions are async — this is the one place the spec's
    "async where MCP requires it" actually bites. Loaded once at graph-build time.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient  # lazy

    client = MultiServerMCPClient(_mcp_server_config(settings))
    mcp_tools = await client.get_tools()
    by_name = {t.name: t for t in mcp_tools}

    # The reference servers expose "fetch" and "write_file". We wrap each behind
    # our canonical tool name + JSON contract so agents see no difference.
    raw_fetch = by_name.get("fetch")
    raw_write = by_name.get("write_file")
    if raw_fetch is None or raw_write is None:
        raise RuntimeError(
            f"MCP servers did not expose expected tools. Got: {sorted(by_name)}"
        )

    outputs = Path(__file__).resolve().parents[3] / "outputs"

    @tool
    def fetch_page(url: str) -> str:
        """Fetch a web page via the MCP fetch server; returns our JSON contract."""
        # WHY asyncio.run per call: our V1 graph is synchronous, so we bridge to
        # the async MCP tool here. # V1.5: run the whole graph async with a
        # persistent client instead of a fresh session per call.
        try:
            text = asyncio.run(raw_fetch.ainvoke({"url": url}))
            return json.dumps(
                {"url": url, "ok": True, "title": url, "text": str(text)[:4000], "source": url}
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"url": url, "ok": False, "title": "", "text": "", "source": url,
                 "error": f"mcp fetch failed: {exc}"}
            )

    @tool
    def save_brief(content: str, filename: str = "") -> str:
        """Write the brief via the MCP filesystem server (scoped to outputs/)."""
        try:
            from datetime import datetime

            name = (filename or "brief").replace("/", "-")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(outputs / f"{name}_{stamp}.md")
            asyncio.run(raw_write.ainvoke({"path": path, "content": content}))
            return json.dumps({"ok": True, "path": path, "source": "mcp-filesystem"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "path": "", "source": "mcp-filesystem", "error": str(exc)})

    return [fetch_page, save_brief]


def _mcp_startup_timeout() -> float:
    """Seconds to wait for the MCP servers before falling back (env MCP_STARTUP_TIMEOUT)."""
    try:
        return float(os.getenv("MCP_STARTUP_TIMEOUT", "60"))
    except ValueError:
        return 60.0


async def _load_mcp_tools_timed(settings: Settings, timeout: float) -> list[BaseTool]:
    """Load the MCP tools but give up after `timeout` seconds (cancels cleanly)."""
    return await asyncio.wait_for(_load_mcp_tools(settings), timeout=timeout)


def _mcp_tools(settings: Settings) -> list[BaseTool]:
    """Assemble the mcp-mode toolbox (sync wrapper around the async loader).

    search + jobs stay local; fetch + save come from MCP. Starting the servers can
    be slow (the FIRST run downloads them over the network), so we bound it with a
    TIMEOUT — instead of hanging forever you get an automatic, clean fall back to
    the local fetch/save tools. Raise MCP_STARTUP_TIMEOUT if your first download is
    slow behind a proxy/antivirus.
    """
    from .local_tools import fetch_page as local_fetch
    from .local_tools import find_jobs, save_brief as local_save, web_search

    tools: list[BaseTool] = [web_search, find_jobs]
    timeout = _mcp_startup_timeout()
    print(
        f"[mcp] starting MCP servers via uvx/npx (first run downloads them; up to {timeout:.0f}s)...",
        file=sys.stderr,
    )
    try:
        tools += asyncio.run(_load_mcp_tools_timed(settings, timeout))
        print("[mcp] MCP fetch + filesystem servers ready.", file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 - resilience: fall back to local
        # BaseException (not Exception): a failed/cancelled MCP connect surfaces via
        # anyio as a BaseExceptionGroup (it can wrap asyncio.CancelledError, a
        # BaseException), so `except Exception` would miss it. We catch everything
        # (incl. the timeout) and degrade to the local fetch/save tools.
        reason = "timed out" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else (
            str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        )
        print(
            f"[mcp] Could not start MCP servers ({reason}). Falling back to local "
            "fetch/save. (Needs uv/uvx + Node/npx + network to download the servers "
            "on first run; raise MCP_STARTUP_TIMEOUT if the download is slow.)",
            file=sys.stderr,
        )
        tools += [local_fetch, local_save]
    return tools
