"""Tool layer: one interface, three modes (stub | local | mcp).

The public entry point is `get_tools(mode)` in provider.py. Agents are written
against the uniform tool NAMES (web_search, fetch_page, save_brief) and never know
which mode produced them — that is the whole standardization lesson of this layer.
"""
