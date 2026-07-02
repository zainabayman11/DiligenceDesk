"""Shared pytest setup — keep the whole suite hermetic and offline.

Two jobs:
1. Force the keyless, offline configuration (stub LLM + stub tools) and disable
   LangSmith BEFORE the package is imported, so no test ever needs a key, a
   network connection, or uvx — a hard requirement for this suite.
2. Reset the stub corpus between tests so a scenario injected by one test can't
   leak into the next.
"""

from __future__ import annotations

import os

# Set BEFORE importing the package. config.py calls load_dotenv(override=False) at
# import, so hard-setting these here makes them win over any real .env — the suite
# is ALWAYS fully offline (stub tools + stub LLM for every role), even if .env is
# configured with Gemini/Groq keys for real runs.
os.environ["TOOL_MODE"] = "stub"
os.environ["LLM_PROVIDER"] = "stub"
for _role in ("SUPERVISOR", "WORKER", "WRITER"):
    os.environ[f"{_role}_PROVIDER"] = "stub"
    # Empty string => config falls back to the per-(role,provider) default. Setting
    # it here stops a real .env's SUPERVISOR_MODEL=gemini-... from leaking into the
    # model-default assertions in test_config.
    os.environ[f"{_role}_MODEL"] = ""
# Never phone home to LangSmith during tests, even if a local .env enables it.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest  # noqa: E402

from diligencedesk.tools import stub_tools  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_corpus():
    """Restore the built-in stub corpus before and after each test."""
    stub_tools.reset_corpus()
    yield
    stub_tools.reset_corpus()
