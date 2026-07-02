"""Provider-agnostic, PER-ROLE LLM construction.

build_llm(role) returns a chat model for whichever provider/model the environment
selects FOR THAT ROLE. The rest of the codebase depends only on the LangChain
BaseChatModel interface (`.invoke(...)`), so swapping Groq for Gemini for Ollama
for the offline stub changes ONE function, not the agents.

WHY per-role (the SLM lesson): the supervisor only routes, so it earns a stronger
model; the specialists do narrow, well-scoped jobs, so they run on a small/cheap
SLM. build_llm("supervisor") and build_llm("worker") can therefore resolve to
different models — a real cost/latency lever with no change to agent code.

WHY lazy imports: a provider's SDK is imported only inside its branch, so stub
mode needs none of them installed, and a Groq user never needs the Gemini package.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import RoleConfig, Settings, get_settings
from .stub_llm import build_stub_llm


def build_llm(role: str, settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model for a given role ('supervisor' | 'worker' | 'writer').

    Args:
        role: Which role's provider/model to use. Unknown roles fall back to the
            'worker' config (see Settings.role).
        settings: Optional pre-loaded settings (tests pass their own).

    Returns:
        A LangChain chat model. Callers use `.invoke(messages)` and parse the
        JSON content (we deliberately avoid provider-specific structured-output
        APIs so the stub stays a drop-in — see stub_llm.py).
    """
    settings = settings or get_settings()
    cfg: RoleConfig = settings.role(role)

    if cfg.provider == "stub":
        # No key, no network. The deterministic fake model.
        return build_stub_llm()

    if cfg.provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                f"{role.upper()}_PROVIDER=groq but GROQ_API_KEY is not set. Add it "
                "to .env, or run with --mode stub for a keyless offline demo."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc
        # temperature=0: a router/extractor wants consistent decisions, not
        # creative variance.
        return ChatGroq(model=cfg.model, api_key=settings.groq_api_key, temperature=0)

    if cfg.provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError(
                f"{role.upper()}_PROVIDER=gemini but GOOGLE_API_KEY is not set. Add "
                "it to .env, or run with --mode stub for a keyless offline demo."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=cfg.model, google_api_key=settings.google_api_key, temperature=0
        )

    if cfg.provider == "ollama":
        # Fully local SLMs: no key, runs on your machine (great for the SLM story).
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "langchain-ollama is not installed. Run: pip install langchain-ollama "
                "(and install Ollama + `ollama pull " + cfg.model + "`)."
            ) from exc
        return ChatOllama(model=cfg.model, temperature=0)

    # get_settings() validates providers, so this is defence in depth.
    raise ValueError(f"Unhandled provider: {cfg.provider!r}")
