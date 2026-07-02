"""Centralised configuration — one source of truth for behaviour and policy.

WHY a dedicated config module (Twelve-Factor style): behaviour comes from the
environment, never hard-coded. Every other module imports from here, so nothing
reaches into os.environ ad hoc. Two ideas live here that are worth calling out:

1. PER-ROLE models. The supervisor and the specialist "workers" can run on
   DIFFERENT models. The supervisor only routes, so it benefits from a stronger
   reasoning model; the specialists do narrow, well-scoped work, so they run on a
   cheaper/faster small language model (SLM). That split is a real cost/latency
   lever, configured entirely here.

2. POLICY as config. The Responsible-AI golden rule (never invent facts; keep
   inferred evidence separate from confirmed) is a constant in this file, not a
   comment buried in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load a local .env (if present). On a fresh clone / CI there is none, and that
# is fine: stub mode needs no keys, so the defaults below stay fully runnable.
load_dotenv()


def _enable_truststore_if_available() -> None:
    """Trust the OS certificate store, if truststore is installed.

    WHY: corporate proxies / antivirus suites (Avast, Zscaler, ...) MITM HTTPS
    and present their own root CA. Python's bundled certifi store doesn't know
    that CA, so live calls fail with CERTIFICATE_VERIFY_FAILED. truststore
    delegates verification to the OS store, which does trust it. No-op if missing,
    so the project still runs without it (stub mode never makes a network call).
    """
    try:
        import truststore  # optional dependency

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - environment-specific best-effort
        pass


_enable_truststore_if_available()


# --------------------------------------------------------------------------- #
# Providers + per-role default models
# --------------------------------------------------------------------------- #
# "stub" is a first-class provider: keyless, offline, deterministic. It is what
# makes the demo, the tests, and the eval run with no secrets and no network.
SUPPORTED_PROVIDERS = ("groq", "gemini", "ollama", "stub")

# Tool layer modes (see tools/provider.py). Separate axis from the LLM provider:
# you can run the stub LLM with local tools, or a real LLM with stub tools, etc.
SUPPORTED_TOOL_MODES = ("stub", "local", "mcp")

# The roles the LLM layer knows about. The three research specialists all share
# the "worker" role (small/cheap model); the supervisor and writer have their own.
ROLES = ("supervisor", "worker", "writer")

# Default model per (role, provider). The design intent in one table:
#   supervisor -> a STRONGER model (it makes routing decisions),
#   worker/writer -> a small, cheap, fast SLM (narrow, well-scoped jobs).
# Override any of these per role via the *_MODEL env vars below.
_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "groq": {
        # Groq is the default provider: very fast, generous free tier.
        "supervisor": "llama-3.3-70b-versatile",  # stronger router
        "worker": "llama-3.1-8b-instant",  # SLM specialist
        "writer": "llama-3.1-8b-instant",
    },
    "gemini": {
        "supervisor": "gemini-2.5-flash",
        "worker": "gemini-2.5-flash-lite",  # smaller/cheaper SLM
        "writer": "gemini-2.5-flash-lite",
    },
    "ollama": {
        # Fully local SLMs (no key, runs on your machine). Pull these first.
        "supervisor": "llama3.1:8b",
        "worker": "llama3.2:3b",
        "writer": "llama3.2:3b",
    },
    "stub": {"supervisor": "stub", "worker": "stub", "writer": "stub"},
}


def default_model(role: str, provider: str) -> str:
    """Return the built-in default model name for a role on a given provider."""
    return _DEFAULT_MODELS.get(provider, {}).get(role, "stub")


@dataclass(frozen=True)
class RoleConfig:
    """Provider + model for a single role. Frozen so a node can't mutate it."""

    role: str
    provider: str
    model: str


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration for one run."""

    # One RoleConfig per role (supervisor / worker / writer).
    roles: dict[str, RoleConfig]
    # Tool layer mode: stub | local | mcp.
    tool_mode: str
    # API keys (None when unset). Lazily required only by the provider that needs
    # them — stub mode needs neither.
    groq_api_key: str | None
    google_api_key: str | None
    # Hard cap on supervisor routing turns: a buggy router that never finishes is
    # a cost/latency risk, so we bound it. 12 is generous for a 4-specialist team.
    max_supervisor_turns: int
    # Per-HTTP-request timeout (seconds) for local-mode fetch/search.
    http_timeout: int
    # Optional domain allowlist for the local fetch tool. Empty => not enforced.
    # Non-empty => only these domains may be fetched (a read-only safety control).
    allowed_fetch_domains: list[str] = field(default_factory=list)

    def role(self, name: str) -> RoleConfig:
        """Look up a role's config, defaulting unknown roles to 'worker'."""
        return self.roles.get(name) or self.roles["worker"]


def _int_env(name: str, default: int) -> int:
    """Read an int from the env, falling back to a default on missing/garbage.

    Defensive on purpose: an operator typo like MAX_SUPERVISOR_TURNS=twelve must
    not crash startup — we quietly use the safe default.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _provider_env(name: str, default: str) -> str:
    """Read + validate a provider env var, falling back to a sane default."""
    value = os.getenv(name, default).strip().lower()
    return value if value in SUPPORTED_PROVIDERS else default


def _role_config(role: str, default_provider: str) -> RoleConfig:
    """Build one role's config from the environment.

    Env var convention: SUPERVISOR_PROVIDER / WORKER_PROVIDER / WRITER_PROVIDER
    and SUPERVISOR_MODEL / WORKER_MODEL / WRITER_MODEL. If a model isn't set we
    use the per-(role, provider) default, so swapping just the provider still
    picks a sensible model.
    """
    provider = _provider_env(f"{role.upper()}_PROVIDER", default_provider)
    model = os.getenv(f"{role.upper()}_MODEL") or default_model(role, provider)
    return RoleConfig(role=role, provider=provider, model=model)


def get_settings() -> Settings:
    """Build a Settings object from the current environment.

    A function (not a module constant) so tests can monkeypatch os.environ and
    call it again for fresh settings.
    """
    # A single LLM_PROVIDER acts as the default for every role, so the common
    # case ("use Groq for everything") needs one env var. Per-role vars override.
    default_provider = _provider_env("LLM_PROVIDER", "groq")

    roles = {r: _role_config(r, default_provider) for r in ROLES}

    tool_mode = os.getenv("TOOL_MODE", "stub").strip().lower()
    if tool_mode not in SUPPORTED_TOOL_MODES:
        raise ValueError(
            f"Unsupported TOOL_MODE={tool_mode!r}. Choose one of {SUPPORTED_TOOL_MODES}."
        )

    allow_raw = os.getenv("ALLOWED_FETCH_DOMAINS", "").strip()
    allowed = [d.strip() for d in allow_raw.split(",") if d.strip()]

    return Settings(
        roles=roles,
        tool_mode=tool_mode,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        max_supervisor_turns=_int_env("MAX_SUPERVISOR_TURNS", 12),
        http_timeout=_int_env("HTTP_TIMEOUT", 15),
        allowed_fetch_domains=allowed,
    )


# --------------------------------------------------------------------------- #
# Responsible-AI policy (it's config, not a comment — it's the core control)
# --------------------------------------------------------------------------- #
# This single string is prepended to every LLM prompt in the system. It is the
# most important safety control in the project: it forbids inventing facts and
# forces grounding + honest evidence levels. Stated in plain language so a model
# (and a reviewer) cannot miss it.
RESPONSIBLE_AI_POLICY = (
    "You are part of a read-only company TECHNICAL-INTELLIGENCE assistant working "
    "ONLY from public information about a user-provided company. Your output is a "
    "FACTUAL DATA SUMMARY for technical due diligence, not advice. Ground every "
    "statement in a tool result and cite its source. GOLDEN RULE: never invent "
    "technologies, products, or facts. Distinguish CONFIRMED evidence (official "
    "or engineering sources) from INFERRED evidence (job posts, news) and never "
    "present inferred as confirmed. If something is not found in public sources, "
    "say so honestly rather than guessing."
)
