"""Central application configuration.

All tunables (model names, file paths, limits, endpoints) live here so the
rest of the code never hard-codes magic strings. Values can be overridden via
environment variables, which keeps deployments flexible without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from os import getenv
from pathlib import Path

# Repository root (this file lives in <root>/oekg/config.py).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Both endpoints below are operated by Osnabrück University and run on its own
# infrastructure, so no user input leaves German hosting.

# KiWi, the university's LibreChat agents API, serves the chat model. Its
# "model" is an agent id; the agent pins the underlying model and its
# reasoning behaviour server-side.
_KIWI_URL = "https://kiwi-secure.uni-osnabrueck.de/api/agents/v1"

# Two agents rather than one, because KiWi ignores the per-request reasoning
# switch: whether a call reasons is decided by which agent it goes to. The
# reasoning agent spends thousands of hidden tokens before answering, which
# pays off for SPARQL generation and is pure latency for a summary.
_KIWI_REASONING_AGENT = "agent_NpAmRwLMlX5QBrUiqcwjM"
_KIWI_PLAIN_AGENT = "agent_7qE8YPFNPQ9KQQInK9FNc"

# The university's LiteLLM gateway serves the embedding model. bge-m3 returns
# unit-norm 1024-d vectors, so the FAISS L2 index ranks exactly as cosine
# similarity would.
_GATEWAY_URL = "https://litellm.uni-osnabrueck.de/v1"
_EMBEDDING_MODEL = "bge-m3"


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to ``default``."""
    value = getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass(frozen=True)
class AppConfig:
    """Immutable, process-wide configuration."""

    # --- Chat model (KiWi, an OpenAI-compatible Chat Completions endpoint) -
    # The model name is a KiWi agent id, not a model id. Reasoning is bound to
    # the agent, so the split below is what keeps summaries fast.
    llm_base_url: str = field(default_factory=lambda: _env("OEKG_LLM_BASE_URL", _KIWI_URL))
    sparql_model: str = field(
        default_factory=lambda: _env("OEKG_SPARQL_MODEL", _KIWI_REASONING_AGENT)
    )
    summary_model: str = field(
        default_factory=lambda: _env("OEKG_SUMMARY_MODEL", _KIWI_PLAIN_AGENT)
    )
    # Also ask for the reasoning pass to be skipped on the short tasks. KiWi
    # ignores this (hence the two agents above), but an endpoint that honours
    # it -- e.g. the gateway's Qwen -- then needs no second model. See oekg/llm.py.
    no_think_enabled: bool = _env("OEKG_LLM_NO_THINK", "1") not in ("0", "false", "False")
    # Per-attempt timeout. The reasoning agent was measured at over two minutes
    # for a single SPARQL query, so this is deliberately generous: a timeout
    # near the real runtime turns slow answers into failed ones.
    llm_timeout: float = float(_env("OEKG_LLM_TIMEOUT", "300"))
    # Retries after a timeout or a 429/5xx. The SDK default (2) would let one
    # stalled request block the UI for three full timeouts.
    llm_max_retries: int = int(_env("OEKG_LLM_MAX_RETRIES", "1"))

    # Optional prompt-cache routing key (empty = unset; caching of the large
    # static system prompt is automatic regardless).
    prompt_cache_key: str = field(default_factory=lambda: _env("OEKG_PROMPT_CACHE_KEY", ""))

    # --- Embedding model (the university's LiteLLM gateway) --------------
    # A different service from the chat endpoint above. The shipped FAISS index
    # was built with this model on this endpoint; changing either requires
    # rebuilding it: `python -m oekg.build_index`.
    embedding_base_url: str = field(
        default_factory=lambda: _env("OEKG_EMBEDDING_BASE_URL", _GATEWAY_URL)
    )
    embedding_model: str = field(
        default_factory=lambda: _env("OEKG_EMBEDDING_MODEL", _EMBEDDING_MODEL)
    )

    # --- Credentials -----------------------------------------------------
    # Chat and embeddings are two separate services, so they need two keys and
    # neither falls back to the other: sending KiWi's key to the gateway (or
    # vice versa) would hand one service a credential for another.
    llm_api_key: str | None = field(default_factory=lambda: getenv("LLM_API_KEY"))
    embedding_api_key: str | None = field(default_factory=lambda: getenv("EMBEDDING_API_KEY"))
    oep_token: str | None = field(default_factory=lambda: getenv("OEP_TOKEN"))

    # --- Resource files --------------------------------------------------
    faiss_index_path: Path = ROOT_DIR / "vector_store.faiss"
    classes_path: Path = ROOT_DIR / "classes.json"
    relations_path: Path = ROOT_DIR / "relations_final.json"
    ids_path: Path = ROOT_DIR / "ids.json"
    sparql_prompt_path: Path = ROOT_DIR / "sparql_system_prompt.txt"
    summary_prompt_path: Path = ROOT_DIR / "summary_system_prompt.txt"
    logo_path: Path = ROOT_DIR / "logo.svg"

    # --- Privacy / consent gate ------------------------------------------
    # The policy shown before the chatbot may be used. A deployment can point
    # this at its own document; consent is re-requested whenever it changes.
    privacy_policy_path: Path = Path(
        _env("OEKG_PRIVACY_POLICY_PATH", str(ROOT_DIR / "PRIVACY.md"))
    )
    # Only disable this where consent is already obtained elsewhere (e.g. an
    # embedding portal) -- the chatbot sends user input to a third country.
    require_consent: bool = _env("OEKG_REQUIRE_CONSENT", "1") not in ("0", "false", "False")

    # --- OEP data REST API (dataset preview + metadata) -----------------
    oep_api_base: str = field(
        default_factory=lambda: _env("OEKG_OEP_API_BASE", "https://openenergyplatform.org/api/v0")
    )
    dataset_preview_rows: int = int(_env("OEKG_DATASET_PREVIEW_ROWS", "50"))

    # --- Retrieval / pipeline limits ------------------------------------
    retrieval_top_k: int = int(_env("OEKG_RETRIEVAL_TOP_K", "10"))
    max_user_history: int = int(_env("OEKG_MAX_USER_HISTORY", "5"))
    max_summary_input_chars: int = int(_env("OEKG_MAX_SUMMARY_CHARS", "10000"))
    max_input_chars: int = int(_env("OEKG_MAX_INPUT_CHARS", "2000"))

    # --- SPARQL generation guards ---------------------------------------
    uri_validation_enabled: bool = _env("OEKG_URI_VALIDATION", "1") not in ("0", "false", "False")
    # Strip stray markdown fences / labels from the model's raw output.
    normalise_output_enabled: bool = _env("OEKG_NORMALISE_OUTPUT", "1") not in ("0", "false", "False")
    # Parse the query locally (rdflib) before executing; advisory repair only.
    syntax_validation_enabled: bool = _env("OEKG_SYNTAX_VALIDATION", "1") not in ("0", "false", "False")
    # Flag predicate objects outside a relation's declared range.
    schema_validation_enabled: bool = _env("OEKG_SCHEMA_VALIDATION", "1") not in ("0", "false", "False")
    # Resolve entity mentions against real graph labels before generation.
    entity_grounding_enabled: bool = _env("OEKG_ENTITY_GROUNDING", "1") not in ("0", "false", "False")
    grounding_max_mentions: int = int(_env("OEKG_GROUNDING_MAX_MENTIONS", "6"))
    grounding_min_mention_len: int = int(_env("OEKG_GROUNDING_MIN_LEN", "3"))
    grounding_results_per_mention: int = int(_env("OEKG_GROUNDING_RESULTS_PER_MENTION", "5"))
    # On empty results, fetch real near-miss labels to enrich the retry.
    near_miss_enabled: bool = _env("OEKG_NEAR_MISS", "1") not in ("0", "false", "False")
    near_miss_limit: int = int(_env("OEKG_NEAR_MISS_LIMIT", "8"))
    near_miss_min_token_len: int = int(_env("OEKG_NEAR_MISS_MIN_TOKEN", "4"))

    # --- OEP SPARQL endpoint --------------------------------------------
    sparql_endpoint: str = field(
        default_factory=lambda: _env(
            "OEKG_SPARQL_ENDPOINT", "https://openenergyplatform.org/api/v0/oekg/sparql/"
        )
    )
    sparql_retries: int = int(_env("OEKG_SPARQL_RETRIES", "3"))
    sparql_retry_delay: float = float(_env("OEKG_SPARQL_RETRY_DELAY", "2"))
    sparql_timeout: int = int(_env("OEKG_SPARQL_TIMEOUT", "30"))
    # How many times a query that errors at the endpoint may be regenerated by
    # the LLM from the endpoint's error message before giving up.
    sparql_repair_rounds: int = int(_env("OEKG_SPARQL_REPAIR_ROUNDS", "2"))

    # --- Caching ---------------------------------------------------------
    cache_enabled: bool = _env("OEKG_CACHE_ENABLED", "1") not in ("0", "false", "False")
    cache_path: Path = Path(_env("OEKG_CACHE_PATH", str(ROOT_DIR / "query_cache.json")))
    # Semantic (paraphrase) cache: when no exact match exists, offer the closest
    # cached question above this cosine similarity for the user to confirm.
    semantic_cache_enabled: bool = _env("OEKG_SEMANTIC_CACHE", "1") not in ("0", "false", "False")
    semantic_cache_threshold: float = float(_env("OEKG_SEMANTIC_CACHE_THRESHOLD", "0.92"))

    # --- Health check (python -m oekg.healthcheck) -----------------------
    # No recipient or relay is baked in: set them in the environment of the
    # host that runs the check. All lazy, see the note above the class.
    health_app_url: str = field(
        default_factory=lambda: _env(
            "OEKG_HEALTH_APP_URL", "http://127.0.0.1:3030/_stcore/health"
        )
    )
    # Consecutive failing runs before the first alert mail, so one blip is not one mail.
    health_failure_threshold: int = field(
        default_factory=lambda: int(_env("OEKG_HEALTH_FAILURES", "2"))
    )
    health_mail_to: str = field(default_factory=lambda: _env("OEKG_HEALTH_MAIL_TO", ""))
    health_mail_from: str = field(default_factory=lambda: _env("OEKG_HEALTH_MAIL_FROM", ""))
    health_smtp_host: str = field(default_factory=lambda: _env("OEKG_HEALTH_SMTP_HOST", ""))
    health_smtp_port: int = field(
        default_factory=lambda: int(_env("OEKG_HEALTH_SMTP_PORT", "587"))
    )
    health_smtp_starttls: bool = field(
        default_factory=lambda: _env("OEKG_HEALTH_SMTP_STARTTLS", "1")
        not in ("0", "false", "False")
    )
    health_smtp_user: str = field(default_factory=lambda: _env("OEKG_HEALTH_SMTP_USER", ""))
    health_smtp_password: str | None = field(
        default_factory=lambda: getenv("HEALTH_SMTP_PASSWORD")
    )

    # --- Logging ---------------------------------------------------------
    log_dir: Path = Path(_env("OEKG_LOG_DIR", str(ROOT_DIR / "logs")))
    log_level: str = _env("OEKG_LOG_LEVEL", "INFO")

    @property
    def request_log_path(self) -> Path:
        return self.log_dir / "requests.log"

    @property
    def error_log_path(self) -> Path:
        return self.log_dir / "errors.log"

    @property
    def health_state_path(self) -> Path:
        return self.log_dir / "health_state.json"


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide :class:`AppConfig` (built once, then cached)."""
    return AppConfig()
