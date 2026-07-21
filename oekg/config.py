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


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to ``default``."""
    value = getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass(frozen=True)
class AppConfig:
    """Immutable, process-wide configuration."""

    # --- OpenAI models ---------------------------------------------------
    sparql_model: str = field(default_factory=lambda: _env("OEKG_SPARQL_MODEL", "gpt-5.4-mini"))
    summary_model: str = field(default_factory=lambda: _env("OEKG_SUMMARY_MODEL", "gpt-5.4-nano"))
    embedding_model: str = field(
        default_factory=lambda: _env("OEKG_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    # Optional OpenAI prompt-cache routing key (empty = unset; caching of the
    # large static system prompt is automatic regardless).
    prompt_cache_key: str = field(default_factory=lambda: _env("OEKG_PROMPT_CACHE_KEY", ""))

    # --- Credentials -----------------------------------------------------
    openai_api_key: str | None = field(default_factory=lambda: getenv("OPENAI_API_KEY"))
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

    # --- Logging ---------------------------------------------------------
    log_dir: Path = Path(_env("OEKG_LOG_DIR", str(ROOT_DIR / "logs")))
    log_level: str = _env("OEKG_LOG_LEVEL", "INFO")

    @property
    def request_log_path(self) -> Path:
        return self.log_dir / "requests.log"

    @property
    def error_log_path(self) -> Path:
        return self.log_dir / "errors.log"


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the process-wide :class:`AppConfig` (built once, then cached)."""
    return AppConfig()
