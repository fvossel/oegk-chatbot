"""End-to-end orchestration of the RAG pipeline.

Flow: build the dialogue context, look up the question in the cache (or
retrieve context + generate a SPARQL query via the LLM), validate the generated
identifiers, execute the query against the OEP (with LLM self-correction on
endpoint errors), then format and summarise the results. Every request is
logged. Queries are not cached automatically -- the UI exposes an explicit
"save to cache" action backed by the ``sparql`` field of :class:`PipelineResult`.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, ContextManager, Iterable, Iterator

from openai import OpenAI
from pandas import DataFrame

from oekg.cache import CacheEntry, QueryCache
from oekg.config import AppConfig, get_config
from oekg.llm import LLMClient
from oekg.logging_setup import get_request_logger
from oekg.oep import OEPClient
from oekg.resources import Resources
from oekg.retrieval import Retriever
from oekg.validation import build_known_vocabulary, find_unknown_uris

logger = logging.getLogger(__name__)

# A status reporter yields a context manager for a step label (e.g. a spinner).
StatusReporter = Callable[[str], ContextManager]

BOT_INFO_TAG = "<bot-information>"
GENERATION_ERROR_MESSAGE = (
    "⚠️ Sorry, I couldn't generate a query for that right now because the "
    "language-model request failed. Please try again in a moment."
)
_TABLE_FALLBACK = "The results are shown in the table below."


@contextmanager
def _null_status(_label: str):
    """No-op status reporter used when the caller provides none."""
    yield


@dataclass(frozen=True)
class PipelineResult:
    """Result of a single pipeline run."""

    answer: str
    cache_hit: bool
    is_bot_message: bool
    row_count: int
    duration_s: float
    sparql: str | None = None
    results_df: DataFrame | None = None
    full_query: str | None = None
    error: bool = False
    corrected: bool = False
    # Transient: a lazy summary stream for the UI to render; never stored.
    summary_stream: Iterator[str] | None = None


class RagPipeline:
    """Coordinates retrieval, generation, execution and summarisation."""

    def __init__(
        self,
        resources: Resources,
        cache: QueryCache,
        config: AppConfig | None = None,
        *,
        client: OpenAI | None = None,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        oep: OEPClient | None = None,
    ) -> None:
        self._config = config or get_config()
        self._resources = resources
        self._cache = cache
        self._logger = get_request_logger()
        self._known_vocab: set[str] | None = None

        # Collaborators default to real implementations but can be injected
        # (e.g. for testing). The OpenAI client is reused across requests.
        self._client = client or OpenAI(api_key=self._config.openai_api_key)
        self._retriever = retriever or Retriever(self._client, resources, self._config)
        self._llm = llm or LLMClient(self._client, resources, self._config)
        self._oep = oep or OEPClient(self._config.oep_token, self._config)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _build_dialogue(history: Iterable[tuple[str, str]], question: str) -> str:
        """Render the chat history plus the new question as a single prompt."""
        lines = []
        for role, msg in history:
            prefix = "User:" if role == "user" else "Assistant:"
            lines.append(f"{prefix} {msg}")
        lines.append(f"User: {question}")
        return "\n".join(lines) + "\n"

    def _retry_prompt(self, dialogue: str, previous_sparql: str) -> str:
        return (
            dialogue
            + "The previous SPARQL query returned no results: \n"
            + previous_sparql
            + "\n Please check the query again and also check if your constraints were "
            "too harsh. But also take into account that maybe you have to search for the "
            "pattern in the subclasses labels of the mentioned entity. So try again and "
            "maybe use some fancy tricks."
        )

    def _repair_prompt(self, dialogue: str, previous_sparql: str, error: str) -> str:
        return (
            dialogue
            + "\nThe previous SPARQL query failed when executed against the endpoint.\n"
            + "Previous query:\n"
            + previous_sparql
            + "\nEndpoint error:\n"
            + error
            + "\nFix the query so it is valid SPARQL for this knowledge graph. "
            "Return ONLY the corrected query."
        )

    def _uri_repair_prompt(
        self, dialogue: str, previous_sparql: str, unknown: dict[str, tuple[str | None, str | None]]
    ) -> str:
        hints = []
        for term, (suggestion, label) in unknown.items():
            if suggestion:
                hint = f"- {term} is not a known identifier; did you mean {suggestion}"
                hint += f" ({label})?" if label else "?"
            else:
                hint = f"- {term} is not a known identifier in this knowledge graph."
            hints.append(hint)
        return (
            dialogue
            + "\nThe previous SPARQL query used identifiers that are not in the "
            "knowledge-graph vocabulary:\n"
            + previous_sparql
            + "\n"
            + "\n".join(hints)
            + "\nRegenerate the query using only valid identifiers from the provided "
            "context. Return ONLY the corrected query."
        )

    def _vocabulary(self) -> set[str]:
        if self._known_vocab is None:
            self._known_vocab = build_known_vocabulary(self._resources)
        return self._known_vocab

    # -- public API -------------------------------------------------------
    def explain_sparql(self, query: str) -> str:
        """Return a plain-language explanation of a SPARQL query (best effort)."""
        return self._llm.explain_sparql(query)

    def embed(self, text: str) -> list[float]:
        """Embed a piece of text with the configured embedding model."""
        response = self._client.embeddings.create(
            model=self._config.embedding_model, input=text
        )
        return list(response.data[0].embedding)

    def find_semantic_match(self, question: str) -> tuple[CacheEntry, float] | None:
        """Return the closest paraphrase in the cache (no exact hit), or ``None``.

        Returns ``None`` when semantic matching is disabled, an exact hit exists,
        or there are no embedded cache entries to compare against (so no
        embedding API call is made in that case).
        """
        if not self._config.semantic_cache_enabled:
            return None
        if self._cache.get(question) is not None:
            return None
        if not self._cache.has_embeddings():
            return None
        return self._cache.semantic_match(
            self.embed(question), self._config.semantic_cache_threshold
        )

    def answer(
        self,
        question: str,
        history: Iterable[tuple[str, str]] = (),
        status: StatusReporter | None = None,
        prefer_cache: bool = True,
        stream: bool = False,
        forced_sparql: str | None = None,
    ) -> PipelineResult:
        """Answer ``question`` and return a :class:`PipelineResult`.

        When a cached SPARQL query exists for the question (and ``prefer_cache``
        is set), generation and retrieval are skipped and the query is executed
        directly. ``forced_sparql`` runs a specific query (e.g. a user-confirmed
        semantic-cache match) the same way. ``stream`` returns the summary as a
        lazy text stream in :attr:`PipelineResult.summary_stream`.
        """
        status = status or _null_status
        history = list(history)
        started = time.perf_counter()
        dialogue = self._build_dialogue(history, question)
        corrected = False

        cached = self._cache.get(question) if (prefer_cache and forced_sparql is None) else None

        if forced_sparql is not None:
            sparql_query = forced_sparql
            cache_hit = True
            top_k: list | None = None
            on_empty: Callable[[str], str] | None = None
            on_error: Callable[[str, str], str] | None = None
        elif cached is not None:
            self._cache.record_hit(question)
            sparql_query = cached.sparql
            cache_hit = True
            top_k = None
            on_empty = None
            on_error = None
        else:
            cache_hit = False
            with status("Retrieving context information..."):
                # Embed only the question (not the whole dialogue) so the query
                # vector stays focused on the current intent.
                top_k = self._retriever.top_k(question)
            try:
                with status("Generating SPARQL query..."):
                    sparql_query = self._llm.generate_sparql(dialogue, top_k)
            except Exception:  # noqa: BLE001 - surface a clean error to the user
                logger.exception("SPARQL generation failed")
                return self._finish(
                    PipelineResult(
                        answer=GENERATION_ERROR_MESSAGE,
                        cache_hit=False,
                        is_bot_message=False,
                        row_count=0,
                        duration_s=time.perf_counter() - started,
                        error=True,
                    ),
                    question,
                )
            on_empty = lambda prev: self._llm.generate_sparql(  # noqa: E731
                self._retry_prompt(dialogue, prev), top_k
            )
            on_error = lambda prev, err: self._llm.generate_sparql(  # noqa: E731
                self._repair_prompt(dialogue, prev, err), top_k
            )

        # Conversational (non-query) replies are returned verbatim, not cached.
        # Only ever triggered on freshly generated output, never on cached SPARQL.
        if not cache_hit and BOT_INFO_TAG in sparql_query:
            answer = sparql_query[sparql_query.find(BOT_INFO_TAG) + len(BOT_INFO_TAG):]
            return self._finish(
                PipelineResult(
                    answer=answer,
                    cache_hit=cache_hit,
                    is_bot_message=True,
                    row_count=0,
                    duration_s=time.perf_counter() - started,
                    sparql=None,
                ),
                question,
            )

        # Schema-grounded check: if the model invented identifiers, ask it once
        # to regenerate using valid ones (cheap, offline detection).
        if not cache_hit and top_k is not None and self._config.uri_validation_enabled:
            unknown = find_unknown_uris(
                sparql_query, self._vocabulary(), self._resources.documents_dict
            )
            if unknown:
                logger.info("URI validation flagged unknown term(s): %s", list(unknown))
                try:
                    with status("Checking and fixing identifiers..."):
                        sparql_query = self._llm.generate_sparql(
                            self._uri_repair_prompt(dialogue, sparql_query, unknown), top_k
                        )
                    corrected = True
                except Exception:  # noqa: BLE001 - keep the original query on failure
                    logger.exception("URI repair failed")

        with status("Extracting requested information..."):
            execution = self._oep.execute(sparql_query, on_empty=on_empty, on_error=on_error)
        corrected = corrected or execution.repaired

        df = execution.results_df
        results_md = "Query Results:\n" + (
            df.to_markdown(index=False) if not df.empty else "No results found."
        )
        should_summarise = not df.empty and len(results_md) < self._config.max_summary_input_chars

        summary_stream: Iterator[str] | None = None
        if df.empty:
            content = "No results found for this query."
        elif should_summarise and stream:
            content = ""
            summary_stream = self._llm.summarise_stream(results_md, dialogue)
        elif should_summarise:
            with status("Finalizing result formatting..."):
                content = self._llm.summarise(results_md, dialogue) or _TABLE_FALLBACK
        else:
            content = _TABLE_FALLBACK

        # Note: queries are *not* cached automatically. Caching is an explicit
        # user action (see the "Save to cache" button in the UI), driven by the
        # ``sparql`` field returned below.
        return self._finish(
            PipelineResult(
                answer=content,
                cache_hit=cache_hit,
                is_bot_message=False,
                row_count=0 if df.empty else len(df),
                duration_s=time.perf_counter() - started,
                sparql=execution.sparql_used,
                results_df=None if df.empty else df,
                full_query=execution.full_query,
                corrected=corrected,
                summary_stream=summary_stream,
            ),
            question,
        )

    def close(self) -> None:
        """Release the underlying OpenAI client."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            pass

    # -- logging ----------------------------------------------------------
    def _finish(self, result: PipelineResult, question: str) -> PipelineResult:
        self._logger.info(
            "request | cache_hit=%s | bot_message=%s | error=%s | corrected=%s | rows=%s | duration=%.2fs | question=%r",
            result.cache_hit,
            result.is_bot_message,
            result.error,
            result.corrected,
            result.row_count,
            result.duration_s,
            question,
        )
        return result
