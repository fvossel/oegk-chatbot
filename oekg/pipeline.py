"""End-to-end orchestration of the RAG pipeline.

Flow (fresh path): retrieve context -> ground entity mentions against the graph
-> generate SPARQL -> normalise output -> validate identifiers (existence) ->
validate predicate ranges -> execute (with local syntax pre-flight, LLM
self-correction on endpoint errors, and near-miss-enriched retry on empty
results) -> summarise. Every request is logged. Queries are not cached
automatically -- the UI exposes an explicit "save to cache" action backed by the
``sparql`` field of :class:`PipelineResult`.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterable, Iterator

from openai import OpenAI
from pandas import DataFrame

from oekg.cache import CacheEntry, QueryCache
from oekg.config import AppConfig, get_config
from oekg.dataapi import OEPDataClient
from oekg.grounding import ground_question
from oekg.llm import LLMClient
from oekg.logging_setup import get_request_logger
from oekg.oep import OEPClient
from oekg.ontology import term_links
from oekg.resources import Resources
from oekg.retrieval import Retriever
from oekg.validation import (
    build_known_vocabulary,
    build_relation_schema,
    extract_literal_mentions,
    find_range_violations,
    find_unknown_uris,
    normalise_sparql,
    summarise_predicate_schema,
)

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
        data: OEPDataClient | None = None,
    ) -> None:
        self._config = config or get_config()
        self._resources = resources
        self._cache = cache
        self._logger = get_request_logger()
        self._known_vocab: set[str] | None = None
        self._relation_schema: dict[str, dict[str, Any]] | None = None

        # Collaborators default to real implementations but can be injected
        # (e.g. for testing). The OpenAI client is reused across requests.
        self._client = client or OpenAI(api_key=self._config.openai_api_key)
        self._retriever = retriever or Retriever(self._client, resources, self._config)
        self._llm = llm or LLMClient(self._client, resources, self._config)
        self._oep = oep or OEPClient(self._config.oep_token, self._config)
        self._data = data or OEPDataClient(self._config.oep_token, self._config)

    # -- prompt helpers ---------------------------------------------------
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

    def _near_miss_retry_prompt(
        self, dialogue: str, previous_sparql: str, mentions: list[str], labels: list[str]
    ) -> str:
        return (
            dialogue
            + "\nThe previous SPARQL query returned no results:\n"
            + previous_sparql
            + "\nNo exact match was found for: "
            + ", ".join(f'"{m}"' for m in mentions)
            + ".\nLabels that DO exist in the graph and are similar: "
            + ", ".join(f'"{label}"' for label in labels)
            + ".\nRewrite the query to match one of these existing labels (an exact "
            "rdfs:label or a CONTAINS filter on a shared token), or relax the constraint. "
            "Return ONLY the corrected query."
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

    def _range_repair_prompt(
        self,
        dialogue: str,
        previous_sparql: str,
        violations: dict[str, dict[str, Any]],
        guidance: dict[str, dict[str, Any]],
    ) -> str:
        hints = []
        for pred, info in violations.items():
            allowed = ", ".join(info["range_labels"][:20]) or "(none listed)"
            line = (
                f"- you used {info['object']} as the object of "
                f"'{info['pred_label']}' ({pred}), but its allowed values are: {allowed}."
            )
            suggestion, suggestion_label = info["suggestion"]
            if suggestion:
                line += f" Did you mean {suggestion}"
                line += f" ({suggestion_label})?" if suggestion_label else "?"
            hints.append(line)

        guidance_lines = [
            f"- {info['label']} ({pred}): range = {', '.join(info['range_labels'][:20])}"
            for pred, info in guidance.items()
            if info["range_labels"]
        ]
        reference = (
            "\n\nSchema reference for predicates you used:\n" + "\n".join(guidance_lines)
            if guidance_lines
            else ""
        )
        return (
            dialogue
            + "\nThe previous SPARQL query used predicate objects outside their allowed range:\n"
            + previous_sparql
            + "\n"
            + "\n".join(hints)
            + reference
            + "\nRegenerate using only objects allowed by each predicate's range. "
            "Return ONLY the corrected query."
        )

    # -- memoised resources ----------------------------------------------
    def _vocabulary(self) -> set[str]:
        if self._known_vocab is None:
            self._known_vocab = build_known_vocabulary(self._resources)
        return self._known_vocab

    def _schema(self) -> dict[str, dict[str, Any]]:
        if self._relation_schema is None:
            self._relation_schema = build_relation_schema(self._resources)
        return self._relation_schema

    # -- generation + grounding ------------------------------------------
    def _gen(self, prompt: str, top_k: list, grounding: str = "") -> str:
        """Generate SPARQL and normalise the raw output if enabled."""
        out = self._llm.generate_sparql(prompt, top_k, grounding=grounding or None)
        if self._config.normalise_output_enabled:
            out = normalise_sparql(out)
        return out

    def _run_grounding_query(self, query: str) -> dict[str, Any]:
        return self._oep.run_sparql(self._oep.build_full_query(query))

    def _ground(self, question: str) -> str:
        try:
            return ground_question(
                question,
                self._run_grounding_query,
                max_mentions=self._config.grounding_max_mentions,
                min_len=self._config.grounding_min_mention_len,
                limit_per_mention=self._config.grounding_results_per_mention,
            )
        except Exception:  # noqa: BLE001 - grounding must never break a request
            logger.exception("entity grounding failed")
            return ""

    def _on_empty(self, dialogue: str, top_k: list, grounding: str) -> Callable[[str], str]:
        """Build the empty-result retry callback, enriched with near-miss labels."""

        def callback(previous_sparql: str) -> str:
            if self._config.near_miss_enabled:
                try:
                    mentions = extract_literal_mentions(previous_sparql)
                    if mentions:
                        labels = self._oep.find_near_miss_labels(
                            mentions, self._config.near_miss_limit
                        )
                        if labels:
                            return self._gen(
                                self._near_miss_retry_prompt(
                                    dialogue, previous_sparql, mentions, labels
                                ),
                                top_k,
                                grounding,
                            )
                except Exception:  # noqa: BLE001 - degrade to the plain retry
                    logger.exception("near-miss label fetch failed")
            return self._gen(self._retry_prompt(dialogue, previous_sparql), top_k, grounding)

        return callback

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
        """Return the closest paraphrase in the cache (no exact hit), or ``None``."""
        if not self._config.semantic_cache_enabled:
            return None
        if self._cache.get(question) is not None:
            return None
        if not self._cache.has_embeddings():
            return None
        return self._cache.semantic_match(
            self.embed(question), self._config.semantic_cache_threshold
        )

    # -- OEP data integration (lazy, on-demand) --------------------------
    def dataset_references(self, df: DataFrame) -> list[tuple[str, str]]:
        """Return distinct ``(url, table)`` pairs for OEP datasets in a result."""
        seen: dict[str, str] = {}
        for column in df.columns:
            for value in df[column].tolist():
                table = OEPDataClient.table_from_url(value) if isinstance(value, str) else None
                if table and value not in seen:
                    seen[value] = table
        return list(seen.items())

    def dataset_preview(self, table: str) -> DataFrame | None:
        """Fetch a small preview of an OEP dataset table's actual rows."""
        return self._data.fetch_rows(table, self._config.dataset_preview_rows)

    def dataset_metadata(self, table: str) -> dict[str, Any] | None:
        """Fetch the summarised OEMetadata card for an OEP dataset table."""
        return self._data.metadata_card(table)

    def ontology_links(self, df: DataFrame) -> list[dict[str, str]]:
        """Return OEO term links for any ontology IRIs present in a result."""
        values = [value for column in df.columns for value in df[column].tolist()]
        return term_links(values, self._resources.documents_dict)

    def answer(
        self,
        question: str,
        history: Iterable[tuple[str, str]] = (),
        status: StatusReporter | None = None,
        prefer_cache: bool = True,
        stream: bool = False,
        forced_sparql: str | None = None,
    ) -> PipelineResult:
        """Answer ``question`` and return a :class:`PipelineResult`."""
        status = status or _null_status
        history = list(history)
        started = time.perf_counter()
        dialogue = self._build_dialogue(history, question)
        corrected = False
        grounding = ""

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
                top_k = self._retriever.top_k(question)
            if self._config.entity_grounding_enabled:
                with status("Grounding entity names..."):
                    grounding = self._ground(question)
            try:
                with status("Generating SPARQL query..."):
                    sparql_query = self._gen(dialogue, top_k, grounding)
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
            on_empty = self._on_empty(dialogue, top_k, grounding)
            on_error = lambda prev, err: self._gen(  # noqa: E731
                self._repair_prompt(dialogue, prev, err), top_k, grounding
            )

        # Conversational (non-query) replies are returned verbatim, not cached.
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

        # Schema-grounded check: invented identifiers -> one regeneration.
        if not cache_hit and top_k is not None and self._config.uri_validation_enabled:
            unknown = find_unknown_uris(
                sparql_query, self._vocabulary(), self._resources.documents_dict
            )
            if unknown:
                logger.info("URI validation flagged unknown term(s): %s", list(unknown))
                try:
                    with status("Checking and fixing identifiers..."):
                        sparql_query = self._gen(
                            self._uri_repair_prompt(dialogue, sparql_query, unknown),
                            top_k,
                            grounding,
                        )
                    corrected = True
                except Exception:  # noqa: BLE001 - keep the original query on failure
                    logger.exception("URI repair failed")

        # Schema (domain/range) check: predicate objects outside their range.
        if not cache_hit and top_k is not None and self._config.schema_validation_enabled:
            violations = find_range_violations(sparql_query, self._schema())
            if violations:
                logger.info("schema validation flagged range violation(s): %s", list(violations))
                guidance = summarise_predicate_schema(sparql_query, self._schema())
                try:
                    with status("Checking predicate usage..."):
                        sparql_query = self._gen(
                            self._range_repair_prompt(dialogue, sparql_query, violations, guidance),
                            top_k,
                            grounding,
                        )
                    corrected = True
                except Exception:  # noqa: BLE001 - keep the original query on failure
                    logger.exception("schema repair failed")

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
