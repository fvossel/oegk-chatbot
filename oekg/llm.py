"""OpenAI-backed language-model operations.

Wraps the LLM tasks of the pipeline -- generating a SPARQL query from a
natural-language request, summarising the query results (optionally streamed),
and explaining a query in plain language -- behind a single client so the model
names and prompt assembly live in one place.

The large static system prompt is always the leading content of every request,
so OpenAI's automatic prompt caching applies to it; an optional
``prompt_cache_key`` (passed via ``extra_body`` for SDK-version safety) improves
cache routing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from openai import OpenAI

from oekg.config import AppConfig, get_config
from oekg.resources import Resources

logger = logging.getLogger(__name__)

_EXPLAIN_PROMPT = (
    "You explain SPARQL queries in plain language for an energy researcher who "
    "may not know SPARQL. In 2-4 short sentences, describe what the query "
    "retrieves and the main conditions it applies. Do not repeat the query and "
    "do not include any code."
)


def _compact_documents(documents: list[Any]) -> list[dict[str, Any]]:
    """Shrink ontology docs for the prompt.

    Keeps ``uri``/``label``/``description`` but collapses the (potentially huge)
    nested ``domain``/``range`` lists of ``{uri, label}`` objects down to plain
    label lists -- the URIs there are noise for generation and bloat the prompt.
    """
    compact: list[dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        item: dict[str, Any] = {
            "uri": doc.get("uri") or doc.get("class"),
            "label": doc.get("label"),
        }
        if doc.get("description"):
            item["description"] = doc["description"]
        for key in ("domain", "range"):
            entries = doc.get(key)
            if entries:
                item[key] = [
                    e.get("label", e.get("uri")) for e in entries if isinstance(e, dict)
                ]
        compact.append(item)
    return compact


class LLMClient:
    """Thin wrapper around the OpenAI Responses API for this app's tasks."""

    def __init__(self, client: OpenAI, resources: Resources, config: AppConfig | None = None) -> None:
        self._client = client
        self._resources = resources
        self._config = config or get_config()

    def _extra_body(self) -> dict[str, Any] | None:
        """Pass a prompt-cache routing key when configured (SDK-version safe)."""
        key = self._config.prompt_cache_key
        return {"prompt_cache_key": key} if key else None

    @staticmethod
    def cached_tokens(response: Any) -> int:
        """Safely read the number of cached input tokens from a response."""
        details = getattr(getattr(response, "usage", None), "input_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0)

    def generate_sparql(self, query: str, documents: list, grounding: str | None = None) -> str:
        """Generate a SPARQL query from an NL request and context documents."""
        user_prompt = (
            "Request: "
            + query
            + "\n\nContext with classes and their allowed relations and properties:\n"
            + json.dumps(_compact_documents(documents), ensure_ascii=False, indent=2)
        )
        if grounding:
            user_prompt += "\n\n" + grounding
        response = self._client.responses.create(
            model=self._config.sparql_model,
            input=[
                {"role": "developer", "content": self._resources.sparql_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body=self._extra_body(),
        )
        logger.debug("sparql generation | cached_tokens=%s", self.cached_tokens(response))
        return response.output_text

    def _summary_input(self, sparql_results: str, nl_query: str) -> list[Any]:
        return [
            {"role": "developer", "content": self._resources.summary_system_prompt},
            {
                "role": "user",
                "content": "The users question:"
                + nl_query
                + "\nThe formatted results from the SPARQL query:"
                + sparql_results,
            },
        ]

    def summarise(self, sparql_results: str, nl_query: str) -> str:
        """Summarise SPARQL results in natural language (best effort)."""
        try:
            response = self._client.responses.create(
                model=self._config.summary_model,
                input=self._summary_input(sparql_results, nl_query),
                extra_body=self._extra_body(),
            )
            return response.output_text
        except Exception:  # noqa: BLE001 - summary is optional
            logger.exception("Result summarisation failed")
            return ""

    def summarise_stream(self, sparql_results: str, nl_query: str) -> Iterator[str]:
        """Yield the summary in text chunks via the Responses streaming API."""
        try:
            stream = self._client.responses.create(
                model=self._config.summary_model,
                input=self._summary_input(sparql_results, nl_query),
                extra_body=self._extra_body(),
                stream=True,
            )
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    yield getattr(event, "delta", "")
        except Exception:  # noqa: BLE001 - summary is optional
            logger.exception("Streaming summarisation failed")

    def explain_sparql(self, query: str) -> str:
        """Explain a SPARQL query in plain language (best effort)."""
        try:
            response = self._client.responses.create(
                model=self._config.summary_model,
                input=[
                    {"role": "developer", "content": _EXPLAIN_PROMPT},
                    {"role": "user", "content": query},
                ],
                extra_body=self._extra_body(),
            )
            return response.output_text
        except Exception:  # noqa: BLE001 - explanation is optional
            logger.exception("SPARQL explanation failed")
            return ""
