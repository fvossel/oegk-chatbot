"""Chat-model operations over an OpenAI-compatible Chat Completions endpoint.

Wraps the LLM tasks of the pipeline -- generating a SPARQL query from a
natural-language request, summarising the query results (optionally streamed),
and explaining a query in plain language -- behind a single client so the model
names and prompt assembly live in one place.

The endpoint defaults to Osnabrück University's LiteLLM gateway, the same service
that computes the embeddings. Any Chat Completions endpoint works instead. Chat
Completions is used rather than the Responses API because it is the surface
OpenAI-compatible servers actually implement fully, and every prompt uses the
``system`` role because some of them reject ``developer``.

Reasoning ("thinking") is requested only where it pays for itself. Generating a
SPARQL query is the one genuinely hard step, so it keeps thinking; summarising a
result table, explaining a query and checking a question for ambiguity are short,
mechanical tasks where it only adds latency (about 0.3 s without, tens of seconds
with). The switch travels in ``extra_body``, which the gateway's Qwen honours; a
server that does not know it ignores the key, and an endpoint that bakes
reasoning into the model id (a LibreChat agents API does) needs one model per
behaviour instead. Set ``OEKG_LLM_NO_THINK=0`` for a server that rejects unknown
parameters outright.

``strip_reasoning`` below is a safety net: should an endpoint emit ``<think>``
blocks inline instead of in a separate field, they never reach the user.

The large static system prompt is always the leading message of every request,
so whatever prefix caching the server implements applies to it; an optional
``prompt_cache_key`` (passed via ``extra_body``, which servers that do not know
it ignore) improves cache routing where it is supported.
"""

from __future__ import annotations

import json
import logging
import re
from itertools import chain
from typing import Any, Iterable, Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from oekg.config import AppConfig, get_config
from oekg.disambiguation import Clarification, Option, build_user_prompt, parse_clarification
from oekg.resources import Resources

logger = logging.getLogger(__name__)

# Left implicit, a model tends to settle on one language (an earlier endpoint
# always answered in German), so the rule names the English case as well as the
# German one.
_LANGUAGE_RULE = (
    "the language the user wrote in (English for an English question, German for a "
    "German question, and likewise for any other language)"
)

_AMBIGUITY_PROMPT = (
    "You check whether a question about the Open Energy Knowledge Graph is ambiguous "
    "between several ontology terms. You get the conversation and numbered candidate "
    "terms with their definitions.\n\n"
    "The question IS ambiguous only if it could refer to two or more candidates AND the "
    "answer would materially differ depending on which one is meant (different counts, "
    "different lists).\n"
    "It is NOT ambiguous if: it names or describes one candidate unmistakably; the "
    "conversation already settled which term is meant; only one candidate fits; it is "
    "not a data question (a greeting, thanks, a question about the bot or about the "
    "knowledge graph as a whole rather than one kind of thing); or the "
    "candidates merely share a topic without being alternatives for the same question.\n"
    "A candidate whose label is the question's key noun (or its plural, or a translation "
    "of it) is the one meant; terms that are merely related to it are not alternatives.\n"
    "Asking costs the user a click and a needless question is an annoyance, so when you "
    "are unsure whether the question is ambiguous, answer {\"ambiguous\": false}.\n\n"
    "Reply with JSON only, no other text: {\"ambiguous\": false} or "
    "{\"ambiguous\": true, \"options\": [\"C1\", \"C3\"], \"language\": \"...\", "
    "\"ask\": \"...\", \"skip\": \"...\"} naming the 2-4 candidates that are the real "
    "alternatives. \"language\" is the language of the user's last question, named in "
    "English; \"ask\" is one short sentence asking the user which of the options they "
    "mean and \"skip\" is a 2-5 word button label meaning \"none of these, you decide\"; "
    "write both in " + _LANGUAGE_RULE + ", i.e. in the language you named."
)

_EXPLAIN_PROMPT = (
    "You explain SPARQL queries in plain language for an energy researcher who "
    "may not know SPARQL. In 2-4 short sentences, describe what the query "
    "retrieves and the main conditions it applies. Do not repeat the query and "
    "do not include any code. Ontology identifiers such as oeo:OEO_00020227 are "
    "opaque: refer to them only by the labels listed with the query, and never "
    "guess what an identifier without a listed label means. If the user's "
    "question is given, write in " + _LANGUAGE_RULE + "; otherwise write English."
)

# Reasoning models (e.g. Qwen) may emit their chain of thought inline when the
# server does not separate it. Left in, it would break every generated query.
# (The default gateway returns it in a separate ``reasoning_content`` field,
# which the SDK keeps out of ``content``; this stays as a safety net.)
_REASONING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Turns the reasoning pass off for a single request. This is the vLLM/Qwen chat
# template switch; a server that does not know it ignores it.
_NO_THINK_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def request_body(config: AppConfig, *, thinking: bool) -> dict[str, Any] | None:
    """The non-standard request body: cache routing and the reasoning switch.

    Shared with the health check so its probe asks for exactly what a short task
    asks for. A probe that left the model thinking would spend its whole token
    budget on reasoning and report an empty answer for a healthy endpoint.
    """
    body: dict[str, Any] = {}
    if config.prompt_cache_key:
        body["prompt_cache_key"] = config.prompt_cache_key
    if not thinking and config.no_think_enabled:
        body.update(_NO_THINK_BODY)
    return body or None


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Remove inline ``<think>...</think>`` blocks from a model reply."""
    return _REASONING_RE.sub("", text).strip()


def _lstrip_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Drop leading whitespace from a chunk stream, however it is split."""
    it = iter(chunks)
    for chunk in it:
        chunk = chunk.lstrip()
        if chunk:
            yield chunk
            yield from it
            return


def strip_reasoning_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Streaming counterpart of :func:`strip_reasoning` for a leading block.

    Reasoning models emit their thinking at the very start of a reply, so only
    the beginning is buffered: as soon as it is clear there is no leading
    ``<think>`` block, the buffer and every further chunk pass straight through
    and the reply keeps streaming live. A block that never closes (a truncated
    reply) is dropped entirely rather than shown.
    """
    buf = ""
    it = iter(chunks)
    for chunk in it:
        buf += chunk
        head = buf.lstrip().lower()
        if _THINK_OPEN.startswith(head):
            continue  # empty so far, or could still become the opening tag
        if not head.startswith(_THINK_OPEN):
            yield buf
            yield from it
            return
        close = buf.lower().find(_THINK_CLOSE)
        if close != -1:
            yield from _lstrip_stream(chain([buf[close + len(_THINK_CLOSE):]], it))
            return
    head = buf.lstrip().lower()
    if head and not head.startswith(_THINK_OPEN):
        yield buf  # a short reply that merely looked like an opening tag


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


def _messages(system: str, user: str) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _deltas(stream: Iterable[Any]) -> Iterator[str]:
    """Yield the text of each streamed chunk, skipping empty and usage-only ones."""
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class LLMClient:
    """Thin wrapper around the Chat Completions API for this app's tasks."""

    def __init__(self, client: OpenAI, resources: Resources, config: AppConfig | None = None) -> None:
        self._client = client
        self._resources = resources
        self._config = config or get_config()

    def _extra_body(self, *, thinking: bool) -> dict[str, Any] | None:
        """Build the non-standard request body: cache routing and the think switch."""
        return request_body(self._config, thinking=thinking)

    @staticmethod
    def cached_tokens(response: Any) -> int:
        """Safely read the number of cached prompt tokens from a response."""
        details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0)

    def _complete(self, model: str, system: str, user: str, *, thinking: bool) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=_messages(system, user),
            extra_body=self._extra_body(thinking=thinking),
        )
        logger.debug("completion | model=%s | cached_tokens=%s", model, self.cached_tokens(response))
        return strip_reasoning(response.choices[0].message.content or "")

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
        # The one step worth thinking about: it has to pick the right classes,
        # relations and shape from the retrieved context.
        return self._complete(
            self._config.sparql_model,
            self._resources.sparql_system_prompt,
            user_prompt,
            thinking=True,
        )

    @staticmethod
    def _summary_user_prompt(sparql_results: str, nl_query: str) -> str:
        return (
            "The users question:"
            + nl_query
            + "\nThe formatted results from the SPARQL query:"
            + sparql_results
        )

    def summarise(self, sparql_results: str, nl_query: str) -> str:
        """Summarise SPARQL results in natural language (best effort)."""
        try:
            return self._complete(
                self._config.summary_model,
                self._resources.summary_system_prompt,
                self._summary_user_prompt(sparql_results, nl_query),
                thinking=False,
            )
        except Exception:  # noqa: BLE001 - summary is optional
            logger.exception("Result summarisation failed")
            return ""

    def summarise_stream(self, sparql_results: str, nl_query: str) -> Iterator[str]:
        """Yield the summary in text chunks via the streaming Chat Completions API."""
        try:
            stream = self._client.chat.completions.create(
                model=self._config.summary_model,
                messages=_messages(
                    self._resources.summary_system_prompt,
                    self._summary_user_prompt(sparql_results, nl_query),
                ),
                extra_body=self._extra_body(thinking=False),
                stream=True,
            )
            yield from strip_reasoning_stream(_deltas(stream))
        except Exception:  # noqa: BLE001 - summary is optional
            logger.exception("Streaming summarisation failed")

    def find_ambiguity(
        self, dialogue: str, candidates: list[Option]
    ) -> Clarification | None:
        """The choice to put to the user, or ``None`` when the question is clear.

        Best effort: a failure here must never stop a question from being
        answered, so any error means "not ambiguous".
        """
        if len(candidates) < 2:
            return None
        try:
            reply = self._complete(
                self._config.summary_model,
                _AMBIGUITY_PROMPT,
                build_user_prompt(dialogue, candidates),
                thinking=False,
            )
        except Exception:  # noqa: BLE001 - asking is optional
            logger.exception("ambiguity check failed")
            return None
        return parse_clarification(
            reply, candidates, max_options=self._config.disambiguation_max_options
        )

    def explain_sparql(
        self, query: str, term_labels: dict[str, str] | None = None, question: str = ""
    ) -> str:
        """Explain a SPARQL query in plain language (best effort).

        ``term_labels`` maps the query's ontology identifiers to their labels.
        Without them a model has to guess what an opaque id like OEO_00020227
        stands for -- and some models guess confidently and wrongly.
        """
        user_prompt = query
        if question:
            user_prompt = f"The user's question: {question}\n\n" + user_prompt
        if term_labels:
            user_prompt += "\n\nLabels of the ontology identifiers used in this query:\n" + "\n".join(
                f"- {term}: {label}" for term, label in term_labels.items()
            )
        try:
            return self._complete(
                self._config.summary_model, _EXPLAIN_PROMPT, user_prompt, thinking=False
            )
        except Exception:  # noqa: BLE001 - explanation is optional
            logger.exception("SPARQL explanation failed")
            return ""
