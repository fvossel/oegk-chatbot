"""Ask which ontology term a question means, when it could mean several.

"How many scenarios are there?" fits both a *scenario factsheet* and a *scenario
bundle* of the OEKG, and the two counts differ (147 vs 55). Guessing silently
gives a confident answer to a question the user may not have asked. Instead the
model is shown the definitions of the best-matching terms and says whether the
question is determinate against them; when it is not, the user picks one in the
chat and generation continues with that definition.

Nothing here is specific to a term. The model judges ambiguity against whatever
definitions retrieval surfaced, and its answer is only ever trusted as a choice
among those candidates: an identifier it invents is dropped, and fewer than two
real options means "not ambiguous". The helpers are pure; the model call lives
in :mod:`oekg.llm`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

# Definitions are long; the first sentences carry the meaning and keep the
# prompt, and the choice list shown to the user, readable.
_DEFINITION_CHARS = 320
_ASK_CHARS = 240
_SKIP_CHARS = 60
_QUOTED_LEAD_RE = re.compile(r'^"([^"]*)"')
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class Option:
    """One ontology term the user can pick."""

    uri: str
    label: str
    definition: str


@dataclass(frozen=True)
class Clarification:
    """The choice put to the user: the options and the words around them.

    ``ask`` and ``skip`` are written by the model in the language the question
    was asked in; empty means the caller falls back to its own wording.
    """

    options: tuple[Option, ...]
    ask: str = ""
    skip: str = ""


def candidate_options(documents: Iterable[Any], limit: int) -> list[Option]:
    """The first ``limit`` *class* documents, as options.

    Retrieval returns ranked classes followed by every relation; relations carry
    a ``uri`` key and no ``class`` key, which is what tells them apart.
    """
    options: list[Option] = []
    for doc in documents:
        if not isinstance(doc, dict) or "class" not in doc:
            continue
        label = str(doc.get("label") or doc["class"])
        options.append(
            Option(
                uri=str(doc["class"]),
                label=label,
                definition=_shorten(_tidy(str(doc.get("description") or ""), label)),
            )
        )
        if len(options) >= limit:
            break
    return options


def _tidy(text: str, label: str) -> str:
    """Drop the ontology's quoting around a definition.

    OEO elucidations come as ``'term' \"The definition.\" More text``; the term is
    already the button label, and the quote marks are noise in a chat.
    """
    text = text.strip()
    if text.lower().startswith(f"'{label.lower()}'"):
        text = text[len(label) + 2 :].lstrip()
    return _QUOTED_LEAD_RE.sub(r"\1", text)


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _DEFINITION_CHARS:
        return text
    cut = text[:_DEFINITION_CHARS].rsplit(" ", 1)[0]
    return cut + " …"


def build_user_prompt(dialogue: str, candidates: list[Option]) -> str:
    """The user message of the ambiguity check; candidates are numbered C1, C2, ..."""
    lines = [f"C{i}: {option.label} -- {option.definition}" for i, option in enumerate(candidates, 1)]
    return (
        "Conversation (the last line is the question to judge):\n"
        f"{dialogue}\n\n"
        "Candidate ontology terms:\n" + "\n".join(lines)
    )


def parse_clarification(
    reply: str, candidates: list[Option], *, max_options: int
) -> Clarification | None:
    """Turn the model's JSON reply into a choice, trusting only real candidates.

    Returns ``None`` unless the model says the question is ambiguous AND names
    at least two distinct candidates. Anything malformed also yields ``None``:
    failing to ask is always safe, asking about something invented is not.
    """
    text = _FENCE_RE.sub("", reply.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("ambiguous") is not True:
        return None
    named = data.get("options")
    if not isinstance(named, list):
        return None

    by_id = {f"C{i}": option for i, option in enumerate(candidates, 1)}
    chosen: list[Option] = []
    for ref in named:
        option = by_id.get(str(ref).strip().upper())
        if option is not None and option not in chosen:
            chosen.append(option)
    if len(chosen) < 2:
        return None
    return Clarification(
        options=tuple(chosen[:max_options]),
        ask=_phrase(data.get("ask"), _ASK_CHARS),
        skip=_phrase(data.get("skip"), _SKIP_CHARS),
    )


def _phrase(value: Any, limit: int) -> str:
    """A short single line of model-written text, or "" if it is not usable."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else ""


def _stem(word: str) -> str:
    """Fold a plural onto its singular ("factsheets" -> "factsheet")."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> str:
    return " ".join(_stem(word) for word in re.findall(r"[a-z0-9]+", text.lower()))


def names_a_term(question: str, candidates: Iterable[Option]) -> bool:
    """Whether the question already uses one candidate term by its own name.

    "Which framework factsheets exist?" says which term it means; asking would be
    noise. Matching is on whole words, case-insensitive, and plural-tolerant. This
    is deliberately checked before the model is asked at all: it is cheaper, and a
    model told to ask only when unsure still sometimes asks about sibling terms.
    """
    haystack = f" {_words(question)} "
    return any(f" {_words(option.label)} " in haystack for option in candidates)


def clarified_question(question: str, option: Option) -> str:
    """The question with the user's choice attached.

    This is also what the shared question cache is keyed on: a saved answer to
    "How many scenarios are there?" is only valid for the term it was asked
    about, so the two readings must not share an entry.
    """
    return f"{question} ({option.label})"


def clarification_hint(option: Option) -> str:
    """The block injected into generation so the chosen term is used as defined."""
    return (
        "The user was asked which ontology term the request refers to and chose:\n"
        f"- {option.label} ({option.uri}): {option.definition}\n"
        "Build the query around exactly this class, not a similar one."
    )
