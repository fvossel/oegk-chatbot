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
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence, Union

# Definitions are long; the first sentences carry the meaning and keep the
# prompt, and the choice list shown to the user, readable.
_DEFINITION_CHARS = 320
_ASK_CHARS = 240
_SKIP_CHARS = 60
_GLOSS_CHARS = 120
_QUOTED_LEAD_RE = re.compile(r'^"([^"]*)"')
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class Option:
    """One ontology term the user can pick."""

    uri: str
    label: str
    definition: str
    # Everyday names users call the term by ("scenarios" for scenario factsheet),
    # as data in classes.json and in English only: the models map other
    # languages and spellings onto them, code never does.
    aliases: tuple[str, ...] = ()
    # What picking this option gets the user, in plain words and in the user's
    # language; written by the model per question, empty when unavailable.
    gloss: str = ""


# A pick is one option, or several when the user wants all of them.
Choice = Union[Option, Sequence[Option]]


def as_options(choice: Choice) -> tuple[Option, ...]:
    """The pick as a tuple of options, whatever form it came in."""
    return (choice,) if isinstance(choice, Option) else tuple(choice)


@dataclass(frozen=True)
class Clarification:
    """The choice put to the user: the options and the words around them.

    ``ask``, ``skip`` and ``all`` are written by the model in the language the
    question was asked in; empty means the caller falls back to its own wording.
    """

    options: tuple[Option, ...]
    ask: str = ""
    skip: str = ""
    all: str = ""


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
        aliases = doc.get("aliases") or ()
        options.append(
            Option(
                uri=str(doc["class"]),
                label=label,
                definition=_shorten(_tidy(str(doc.get("description") or ""), label)),
                aliases=tuple(str(alias) for alias in aliases if str(alias).strip()),
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
    lines = [
        f"C{i}: {option.label}{_also_called(option)} -- {option.definition}"
        for i, option in enumerate(candidates, 1)
    ]
    return (
        "Conversation (the last line is the question to judge):\n"
        f"{dialogue}\n\n"
        "Candidate ontology terms:\n" + "\n".join(lines)
    )


def _also_called(option: Option) -> str:
    return f" (also called: {', '.join(option.aliases)})" if option.aliases else ""


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
    by_id = {f"C{i}": option for i, option in enumerate(candidates, 1)}
    # The model first maps the question's key noun onto a candidate's name, in
    # whatever language or spelling it was asked; a match means it is settled,
    # even if the model then contradicts itself.
    if str(data.get("named")).strip().upper() in by_id:
        return None
    named = data.get("options")
    if not isinstance(named, list):
        return None

    chosen: list[Option] = []
    for ref in named:
        # An option is "C1", or {"id": "C1", "gloss": "..."} with a plain-words gloss.
        ref_id, gloss = (ref.get("id"), ref.get("gloss")) if isinstance(ref, dict) else (ref, None)
        option = by_id.get(str(ref_id).strip().upper())
        if option is not None and all(option.uri != seen.uri for seen in chosen):
            chosen.append(replace(option, gloss=_phrase(gloss, _GLOSS_CHARS)))
    if len(chosen) < 2:
        return None
    return Clarification(
        options=tuple(chosen[:max_options]),
        ask=_phrase(data.get("ask"), _ASK_CHARS),
        skip=_phrase(data.get("skip"), _SKIP_CHARS),
        all=_phrase(data.get("all"), _SKIP_CHARS),
    )


def _phrase(value: Any, limit: int) -> str:
    """A short single line of model-written text, or "" if it is not usable."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else ""


def _stem(word: str) -> str:
    """Fold a plural onto its singular ("factsheets" -> "factsheet", "studies" -> "study")."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _raw_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9äöüß]+", text.lower())


def _words(text: str) -> list[str]:
    return [_stem(word) for word in _raw_words(text)]


@dataclass(frozen=True)
class Mention:
    """A term the question names, and the words it used for it (as written, lower-cased)."""

    option: Option
    words: str
    by_alias: bool


def term_mentions(question: str, terms: Iterable[Option]) -> list[Mention]:
    """The terms the question names by their label or by one of their aliases.

    Matching is on whole words, case-insensitive and plural-tolerant, and the
    longest name wins: "scenario bundles" names the scenario bundle, not also the
    term whose alias is "scenario", and "study reports" names the study report,
    not the term whose alias is "study". Each word belongs to at most one term.
    This is only the cheap exact pass; spelling variants, typos and other
    languages are left to the model's ambiguity check, which sees the same names.
    """
    raw = _raw_words(question)
    tokens = [_stem(word) for word in raw]
    names = [
        (_words(name), option, name != option.label)
        for option in terms
        for name in (option.label, *option.aliases)
    ]
    names = [entry for entry in names if entry[0]]
    names.sort(key=lambda entry: (-len(entry[0]), -len(" ".join(entry[0]))))
    taken = [False] * len(tokens)
    found: list[tuple[int, Mention]] = []
    for words, option, by_alias in names:
        width = len(words)
        for start in range(len(tokens) - width + 1):
            span = range(start, start + width)
            if tokens[start : start + width] == words and not any(taken[i] for i in span):
                for i in span:
                    taken[i] = True
                surface = " ".join(raw[start : start + width])
                found.append((start, Mention(option, surface, by_alias)))
    return [mention for _, mention in sorted(found, key=lambda item: item[0])]


def names_a_term(
    question: str, candidates: Iterable[Option], vocabulary: Iterable[Option] = ()
) -> bool:
    """Whether the question already names one candidate term, by label or alias.

    "Which framework factsheets exist?" says which term it means; asking would be
    noise. ``vocabulary`` holds every known term, so that a longer name the
    question uses ("study report") is not misread as a candidate's alias
    ("study"). This is deliberately checked before the model is asked at all: it
    is cheaper, and a model told to ask only when unsure still sometimes asks
    about sibling terms.
    """
    candidates = list(candidates)
    uris = {option.uri for option in candidates}
    terms = {option.uri: option for option in [*vocabulary, *candidates]}
    return any(mention.option.uri in uris for mention in term_mentions(question, terms.values()))


def clarified_question(question: str, choice: Choice) -> str:
    """The question with the user's choice attached.

    This is also what the shared question cache is keyed on: a saved answer to
    "How many scenarios are there?" is only valid for the term it was asked
    about, so the readings must not share an entry.
    """
    return f"{question} ({', '.join(option.label for option in as_options(choice))})"


def clarification_hint(choice: Choice) -> str:
    """The block injected into generation so the chosen term(s) are used as defined."""
    options = as_options(choice)
    lines = "\n".join(f"- {option.label} ({option.uri}): {option.definition}" for option in options)
    if len(options) == 1:
        return (
            "The user was asked which ontology term the request refers to and chose:\n"
            f"{lines}\n"
            "Build the query around exactly this class, not a similar one."
        )
    return (
        "The user was asked which ontology term the request refers to and chose all of these:\n"
        f"{lines}\n"
        "Answer for each of these classes separately (for example one count or one list "
        "per class, labelled with the class), not for a merged set."
    )
