"""Offline guards over LLM-generated SPARQL.

All helpers here are pure and network-free:

- :func:`normalise_sparql` strips stray markdown fences / labels from raw output.
- :func:`validate_syntax` parses a prefixed query locally with rdflib (advisory).
- :func:`build_known_vocabulary` / :func:`find_unknown_uris` check that the
  ``oeo:``/``obo:`` CURIEs used actually exist in the vocabulary.
- :func:`build_relation_schema` / :func:`find_range_violations` /
  :func:`summarise_predicate_schema` add schema (domain/range) awareness.
- :func:`extract_literal_mentions` pulls the string literals a query filters on
  (used to fetch real near-miss labels on empty results).

``oekg:`` is intentionally never validated as vocabulary: that prefix namespaces
concrete instance identifiers (scenario/bundle UUIDs), not closed terms.
"""

from __future__ import annotations

import contextlib
import difflib
import io
import re
import warnings
from typing import Any, Iterable

# CURIEs for any prefix we know about (used when harvesting prompt/vocab terms).
_CURIE_RE = re.compile(r"\b(?:oeo|obo|oekg|dc):[A-Za-z0-9_\-]+")
# String literals, so a colon inside FILTER CONTAINS("a:b") is never matched.
_STRING_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
# Only these prefixes are validated against the closed vocabulary.
_VALIDATED_PREFIXES = ("oeo:", "obo:")

# A direct triple of the form ``<pred-curie> <object-curie> .`` (object in a
# real object position, terminated by . ; or }). Used for range validation.
# The predicate must follow a real subject position (a variable, ``<iri>``,
# CURIE subject, or a ``;`` predicate-list continuation) so property paths
# (``oeo:a/oeo:b``) and ``VALUES`` lists are not mistaken for triples.
_DIRECT_TRIPLE_RE = re.compile(
    r"(?:[?$]\w+|<[^>]+>|;|(?:oeo|obo|oekg|dc):[A-Za-z0-9_]+)\s+"
    r"((?:oeo|obo|oekg|dc):[A-Za-z0-9_]+)\s+"
    r"((?:oeo|obo|oekg|dc):[A-Za-z0-9_]+)\s*(?=[.;}])"
)

# Markdown fences / leading labels the model may emit despite instructions.
_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```\s*$")
_LABEL_RE = re.compile(r"^(?:SPARQL|Query)\s*:\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Output normalisation + local syntax validation (feature #3)
# ---------------------------------------------------------------------------


def normalise_sparql(text: str) -> str:
    """Strip markdown fences and a leading ``SPARQL:``/``Query:`` label.

    Deliberately does NOT require a leading query keyword, so a
    ``<bot-information>`` reply passes through unchanged.
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("```"):
        cleaned = _FENCE_OPEN_RE.sub("", cleaned)
        cleaned = _FENCE_CLOSE_RE.sub("", cleaned)
        cleaned = cleaned.strip()
    cleaned = _LABEL_RE.sub("", cleaned)
    return cleaned.strip()


def validate_syntax(prefixed_query: str) -> str | None:
    """Return a parser error string if the (prefixed) query is invalid, else None.

    Advisory only: rdflib may reject queries the OEP would accept, so callers
    must never hard-block on a non-None result. Returns ``None`` (no-op) if
    rdflib is not installed. The query MUST already carry its PREFIX block, or
    every ``oeo:``/``obo:`` term raises an "unknown prefix" error.
    """
    try:
        from rdflib.plugins.sparql import prepareQuery
    except ImportError:
        return None

    # Some literals (e.g. xsd:dateTime) make rdflib's parser warn to stderr;
    # keep it quiet so logs stay clean.
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            prepareQuery(prefixed_query)
        return None
    except Exception as exc:  # noqa: BLE001 - any parse failure is advisory
        return str(exc)


# ---------------------------------------------------------------------------
# Vocabulary existence check (feature: URI validation)
# ---------------------------------------------------------------------------


def build_known_vocabulary(resources: Any) -> set[str]:
    """Return every controlled-vocabulary CURIE the model may legitimately use."""
    known: set[str] = set(resources.ids)
    for doc in resources.documents_dict.values():
        if not isinstance(doc, dict):
            continue
        for key in ("domain", "range"):
            for entry in doc.get(key, []) or []:
                uri = entry.get("uri") if isinstance(entry, dict) else None
                if uri:
                    known.add(uri)
    known.update(_CURIE_RE.findall(resources.sparql_system_prompt))
    return known


def find_unknown_uris(
    query: str,
    known: Iterable[str],
    documents: dict | None = None,
    max_suggestions: int = 1,
) -> dict[str, tuple[str | None, str | None]]:
    """Map each unknown ``oeo:``/``obo:`` CURIE to ``(suggestion, label)``."""
    known_set = set(known)
    candidates = [k for k in known_set if k.startswith(_VALIDATED_PREFIXES)]
    stripped = _STRING_RE.sub('""', query)

    unknown: dict[str, tuple[str | None, str | None]] = {}
    for match in _CURIE_RE.finditer(stripped):
        term = match.group(0)
        if not term.startswith(_VALIDATED_PREFIXES):
            continue
        if term in known_set or term in unknown:
            continue
        close = difflib.get_close_matches(term, candidates, n=max_suggestions, cutoff=0.7)
        suggestion = close[0] if close else None
        label = None
        if suggestion and documents:
            doc = documents.get(suggestion)
            if isinstance(doc, dict):
                label = doc.get("label")
        unknown[term] = (suggestion, label)
    return unknown


# ---------------------------------------------------------------------------
# Schema (domain/range) awareness (feature #4)
# ---------------------------------------------------------------------------


def build_relation_schema(resources: Any) -> dict[str, dict[str, Any]]:
    """Index relation predicates -> their declared domain/range.

    ``closed`` is True only when the range is a non-empty list of class CURIEs
    with no ``Literal`` sentinel; those are the predicates whose direct object
    we can validate offline without subclass closure.
    """
    pred_uris = set(getattr(resources, "relation_ids", set()))
    if not pred_uris:
        pred_uris = {
            k for k, v in resources.documents_dict.items()
            if isinstance(v, dict) and "range" in v
        }

    schema: dict[str, dict[str, Any]] = {}
    for uri in pred_uris:
        doc = resources.documents_dict.get(uri)
        if not isinstance(doc, dict):
            continue
        range_entries = [e for e in doc.get("range", []) or [] if isinstance(e, dict)]
        has_literal = any(e.get("uri") == "Literal" for e in range_entries)
        range_pairs = [
            (e["uri"], e.get("label") or e["uri"])
            for e in range_entries
            if e.get("uri") and e["uri"] != "Literal"
        ]
        domain = [
            e.get("label") or e.get("uri")
            for e in doc.get("domain", []) or []
            if isinstance(e, dict)
        ]
        schema[uri] = {
            "label": doc.get("label") or uri,
            "domain": domain,
            "range_uris": frozenset(u for u, _ in range_pairs),
            "range_labels": [lbl for _, lbl in range_pairs],
            "range_label_map": dict(range_pairs),
            "closed": bool(range_pairs) and not has_literal,
        }
    return schema


def find_range_violations(
    query: str, schema: dict[str, dict[str, Any]], max_suggestions: int = 1
) -> dict[str, dict[str, Any]]:
    """Flag direct ``?s PRED oeo:CLASS .`` objects outside PRED's closed range."""
    stripped = _STRING_RE.sub('""', query)
    violations: dict[str, dict[str, Any]] = {}
    for match in _DIRECT_TRIPLE_RE.finditer(stripped):
        pred, obj = match.group(1), match.group(2)
        info = schema.get(pred)
        if not info or not info["closed"]:
            continue
        if not obj.startswith(_VALIDATED_PREFIXES):
            continue
        if obj in info["range_uris"] or pred in violations:
            continue
        close = difflib.get_close_matches(
            obj, list(info["range_uris"]), n=max_suggestions, cutoff=0.6
        )
        suggestion = close[0] if close else None
        violations[pred] = {
            "object": obj,
            "pred_label": info["label"],
            "range_labels": info["range_labels"],
            "suggestion": (suggestion, info["range_label_map"].get(suggestion) if suggestion else None),
        }
    return violations


def summarise_predicate_schema(
    query: str, schema: dict[str, dict[str, Any]], max_range_labels: int = 40
) -> dict[str, dict[str, Any]]:
    """Return domain/range labels for every known predicate used in the query."""
    stripped = _STRING_RE.sub('""', query)
    used: dict[str, dict[str, Any]] = {}
    for match in _CURIE_RE.finditer(stripped):
        curie = match.group(0)
        info = schema.get(curie)
        if info is None or curie in used:
            continue
        used[curie] = {
            "label": info["label"],
            "domain": info["domain"],
            "range_labels": info["range_labels"][:max_range_labels],
        }
    return used


# ---------------------------------------------------------------------------
# Literal extraction for near-miss repair (feature #5)
# ---------------------------------------------------------------------------


def extract_literal_mentions(query: str) -> list[str]:
    """Return the de-quoted contents of every string literal in the query."""
    mentions: list[str] = []
    seen: set[str] = set()
    for match in _STRING_RE.finditer(query):
        inner = match.group(0)[1:-1].strip()
        if not inner or inner in seen:
            continue
        seen.add(inner)
        mentions.append(inner)
    return mentions
