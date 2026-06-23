"""Validation of LLM-generated SPARQL against the OEKG vocabulary.

Two cheap, offline guards that run before a query is sent to the endpoint:

- :func:`build_known_vocabulary` collects every controlled-vocabulary term the
  model is allowed to use (the retrieval ids, the domain/range URIs of every
  relation, and any CURIE that appears in the worked examples of the system
  prompt).
- :func:`find_unknown_uris` extracts the ``oeo:``/``obo:`` CURIEs a generated
  query uses and reports the ones outside that vocabulary, together with the
  closest valid suggestion -- the typical symptom of a hallucinated identifier.

``oekg:`` is intentionally *not* validated: that prefix namespaces concrete
instance identifiers (scenario/bundle UUIDs) which are not part of the closed
vocabulary, so flagging them would produce false positives.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Iterable

# CURIEs for any prefix we know about (used when harvesting prompt/vocab terms).
_CURIE_RE = re.compile(r"\b(?:oeo|obo|oekg|dc):[A-Za-z0-9_\-]+")
# String literals, so a colon inside FILTER CONTAINS("a:b") is never matched.
_STRING_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
# Only these prefixes are validated against the closed vocabulary.
_VALIDATED_PREFIXES = ("oeo:", "obo:")


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
    """Map each unknown ``oeo:``/``obo:`` CURIE to ``(suggestion, label)``.

    ``suggestion`` is the closest known CURIE (or ``None``) and ``label`` is its
    human-readable label when available, so the caller can build a helpful
    "did you mean ...?" repair hint.
    """
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
