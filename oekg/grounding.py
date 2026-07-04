"""Entity/label grounding before SPARQL generation (feature #1).

The model's most common "looks right but returns nothing" failure is filtering
on a label string that does not exactly exist in the graph (e.g. it writes
``"MESSAGEix"`` when the real label is ``"MESSAGEix v1.2"``). This module, in
one cheap SPARQL round-trip, resolves the entity mentions in a question against
the graph's real ``rdfs:label`` / ``dc:acronym`` strings and renders a compact
"verified labels" block to inject into the generation prompt.

All helpers are pure and network-free except :func:`ground_question`, which
takes the endpoint call as an injected ``run_query`` callable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# Capitalised / acronym-like proper-noun spans (medium-confidence mentions).
_PROPER_RE = re.compile(r"[A-Z0-9][\w&.\-]*(?:\s+[A-Z0-9][\w&.\-]*)*")
# Quoted spans (high-confidence: the user explicitly delimited a name).
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'|“([^”]+)”")
_YEAR_RE = re.compile(r"^\d{4}$")

# Capitalised-but-generic words that are not entity names.
_STOPWORDS = {
    "give", "show", "list", "find", "get", "what", "which", "who", "where",
    "when", "how", "all", "me", "the", "a", "an", "and", "or", "for", "in",
    "of", "to", "is", "are", "please", "tell", "about", "with", "that", "do",
    "scenario", "scenarios", "bundle", "bundles", "dataset", "datasets",
    "study", "studies", "year", "years",
}


@dataclass(frozen=True)
class GroundedEntity:
    """A mention resolved to a real label/URI in the graph."""

    mention: str
    label: str
    acronym: str
    uri: str
    types: tuple[str, ...]


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def extract_mentions(question: str, *, max_mentions: int = 6, min_len: int = 3) -> list[str]:
    """Deterministically pull candidate entity mentions from a question."""
    quoted: list[str] = []
    for match in _QUOTED_RE.finditer(question):
        inner = next((g for g in match.groups() if g), "").strip()
        if inner:
            quoted.append(inner)

    remainder = _QUOTED_RE.sub(" ", question)
    proper = [m.group(0).strip() for m in _PROPER_RE.finditer(remainder)]

    ordered: list[str] = []
    seen: set[str] = set()
    for mention in quoted + proper:  # quoted first (higher confidence)
        mention = " ".join(mention.split())
        if len(mention) < min_len:
            continue
        if _YEAR_RE.match(mention):
            continue
        # Drop mentions that are entirely stopwords (e.g. "All", "Give Me").
        if all(tok.lower() in _STOPWORDS for tok in mention.split()):
            continue
        key = mention.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(mention)

    return ordered[:max_mentions]


def build_grounding_query(mentions: list[str], *, limit_per_mention: int = 5) -> str | None:
    """Build ONE SPARQL query resolving all mentions (or None if no mentions)."""
    if not mentions:
        return None
    needles = " ".join(f'"{_escape_literal(m.lower())}"' for m in mentions)
    limit = max(1, limit_per_mention) * len(mentions)
    return (
        "SELECT DISTINCT ?needle ?s ?label ?acronym "
        '(GROUP_CONCAT(DISTINCT STR(?type);separator="|") AS ?types) WHERE {\n'
        f"  VALUES ?needle {{ {needles} }}\n"
        "  ?s rdfs:label ?label .\n"
        "  OPTIONAL { ?s dc:acronym ?acronym }\n"
        "  OPTIONAL { ?s rdf:type ?type }\n"
        "  FILTER( CONTAINS(LCASE(STR(?label)), ?needle) || "
        "(BOUND(?acronym) && CONTAINS(LCASE(STR(?acronym)), ?needle)) )\n"
        "} GROUP BY ?needle ?s ?label ?acronym "
        f"LIMIT {limit}"
    )


def _short_type(uri: str) -> str:
    return re.split(r"[/#]", uri.rstrip("/#"))[-1] if uri else uri


def parse_grounding_results(
    raw: dict[str, Any], mentions: list[str], *, limit_per_mention: int = 5
) -> list[GroundedEntity]:
    """Turn endpoint bindings into a deduped, mention-ordered entity list."""
    bindings = (raw or {}).get("results", {}).get("bindings", [])
    needle_to_mention = {m.lower(): m for m in mentions}

    grouped: dict[str, list[GroundedEntity]] = {}
    seen: set[tuple[str, str]] = set()
    for binding in bindings:
        label = binding.get("label", {}).get("value")
        if not label:
            continue
        needle = binding.get("needle", {}).get("value", "").lower()
        mention = needle_to_mention.get(needle, needle)
        uri = binding.get("s", {}).get("value", "")
        dedupe_key = (mention, label)
        if dedupe_key in seen:
            continue
        bucket = grouped.setdefault(mention, [])
        if len(bucket) >= limit_per_mention:
            continue
        seen.add(dedupe_key)
        types_raw = binding.get("types", {}).get("value", "")
        types = tuple(_short_type(t) for t in types_raw.split("|") if t)
        bucket.append(
            GroundedEntity(
                mention=mention,
                label=label,
                acronym=binding.get("acronym", {}).get("value", ""),
                uri=uri,
                types=types,
            )
        )

    entities: list[GroundedEntity] = []
    for mention in mentions:  # stable: input mention order
        entities.extend(grouped.get(mention, []))
    return entities


def render_grounding_prompt(entities: list[GroundedEntity]) -> str:
    """Render the verified-labels block injected into the generation prompt."""
    if not entities:
        return ""
    lines = [
        "### Verified entity labels from the knowledge graph",
        "These are candidate matches found in the graph. Prefer these EXACT label "
        "strings in rdfs:label / dc:acronym filters. If none fits the user's intent, "
        "fall back to a CONTAINS filter on the user's wording.",
    ]
    for entity in entities:
        parts = [f'- for "{entity.mention}": label "{entity.label}"']
        if entity.acronym:
            parts.append(f'acronym "{entity.acronym}"')
        if entity.types:
            parts.append(f"types: {', '.join(entity.types)}")
        line = ", ".join(parts)
        if entity.uri:
            line += f"  <{entity.uri}>"
        lines.append(line)
    return "\n".join(lines)


def ground_question(
    question: str,
    run_query: Callable[[str], dict[str, Any]],
    *,
    max_mentions: int = 6,
    min_len: int = 3,
    limit_per_mention: int = 5,
) -> str:
    """Resolve entity mentions and return a prompt fragment ("" on any miss)."""
    try:
        mentions = extract_mentions(question, max_mentions=max_mentions, min_len=min_len)
        query = build_grounding_query(mentions, limit_per_mention=limit_per_mention)
        if query is None:
            return ""
        raw = run_query(query)
        entities = parse_grounding_results(raw, mentions, limit_per_mention=limit_per_mention)
        return render_grounding_prompt(entities)
    except Exception:  # noqa: BLE001 - grounding must never break a request
        return ""
