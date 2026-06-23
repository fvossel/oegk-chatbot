"""Persistent cache mapping natural-language questions to SPARQL queries.

The cache lets the app:

1. skip the (expensive) LLM SPARQL-generation step for a previously seen
   question, and
2. offer users a list of known questions to re-run directly.

It also stores an optional question embedding per entry so the app can offer
the closest *paraphrase* of a new question (semantic match) when there is no
exact hit -- the user confirms before such a match is used.

It is intentionally simple and dependency-light: a JSON file on disk. Writes are
atomic (tmp file + ``os.replace``) and serialised by an in-process thread lock,
then merged on every write so concurrent threads in one Streamlit process do not
clobber each other. Across *separate* processes the thread lock offers no
protection; since entries are cheaply regenerable that is an accepted trade-off.
Reads are intentionally lock-free and stay consistent thanks to the atomic write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np


def normalise_question(question: str) -> str:
    """Normalise a question into a stable cache key."""
    return " ".join(question.strip().lower().split())


@dataclass(frozen=True)
class CacheEntry:
    """A single cached question/SPARQL pair plus usage metadata."""

    question: str
    sparql: str
    created_at: str
    last_used: str
    hit_count: int
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        data = {
            "question": self.question,
            "sparql": self.sparql,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "hit_count": self.hit_count,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            question=data["question"],
            sparql=data["sparql"],
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used", data.get("created_at", "")),
            hit_count=int(data.get("hit_count", 0)),
            embedding=data.get("embedding"),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class QueryCache:
    """File-backed cache of question -> SPARQL query."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = enabled
        self._lock = Lock()

    # -- internal IO ------------------------------------------------------
    def _load_raw(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            # A corrupt cache must never crash the app; treat it as empty.
            return {}

    def _write_raw(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    # -- public API -------------------------------------------------------
    def get(self, question: str) -> CacheEntry | None:
        """Return the cached entry for ``question`` or ``None``."""
        if not self._enabled:
            return None
        raw = self._load_raw().get(normalise_question(question))
        return CacheEntry.from_dict(raw) if raw else None

    def put(self, question: str, sparql: str, embedding: list[float] | None = None) -> None:
        """Store (or refresh) the SPARQL query generated for ``question``."""
        if not self._enabled or not sparql.strip():
            return
        key = normalise_question(question)
        with self._lock:
            data = self._load_raw()
            existing = data.get(key)
            now = _now()
            data[key] = CacheEntry(
                question=question.strip(),
                sparql=sparql,
                created_at=existing.get("created_at", now) if existing else now,
                last_used=now,
                hit_count=int(existing.get("hit_count", 0)) if existing else 0,
                embedding=embedding
                if embedding is not None
                else (existing.get("embedding") if existing else None),
            ).to_dict()
            self._write_raw(data)

    def record_hit(self, question: str) -> None:
        """Increment usage stats for a cache hit."""
        if not self._enabled:
            return
        key = normalise_question(question)
        with self._lock:
            data = self._load_raw()
            entry = data.get(key)
            if entry:
                entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
                entry["last_used"] = _now()
                self._write_raw(data)

    def list_questions(self) -> list[CacheEntry]:
        """Return cached entries, most-used and most-recent first."""
        if not self._enabled:
            return []
        entries = [CacheEntry.from_dict(v) for v in self._load_raw().values()]
        return sorted(entries, key=lambda e: (e.hit_count, e.last_used), reverse=True)

    def has_embeddings(self) -> bool:
        """Whether any cached entry carries an embedding (for semantic match)."""
        if not self._enabled:
            return False
        return any(v.get("embedding") for v in self._load_raw().values())

    def semantic_match(
        self, embedding: list[float], threshold: float
    ) -> tuple[CacheEntry, float] | None:
        """Return the closest cached entry above ``threshold`` cosine similarity."""
        if not self._enabled or not embedding:
            return None
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return None

        best: CacheEntry | None = None
        best_sim = -1.0
        for raw in self._load_raw().values():
            stored = raw.get("embedding")
            if not stored:
                continue
            vec = np.asarray(stored, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                continue
            sim = float(query @ vec / (query_norm * norm))
            if sim > best_sim:
                best_sim = sim
                best = CacheEntry.from_dict(raw)

        if best is not None and best_sim >= threshold:
            return best, best_sim
        return None

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._write_raw({})
