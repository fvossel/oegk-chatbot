"""Loading of the static resources the RAG pipeline depends on.

Bundles the FAISS index, ontology documents, id mapping and system prompts
into a single :class:`Resources` value object so they can be passed around
explicitly instead of as a loose tuple. The loader also validates that the
index and the id list are consistent, failing fast on drift instead of
producing silently wrong retrievals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from faiss import Index, read_index

from oekg.config import AppConfig, get_config


@dataclass(frozen=True)
class Resources:
    """Everything the pipeline needs that is loaded once at start-up."""

    faiss_index: Index
    documents_dict: dict[str, Any]
    ids: list[str]
    sparql_system_prompt: str
    summary_system_prompt: str
    # URIs that are relations/properties (as opposed to classes); always
    # injected into the generation context regardless of retrieval ranking.
    relation_ids: frozenset[str] = field(default_factory=frozenset)


def _read_text(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate(index: Index, ids: list[str], documents_dict: dict[str, Any]) -> None:
    if index.ntotal != len(ids):
        raise ValueError(
            f"FAISS index size ({index.ntotal}) does not match ids.json ({len(ids)}). "
            "Rebuild the index with `python -m oekg.build_index`."
        )
    missing = [i for i in ids if i not in documents_dict]
    if missing:
        raise ValueError(
            f"ids.json references {len(missing)} id(s) absent from "
            f"classes/relations, e.g. {missing[:3]}."
        )


def load_resources(config: AppConfig | None = None) -> Resources:
    """Load the FAISS index, ontology documents and system prompts."""
    config = config or get_config()

    classes_list = _read_json(config.classes_path)
    relations_list = _read_json(config.relations_path)
    ids = _read_json(config.ids_path)

    documents_dict: dict[str, Any] = {doc["class"]: doc for doc in classes_list}
    documents_dict.update({rel["uri"]: rel for rel in relations_list})
    relation_ids = frozenset(rel["uri"] for rel in relations_list)

    index = read_index(str(config.faiss_index_path))
    _validate(index, ids, documents_dict)

    return Resources(
        faiss_index=index,
        documents_dict=documents_dict,
        ids=ids,
        sparql_system_prompt=_read_text(config.sparql_prompt_path),
        summary_system_prompt=_read_text(config.summary_prompt_path),
        relation_ids=relation_ids,
    )
