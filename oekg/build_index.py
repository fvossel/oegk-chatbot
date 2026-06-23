"""Rebuild the FAISS vector store and id list from the ontology JSON files.

Run as ``python -m oekg.build_index``. This embeds every class and relation with
the configured embedding model and writes a fresh ``vector_store.faiss`` and
matching ``ids.json`` (index row *i* corresponds to ``ids[i]``). It is the
canonical, reproducible way to regenerate the index -- and the only safe way to
switch ``OEKG_EMBEDDING_MODEL``, since the query and document embeddings must
come from the same model.

This OVERWRITES the configured index/ids paths, so it is a deliberate manual
step, never run automatically.
"""

from __future__ import annotations

import json

import faiss
import numpy as np
from openai import OpenAI

from oekg.config import AppConfig, get_config
from oekg.resources import _read_json


def _doc_text(doc: dict) -> str:
    """Canonical text embedded for a class or relation document."""
    label = doc.get("label", "")
    description = doc.get("description", "")
    return f"{label}. {description}".strip()


def build_index(config: AppConfig | None = None) -> int:
    """Build the FAISS index + ids.json. Returns the number of documents indexed."""
    config = config or get_config()
    client = OpenAI(api_key=config.openai_api_key)

    classes = _read_json(config.classes_path)
    relations = _read_json(config.relations_path)

    ids: list[str] = [c["class"] for c in classes] + [r["uri"] for r in relations]
    texts: list[str] = [_doc_text(c) for c in classes] + [_doc_text(r) for r in relations]

    response = client.embeddings.create(model=config.embedding_model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(config.faiss_index_path))

    with open(config.ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    print(
        f"Indexed {len(ids)} documents "
        f"({len(classes)} classes + {len(relations)} relations) "
        f"with {config.embedding_model}.\n"
        f"Wrote {config.faiss_index_path} and {config.ids_path}."
    )
    return len(ids)


if __name__ == "__main__":
    build_index()
