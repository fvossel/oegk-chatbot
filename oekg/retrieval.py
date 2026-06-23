"""Semantic document retrieval over the ontology vector store.

The ontology corpus is small (a few dozen classes + relations), so retrieval is
deliberately asymmetric: the FAISS search only ranks **classes** (where there
are many and relevance matters), while **every relation** is always included.
A generated SPARQL query almost always needs a relation/predicate URI, and a
flat top-k can easily miss the one a question requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI

from oekg.config import AppConfig, get_config
from oekg.resources import Resources


class Retriever:
    """Embeds a query and returns the relevant ontology documents."""

    def __init__(self, client: OpenAI, resources: Resources, config: AppConfig | None = None) -> None:
        self._client = client
        self._resources = resources
        self._config = config or get_config()

    def top_k(self, query: str, k: int | None = None) -> list[Any]:
        """Return the ``k`` most similar class documents plus all relations."""
        k = k or self._config.retrieval_top_k
        docs = self._resources.documents_dict
        ids = self._resources.ids
        relation_ids = self._resources.relation_ids

        response = self._client.embeddings.create(
            model=self._config.embedding_model,
            input=query,
        )
        query_vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

        class_docs: list[Any] = []
        ntotal = self._resources.faiss_index.ntotal
        if ntotal:
            # Rank the whole (tiny) corpus, then keep the top-k *classes*.
            _, indices = self._resources.faiss_index.search(query_vector, ntotal)
            for idx in indices[0]:
                if idx < 0 or idx >= len(ids):
                    continue
                doc_id = ids[idx]
                if doc_id in relation_ids:
                    continue
                class_docs.append(docs[doc_id])
                if len(class_docs) >= k:
                    break

        # Always provide every relation so the model never lacks a predicate.
        relation_docs = [docs[rid] for rid in ids if rid in relation_ids]
        return class_docs + relation_docs
