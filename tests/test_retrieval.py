from types import SimpleNamespace

import faiss
import numpy as np

from oekg.config import get_config
from oekg.resources import Resources
from oekg.retrieval import Retriever


class _FakeClient:
    """Returns a fixed embedding vector for any input."""

    def __init__(self, vector):
        self._vector = vector
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=self._vector)])


def _resources():
    docs = {
        "c1": {"class": "c1", "label": "C1"},
        "c2": {"class": "c2", "label": "C2"},
        "c3": {"class": "c3", "label": "C3"},
        "r1": {"uri": "r1", "label": "R1"},
    }
    ids = ["c1", "c2", "c3", "r1"]
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.5, 0.5]], dtype=np.float32
    )
    index = faiss.IndexFlatL2(2)
    index.add(vectors)
    return Resources(
        faiss_index=index,
        documents_dict=docs,
        ids=ids,
        sparql_system_prompt="",
        summary_system_prompt="",
        relation_ids=frozenset({"r1"}),
    )


def test_top_k_returns_top_classes_plus_all_relations():
    resources = _resources()
    client = _FakeClient([1.0, 0.0])  # closest to c1, then c3
    retriever = Retriever(client, resources, get_config())

    result = retriever.top_k("anything", k=2)
    labels = [doc["label"] for doc in result]

    # Top-2 nearest classes (C1, C3), plus every relation (R1), C2 excluded.
    assert "R1" in labels  # relation always included
    assert "C1" in labels and "C3" in labels
    assert "C2" not in labels
    assert len(result) == 3
