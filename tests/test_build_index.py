"""Tests for the manual vector-store rebuild."""

import json
from dataclasses import replace
from types import SimpleNamespace

import faiss

from oekg import build_index as module
from oekg.build_index import _doc_text
from oekg.config import get_config


def test_doc_text_combines_label_and_description():
    assert _doc_text({"label": "Author", "description": "An author."}) == "Author. An author."


def test_doc_text_handles_missing_description():
    assert _doc_text({"label": "Author"}) == "Author."


def test_build_index_uses_the_configured_embedding_endpoint(tmp_path, monkeypatch):
    # The index must be built with the same endpoint/model that queries use,
    # and must never reach for OPENAI_API_KEY.
    (tmp_path / "classes.json").write_text(
        json.dumps([{"class": "C1", "label": "a", "description": "d"}]), encoding="utf-8"
    )
    (tmp_path / "relations.json").write_text(
        json.dumps([{"uri": "R1", "label": "b", "description": "d"}]), encoding="utf-8"
    )
    cfg = replace(
        get_config(),
        classes_path=tmp_path / "classes.json",
        relations_path=tmp_path / "relations.json",
        ids_path=tmp_path / "ids.json",
        faiss_index_path=tmp_path / "v.faiss",
        embedding_api_key="gateway-test-key",
        embedding_base_url="https://embed.example.org/v1",
        embedding_model="test-embed",
    )
    seen: dict = {}

    class FakeEmbeddings:
        def create(self, model, input):
            seen["model"] = model
            seen["documents"] = list(input)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])

    def fake_client(api_key, base_url):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        return SimpleNamespace(embeddings=FakeEmbeddings())

    monkeypatch.setattr(module, "OpenAI", fake_client)

    assert module.build_index(cfg) == 2
    assert seen["api_key"] == "gateway-test-key"
    assert seen["base_url"] == "https://embed.example.org/v1"
    assert seen["model"] == "test-embed"
    assert seen["documents"] == ["a. d", "b. d"]
    # Row i of the index must correspond to ids[i].
    assert json.loads((tmp_path / "ids.json").read_text(encoding="utf-8")) == ["C1", "R1"]
    assert faiss.read_index(str(tmp_path / "v.faiss")).ntotal == 2
