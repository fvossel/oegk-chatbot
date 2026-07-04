from dataclasses import replace
from types import SimpleNamespace

from oekg.config import get_config
from oekg.llm import LLMClient, _compact_documents


def test_compact_documents_collapses_domain_and_range_to_labels():
    docs = [
        {
            "uri": "oeo:R",
            "label": "has x",
            "description": "a relation",
            "domain": [{"uri": "oeo:D", "label": "dom"}],
            "range": [{"uri": "oeo:A", "label": "a"}, {"uri": "oeo:B", "label": "b"}],
        },
        {"class": "oeo:C", "label": "cls", "description": "a class"},
    ]
    out = _compact_documents(docs)

    assert out[0]["uri"] == "oeo:R"
    assert out[0]["label"] == "has x"
    assert out[0]["domain"] == ["dom"]
    assert out[0]["range"] == ["a", "b"]  # nested {uri,label} flattened to labels

    assert out[1]["uri"] == "oeo:C"  # class id surfaced under "uri"
    assert "domain" not in out[1]
    assert "range" not in out[1]


def test_compact_documents_skips_non_dicts():
    assert _compact_documents(["junk", 5, None]) == []


class _FakeResources:
    sparql_system_prompt = "SYSTEM PROMPT"
    summary_system_prompt = "SUMMARY PROMPT"


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="SELECT 1",
            usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=512)),
        )


class _FakeClient:
    def __init__(self):
        self.responses = _FakeResponses()


def _llm(prompt_cache_key=""):
    cfg = replace(get_config(), prompt_cache_key=prompt_cache_key)
    client = _FakeClient()
    return client, LLMClient(client, _FakeResources(), cfg)


def test_generate_sparql_orders_static_prompt_first():
    client, llm = _llm()
    llm.generate_sparql("find x", [{"class": "oeo:C", "label": "c"}])
    call = client.responses.calls[0]
    assert call["input"][0]["role"] == "developer"
    assert call["input"][0]["content"] == "SYSTEM PROMPT"
    assert call["input"][1]["role"] == "user"
    assert call["input"][1]["content"].startswith("Request: find x")


def test_generate_sparql_appends_grounding():
    client, llm = _llm()
    llm.generate_sparql("find x", [], grounding="### Verified labels block")
    assert "### Verified labels block" in client.responses.calls[0]["input"][1]["content"]
    llm.generate_sparql("find y", [])
    assert "Verified labels" not in client.responses.calls[1]["input"][1]["content"]


def test_prompt_cache_key_passed_via_extra_body():
    client, llm = _llm(prompt_cache_key="oekg-v1")
    llm.generate_sparql("x", [])
    assert client.responses.calls[0]["extra_body"] == {"prompt_cache_key": "oekg-v1"}


def test_no_extra_body_when_cache_key_unset():
    client, llm = _llm()
    llm.generate_sparql("x", [])
    assert client.responses.calls[0]["extra_body"] is None


def test_cached_tokens_reads_usage():
    resp = SimpleNamespace(
        usage=SimpleNamespace(input_tokens_details=SimpleNamespace(cached_tokens=512))
    )
    assert LLMClient.cached_tokens(resp) == 512


def test_cached_tokens_missing_usage_returns_zero():
    assert LLMClient.cached_tokens(SimpleNamespace()) == 0
