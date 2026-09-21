from dataclasses import replace
from types import SimpleNamespace

from oekg.config import get_config
from oekg.llm import LLMClient, _compact_documents, strip_reasoning, strip_reasoning_stream


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


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _FakeCompletions:
    def __init__(self, reply="SELECT 1", stream_parts=None):
        self.calls = []
        self._reply = reply
        self._stream_parts = stream_parts or []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._stream_parts)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._reply))],
            usage=SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=512)),
        )


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(**kwargs))


def _llm(prompt_cache_key="", **client_kwargs):
    cfg = replace(get_config(), prompt_cache_key=prompt_cache_key)
    client = _FakeClient(**client_kwargs)
    return client, LLMClient(client, _FakeResources(), cfg)


def _calls(client):
    return client.chat.completions.calls


def test_generate_sparql_orders_static_prompt_first():
    client, llm = _llm()
    assert llm.generate_sparql("find x", [{"class": "oeo:C", "label": "c"}]) == "SELECT 1"
    messages = _calls(client)[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("Request: find x")


def test_no_developer_role_is_ever_sent():
    # KiWi rejects the "developer" role with HTTP 400.
    client, llm = _llm()
    llm.generate_sparql("x", [])
    llm.summarise("rows", "q")
    llm.explain_sparql("SELECT 1")
    roles = {m["role"] for call in _calls(client) for m in call["messages"]}
    assert roles == {"system", "user"}


def test_generate_sparql_uses_the_configured_model():
    client, llm = _llm()
    llm.generate_sparql("x", [])
    assert _calls(client)[0]["model"] == get_config().sparql_model


def test_generate_sparql_appends_grounding():
    client, llm = _llm()
    llm.generate_sparql("find x", [], grounding="### Verified labels block")
    assert "### Verified labels block" in _calls(client)[0]["messages"][1]["content"]
    llm.generate_sparql("find y", [])
    assert "Verified labels" not in _calls(client)[1]["messages"][1]["content"]


def test_generate_sparql_strips_inline_reasoning():
    _, llm = _llm(reply="<think>\nThe user wants x...\n</think>\n\nSELECT ?x WHERE { ?x ?p ?o }")
    assert llm.generate_sparql("x", []) == "SELECT ?x WHERE { ?x ?p ?o }"


def test_strip_reasoning_handles_multiple_blocks_and_plain_text():
    assert strip_reasoning("<think>a</think>A<THINK>b</THINK>B") == "AB"
    assert strip_reasoning("  plain answer  ") == "plain answer"


def test_none_content_becomes_empty_string():
    _, llm = _llm(reply=None)
    assert llm.generate_sparql("x", []) == ""


def test_summarise_stream_yields_deltas_and_skips_empty_chunks():
    parts = [_chunk("Hel"), SimpleNamespace(choices=[]), _chunk(None), _chunk("lo")]
    client, llm = _llm(stream_parts=parts)
    assert "".join(llm.summarise_stream("rows", "q")) == "Hello"
    call = _calls(client)[0]
    assert call["stream"] is True
    assert call["model"] == get_config().summary_model
    assert call["messages"][0] == {"role": "system", "content": "SUMMARY PROMPT"}


def test_summarise_returns_empty_string_on_failure():
    class _Broken:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: 1 / 0))

    llm = LLMClient(_Broken(), _FakeResources(), get_config())
    assert llm.summarise("rows", "q") == ""
    assert llm.explain_sparql("SELECT 1") == ""
    assert list(llm.summarise_stream("rows", "q")) == []


def test_prompt_cache_key_passed_via_extra_body():
    client, llm = _llm(prompt_cache_key="oekg-v1")
    llm.generate_sparql("x", [])
    assert _calls(client)[0]["extra_body"] == {"prompt_cache_key": "oekg-v1"}


def test_no_extra_body_when_cache_key_unset():
    client, llm = _llm()
    llm.generate_sparql("x", [])
    assert _calls(client)[0]["extra_body"] is None


def test_cached_tokens_reads_usage():
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=512))
    )
    assert LLMClient.cached_tokens(resp) == 512


def test_cached_tokens_missing_usage_returns_zero():
    assert LLMClient.cached_tokens(SimpleNamespace()) == 0
    # KiWi reports usage without any details object.
    assert LLMClient.cached_tokens(SimpleNamespace(usage=SimpleNamespace())) == 0


# -- streaming reasoning filter --------------------------------------------------


def _collect(chunks):
    return "".join(strip_reasoning_stream(chunks))


def test_stream_without_reasoning_passes_through_unchanged():
    assert list(strip_reasoning_stream(["Hel", "lo ", "world"])) == ["Hel", "lo ", "world"]


def test_stream_does_not_buffer_once_there_is_no_reasoning_block():
    seen = []

    def source():
        for part in ["Hi", " there", " again"]:
            seen.append(part)
            yield part

    stream = strip_reasoning_stream(source())
    assert next(stream) == "Hi"
    assert seen == ["Hi"]  # the first chunk is emitted before later ones are read


def test_stream_strips_leading_block_split_across_chunks():
    assert _collect(["<thi", "nk>the user wants x</th", "ink>", "\n\n", "Hello", " world"]) == "Hello world"


def test_stream_strips_block_after_leading_whitespace_and_in_one_chunk():
    assert _collect(["\n <think>reasoning</think>\nAnswer"]) == "Answer"


def test_stream_drops_an_unclosed_reasoning_block():
    assert _collect(["<think>still thinking when the reply was cut"]) == ""


def test_stream_keeps_replies_that_only_look_like_a_tag():
    assert _collect(["<t", "able>"]) == "<table>"
    assert _collect(["<th"]) == "<th"
    assert _collect([]) == ""


def test_summarise_stream_strips_reasoning_from_the_model_stream():
    parts = [_chunk("<think>"), _chunk("plan"), _chunk("</think>\n"), _chunk("147 scenarios.")]
    _, llm = _llm(stream_parts=parts)
    assert "".join(llm.summarise_stream("rows", "q")) == "147 scenarios."


# -- explanation grounding -------------------------------------------------------


def test_explain_sparql_lists_the_labels_of_the_query_terms():
    client, llm = _llm()
    llm.explain_sparql("SELECT ?s { ?s a oeo:OEO_00020227 }", term_labels={"oeo:OEO_00020227": "scenario bundle"})
    user = _calls(client)[0]["messages"][1]["content"]
    assert user.startswith("SELECT ?s { ?s a oeo:OEO_00020227 }")
    assert "- oeo:OEO_00020227: scenario bundle" in user


def test_explain_sparql_without_labels_sends_only_the_query():
    client, llm = _llm()
    llm.explain_sparql("SELECT 1", term_labels={})
    assert _calls(client)[0]["messages"][1]["content"] == "SELECT 1"


def test_explain_prompt_forbids_guessing_identifier_meanings():
    client, llm = _llm()
    llm.explain_sparql("SELECT 1")
    assert "never guess" in _calls(client)[0]["messages"][0]["content"]
