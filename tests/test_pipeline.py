from dataclasses import replace

import pandas as pd

from oekg.cache import CacheEntry
from oekg.config import get_config
from oekg.oep import _EMPTY_RESULT, ExecutionResult
from oekg.pipeline import GENERATION_ERROR_MESSAGE, RagPipeline
from oekg.resources import Resources


def _resources(documents_dict=None, ids=None, relation_ids=frozenset()):
    return Resources(
        faiss_index=None,
        documents_dict=documents_dict or {},
        ids=ids or [],
        sparql_system_prompt="",
        summary_system_prompt="",
        relation_ids=relation_ids,
    )


class FakeCache:
    def __init__(self, entry=None):
        self._entry = entry
        self.hits = []

    def get(self, question):
        return self._entry

    def record_hit(self, question):
        self.hits.append(question)


class FakeRetriever:
    def top_k(self, query, k=None):
        return []


class FakeLLM:
    def __init__(self, sparql="SELECT ?s WHERE { ?s ?p ?o }", raise_gen=False, outputs=None):
        self.sparql = sparql
        self.raise_gen = raise_gen
        self.outputs = list(outputs) if outputs else None
        self.generate_calls = 0
        self.last_prompt = None
        self.last_grounding = None

    def generate_sparql(self, query, documents, grounding=None):
        self.generate_calls += 1
        self.last_prompt = query
        self.last_grounding = grounding
        if self.raise_gen:
            raise RuntimeError("model unavailable")
        if self.outputs:
            return self.outputs.pop(0)
        return self.sparql

    def summarise(self, sparql_results, nl_query):
        return "A short summary."

    def explain_sparql(self, query):
        return "explanation"


class FakeOEP:
    def __init__(self, df, repaired=False, near_miss=None, grounding_bindings=None):
        self.df = df
        self.received = None
        self.repaired = repaired
        self.near_miss = [] if near_miss is None else list(near_miss)
        self.near_miss_calls = 0
        self.grounding_bindings = grounding_bindings or dict(_EMPTY_RESULT)
        self.grounding_called = False

    def execute(self, sparql_query, on_empty=None, on_error=None):
        self.received = sparql_query
        return ExecutionResult(
            results_df=self.df,
            full_query="PREFIX rdf: <...>\n" + sparql_query,
            sparql_used=sparql_query,
            repaired=self.repaired,
        )

    def run_sparql(self, query, strict=False):
        self.grounding_called = True
        return self.grounding_bindings

    def build_full_query(self, query):
        return "PREFIX ...\n" + query

    def find_near_miss_labels(self, mentions, limit):
        self.near_miss_calls += 1
        return list(self.near_miss)


def _build(cache, llm, oep, config=None, resources=None):
    return RagPipeline(
        resources or _resources(),
        cache,
        config or get_config(),
        client=object(),
        retriever=FakeRetriever(),
        llm=llm,
        oep=oep,
    )


# -- core flow ---------------------------------------------------------------


def test_normal_query_returns_structured_table():
    df = pd.DataFrame([{"s": "x"}, {"s": "y"}])
    oep = FakeOEP(df)
    result = _build(FakeCache(), FakeLLM(), oep).answer("list things")
    assert not result.is_bot_message
    assert not result.error
    assert result.results_df is not None
    assert result.row_count == 2
    assert result.answer == "A short summary."


def test_empty_results_message():
    result = _build(FakeCache(), FakeLLM(), FakeOEP(pd.DataFrame())).answer("q")
    assert result.results_df is None
    assert "No results" in result.answer


def test_bot_information_short_circuits():
    llm = FakeLLM(sparql="<bot-information>Hi, I answer OEKG questions.")
    result = _build(FakeCache(), llm, FakeOEP(pd.DataFrame())).answer("hello")
    assert result.is_bot_message
    assert result.answer == "Hi, I answer OEKG questions."
    assert result.sparql is None


def test_generation_error_is_surfaced_not_executed():
    oep = FakeOEP(pd.DataFrame())
    result = _build(FakeCache(), FakeLLM(raise_gen=True), oep).answer("q")
    assert result.error
    assert result.answer == GENERATION_ERROR_MESSAGE
    assert oep.received is None


def test_cache_hit_skips_generation():
    entry = CacheEntry(question="q", sparql="SELECT 1", created_at="", last_used="", hit_count=0)
    llm = FakeLLM()
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    result = _build(FakeCache(entry), llm, oep).answer("q")
    assert result.cache_hit
    assert llm.generate_calls == 0
    assert oep.received == "SELECT 1"


def test_forced_sparql_executes_given_query_without_generation():
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    llm = FakeLLM()
    result = _build(FakeCache(), llm, oep).answer("q", forced_sparql="SELECT FORCED")
    assert result.cache_hit
    assert llm.generate_calls == 0
    assert oep.received == "SELECT FORCED"


def test_corrected_flag_set_when_query_was_repaired():
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]), repaired=True)
    result = _build(FakeCache(), FakeLLM(), oep).answer("q")
    assert result.corrected


# -- entity grounding (#1) ---------------------------------------------------


def test_grounding_fragment_reaches_generation():
    bindings = {
        "results": {
            "bindings": [
                {
                    "needle": {"value": "messageix"},
                    "label": {"value": "MESSAGEix v1.2"},
                    "s": {"value": "http://x"},
                }
            ]
        }
    }
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]), grounding_bindings=bindings)
    llm = FakeLLM()
    _build(FakeCache(), llm, oep).answer("bundles for MESSAGEix")
    assert oep.grounding_called
    assert llm.last_grounding and "MESSAGEix v1.2" in llm.last_grounding


def test_grounding_skipped_when_disabled():
    cfg = replace(get_config(), entity_grounding_enabled=False)
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    _build(FakeCache(), FakeLLM(), oep, config=cfg).answer("bundles for MESSAGEix")
    assert oep.grounding_called is False


# -- schema (domain/range) validation (#4) -----------------------------------

_SCHEMA_DOCS = {
    "oeo:PRED": {
        "uri": "oeo:PRED",
        "label": "has thing",
        "domain": [{"uri": "oeo:D", "label": "dom"}],
        "range": [{"uri": "oeo:INRANGE", "label": "in range"}],
    },
    "oeo:OUTOFRANGE": {"class": "oeo:OUTOFRANGE", "label": "out of range"},
    "oeo:INRANGE": {"class": "oeo:INRANGE", "label": "in range"},
}


def _schema_resources():
    return _resources(
        documents_dict=_SCHEMA_DOCS,
        ids=list(_SCHEMA_DOCS),
        relation_ids=frozenset({"oeo:PRED"}),
    )


def test_schema_validation_repairs_out_of_range_object():
    cfg = replace(get_config(), entity_grounding_enabled=False)
    llm = FakeLLM(
        outputs=[
            "SELECT ?s WHERE { ?s oeo:PRED oeo:OUTOFRANGE . }",
            "SELECT ?s WHERE { ?s oeo:PRED oeo:INRANGE . }",
        ]
    )
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    result = _build(FakeCache(), llm, oep, config=cfg, resources=_schema_resources()).answer("q")
    assert llm.generate_calls == 2
    assert result.corrected
    assert "oeo:INRANGE" in oep.received


def test_schema_validation_ignores_in_range_object():
    cfg = replace(get_config(), entity_grounding_enabled=False)
    llm = FakeLLM(sparql="SELECT ?s WHERE { ?s oeo:PRED oeo:INRANGE . }")
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    result = _build(FakeCache(), llm, oep, config=cfg, resources=_schema_resources()).answer("q")
    assert llm.generate_calls == 1
    assert not result.corrected


def test_schema_validation_skipped_when_disabled():
    cfg = replace(get_config(), entity_grounding_enabled=False, schema_validation_enabled=False)
    llm = FakeLLM(sparql="SELECT ?s WHERE { ?s oeo:PRED oeo:OUTOFRANGE . }")
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    _build(FakeCache(), llm, oep, config=cfg, resources=_schema_resources()).answer("q")
    assert llm.generate_calls == 1


# -- near-miss empty-result repair (#5) --------------------------------------


def test_on_empty_uses_near_miss_labels():
    llm = FakeLLM()
    oep = FakeOEP(pd.DataFrame(), near_miss=["Germany 2030"])
    pipeline = _build(FakeCache(), llm, oep)
    pipeline._on_empty("DIALOGUE\n", [], "")('SELECT ?s WHERE { ?s rdfs:label "Deutschlnd" }')
    assert oep.near_miss_calls == 1
    assert "Germany 2030" in llm.last_prompt


def test_on_empty_falls_back_without_labels():
    llm = FakeLLM()
    oep = FakeOEP(pd.DataFrame(), near_miss=[])
    pipeline = _build(FakeCache(), llm, oep)
    pipeline._on_empty("DIALOGUE\n", [], "")('SELECT ?s WHERE { ?s rdfs:label "X" }')
    assert "returned no results" in llm.last_prompt


def test_on_empty_respects_disabled_flag():
    cfg = replace(get_config(), near_miss_enabled=False)
    llm = FakeLLM()
    oep = FakeOEP(pd.DataFrame(), near_miss=["L"])
    pipeline = _build(FakeCache(), llm, oep, config=cfg)
    pipeline._on_empty("DIALOGUE\n", [], "")('SELECT ?s WHERE { ?s rdfs:label "X" }')
    assert oep.near_miss_calls == 0
    assert "returned no results" in llm.last_prompt


# -- OEP data integration (#1/#3) + ontology term links (#9) ------------------


def test_dataset_references_extracts_distinct_tables():
    df = pd.DataFrame(
        [{"x": "https://openenergyplatform.org/database/tables/foo", "y": "a label"}]
    )
    refs = _build(FakeCache(), FakeLLM(), FakeOEP(df)).dataset_references(df)
    assert refs == [("https://openenergyplatform.org/database/tables/foo", "foo")]


def test_ontology_links_from_results():
    docs = {"oeo:OEO_00000064": {"label": "author"}}
    df = pd.DataFrame([{"t": "https://openenergyplatform.org/ontology/oeo/OEO_00000064"}])
    pipeline = _build(FakeCache(), FakeLLM(), FakeOEP(df), resources=_resources(documents_dict=docs))
    links = pipeline.ontology_links(df)
    assert links[0]["label"] == "author"
    assert links[0]["url"].endswith("/OEO_00000064/")
