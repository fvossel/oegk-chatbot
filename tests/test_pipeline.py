import pandas as pd

from oekg.cache import CacheEntry
from oekg.config import get_config
from oekg.oep import ExecutionResult
from oekg.pipeline import GENERATION_ERROR_MESSAGE, RagPipeline
from oekg.resources import Resources


def _resources():
    return Resources(
        faiss_index=None,
        documents_dict={},
        ids=[],
        sparql_system_prompt="",
        summary_system_prompt="",
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
    def __init__(self, sparql="SELECT ?s WHERE { ?s ?p ?o }", raise_gen=False):
        self.sparql = sparql
        self.raise_gen = raise_gen
        self.generate_calls = 0

    def generate_sparql(self, query, documents):
        self.generate_calls += 1
        if self.raise_gen:
            raise RuntimeError("model unavailable")
        return self.sparql

    def summarise(self, sparql_results, nl_query):
        return "A short summary."

    def explain_sparql(self, query):
        return "explanation"


class FakeOEP:
    def __init__(self, df, repaired=False):
        self.df = df
        self.received = None
        self.repaired = repaired

    def execute(self, sparql_query, on_empty=None, on_error=None):
        self.received = sparql_query
        return ExecutionResult(
            results_df=self.df,
            full_query="PREFIX rdf: <...>\n" + sparql_query,
            sparql_used=sparql_query,
            repaired=self.repaired,
        )


def _build(cache, llm, oep):
    return RagPipeline(
        _resources(),
        cache,
        get_config(),
        client=object(),
        retriever=FakeRetriever(),
        llm=llm,
        oep=oep,
    )


def test_normal_query_returns_structured_table():
    df = pd.DataFrame([{"s": "x"}, {"s": "y"}])
    oep = FakeOEP(df)
    result = _build(FakeCache(), FakeLLM(), oep).answer("list things")
    assert not result.is_bot_message
    assert not result.error
    assert result.results_df is not None
    assert result.row_count == 2
    assert result.full_query.startswith("PREFIX")
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
    llm = FakeLLM(raise_gen=True)
    oep = FakeOEP(pd.DataFrame())
    result = _build(FakeCache(), llm, oep).answer("q")
    assert result.error
    assert result.answer == GENERATION_ERROR_MESSAGE
    assert oep.received is None  # the failed generation never reached the endpoint


def test_cache_hit_skips_generation():
    entry = CacheEntry(
        question="q", sparql="SELECT 1", created_at="", last_used="", hit_count=0
    )
    llm = FakeLLM()
    oep = FakeOEP(pd.DataFrame([{"s": "x"}]))
    cache = FakeCache(entry)
    result = _build(cache, llm, oep).answer("q")
    assert result.cache_hit
    assert llm.generate_calls == 0
    assert oep.received == "SELECT 1"
    assert cache.hits == ["q"]


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
