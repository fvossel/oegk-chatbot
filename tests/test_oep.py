import json

import pandas as pd

import oekg.oep as oep_module
from oekg.config import get_config
from oekg.oep import OEPClient, _EMPTY_RESULT


def _bindings(rows):
    return {"head": {"vars": []}, "results": {"bindings": rows}}


class _FakeResponse:
    def __init__(self, body_text):
        self.text = body_text
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.text)


def test_run_sparql_parses_double_encoded_json(monkeypatch):
    # The OEP endpoint returns a JSON *string* whose content is the JSON object.
    payload = {"head": {"vars": ["s"]}, "results": {"bindings": [{"s": {"value": "x"}}]}}
    body = json.dumps(json.dumps(payload))  # double-encoded
    monkeypatch.setattr(oep_module.requests, "post", lambda *a, **k: _FakeResponse(body))
    client = OEPClient("token", get_config())
    result = client.run_sparql("SELECT ?s WHERE { ?s ?p ?o }")
    assert isinstance(result, dict)
    assert result["results"]["bindings"][0]["s"]["value"] == "x"


def test_run_sparql_parses_plain_json(monkeypatch):
    # Defensive: a normally-encoded JSON object body must also work.
    payload = {"head": {"vars": ["s"]}, "results": {"bindings": []}}
    monkeypatch.setattr(oep_module.requests, "post", lambda *a, **k: _FakeResponse(json.dumps(payload)))
    client = OEPClient("token", get_config())
    result = client.run_sparql("SELECT ?s WHERE { ?s ?p ?o }")
    assert result["head"]["vars"] == ["s"]


def test_bundle_scenarios_batches_and_groups(monkeypatch):
    client = OEPClient("token", get_config())
    s1 = "https://openenergyplatform.org/ontology/oekg/scenario/s1"
    s2 = "https://openenergyplatform.org/ontology/oekg/scenario/s2"
    call_count = {"n": 0}

    def fake_run(query, strict=False):
        call_count["n"] += 1
        if "?bundle obo:BFO_0000051" in query:
            return _bindings(
                [
                    {"scenario": {"value": s1}, "bundle": {"value": "B"}, "bundleLabel": {"value": "Bundle B"}},
                    {"scenario": {"value": s2}, "bundle": {"value": "B"}, "bundleLabel": {"value": "Bundle B"}},
                ]
            )
        if "dc:acronym" in query:
            return _bindings(
                [
                    {"scenario": {"value": s1}, "scenarioLabel": {"value": "Label1"}, "scenarioAcronym": {"value": "A1"}},
                    {"scenario": {"value": s2}, "scenarioLabel": {"value": "Label2"}, "scenarioAcronym": {"value": "A2"}},
                ]
            )
        return dict(_EMPTY_RESULT)

    monkeypatch.setattr(client, "run_sparql", fake_run)
    df = pd.DataFrame([{"scenario": s1, "x": "1"}, {"scenario": s2, "x": "2"}])
    rows = client._bundle_scenarios_by_bundle(df, ["scenario", "x"])

    # Two scenarios resolved with exactly two endpoint calls (no per-row N+1).
    assert call_count["n"] == 2
    assert len(rows) == 2
    assert rows[0]["Bundle URI"] == "B"
    assert rows[0]["Bundle Label"] == "Bundle B"
    assert rows[1]["Bundle URI"] == ""  # repeated bundle blanked after first row
    assert {r["Scenario Acronym"] for r in rows} == {"A1", "A2"}
    assert {r["x"] for r in rows} == {"1", "2"}  # extra columns carried through


def test_resolve_uri_returns_original_on_empty_bindings(monkeypatch):
    client = OEPClient("token", get_config())
    monkeypatch.setattr(client, "run_sparql", lambda query, strict=False: dict(_EMPTY_RESULT))
    uri = "https://openenergyplatform.org/ontology/oekg/input_datasets/abc"
    assert client.resolve_uri(uri) == uri  # no IndexError on empty result


def test_resolve_uri_is_memoised(monkeypatch):
    client = OEPClient("token", get_config())
    calls = []

    def fake_run(query, strict=False):
        calls.append(query)
        return dict(_EMPTY_RESULT)

    monkeypatch.setattr(client, "run_sparql", fake_run)
    uri = "https://openenergyplatform.org/ontology/oekg/input_datasets/abc"
    client.resolve_uri(uri)
    client.resolve_uri(uri)
    assert len(calls) == 1  # second lookup served from the in-instance cache


def test_resolve_uri_passes_through_non_oekg_values():
    client = OEPClient("token", get_config())
    assert client.resolve_uri("just a label") == "just a label"
    assert client.resolve_uri(None) is None


def _label_bindings(values):
    return {
        "head": {"vars": ["label"]},
        "results": {"bindings": [{"label": {"value": v}} for v in values]},
    }


def test_find_near_miss_labels_returns_distinct(monkeypatch):
    client = OEPClient("token", get_config())
    captured = {}

    def fake(query, strict=False):
        captured["q"] = query
        return _label_bindings(["Germany", "Germany", "Germany 2030"])

    monkeypatch.setattr(client, "run_sparql", fake)
    out = client.find_near_miss_labels(["Germany"], 8)
    assert out == ["Germany", "Germany 2030"]
    assert "CONTAINS" in captured["q"] and "LCASE" in captured["q"] and "LIMIT" in captured["q"]


def test_find_near_miss_labels_empty_cases(monkeypatch):
    client = OEPClient("token", get_config())
    calls = []

    def fake(query, strict=False):
        calls.append(query)
        return dict(_EMPTY_RESULT)

    monkeypatch.setattr(client, "run_sparql", fake)
    assert client.find_near_miss_labels([], 8) == []
    assert client.find_near_miss_labels(["EU"], 8) == []  # token below min length
    assert calls == []  # neither issued a query
    assert client.find_near_miss_labels(["wind"], 8) == []  # empty result
    assert len(calls) == 1


def test_execute_advisory_syntax_repair(monkeypatch):
    client = OEPClient("token", get_config())
    monkeypatch.setattr(oep_module, "validate_syntax", lambda q: "boom" if "BAD" in q else None)
    monkeypatch.setattr(client, "run_sparql", lambda q, strict=False: _label_bindings(["X"]))
    seen = {}

    def on_error(previous, error):
        seen["err"] = error
        return "GOOD QUERY"

    result = client.execute("BAD QUERY", on_error=on_error)
    assert seen["err"] == "boom"
    assert result.repaired is True
    assert result.sparql_used == "GOOD QUERY"


def test_execute_syntax_repaired_query_runs_even_with_zero_rounds(monkeypatch):
    from dataclasses import replace

    client = OEPClient("token", replace(get_config(), sparql_repair_rounds=0))
    monkeypatch.setattr(oep_module, "validate_syntax", lambda q: "boom" if "BAD" in q else None)
    monkeypatch.setattr(
        client,
        "run_sparql",
        lambda q, strict=False: _label_bindings(["X"]) if "GOOD" in q else dict(_EMPTY_RESULT),
    )
    result = client.execute("BAD QUERY", on_error=lambda prev, err: "GOOD QUERY")
    assert result.repaired is True
    assert result.sparql_used == "GOOD QUERY"
    assert not result.results_df.empty  # the repaired query was actually executed
