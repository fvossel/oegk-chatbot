from oekg.oep import OEPClient
from oekg.validation import (
    build_known_vocabulary,
    build_relation_schema,
    extract_literal_mentions,
    find_range_violations,
    find_unknown_uris,
    normalise_sparql,
    summarise_predicate_schema,
    validate_syntax,
)


class _Resources:
    ids = ["oeo:OEO_00000365", "obo:BFO_0000051"]
    documents_dict = {
        "oeo:OEO_00390071": {
            "uri": "oeo:OEO_00390071",
            "range": [{"uri": "oeo:OEO_00010444", "label": "sufficiency"}],
        },
        "oeo:OEO_00000365": {"class": "oeo:OEO_00000365", "label": "scenario factsheet"},
    }
    sparql_system_prompt = "An example uses oeo:OEO_00020247 and oekg:abc-123."


def test_build_known_vocabulary_merges_all_sources():
    known = build_known_vocabulary(_Resources())
    assert "oeo:OEO_00000365" in known        # from ids
    assert "oeo:OEO_00010444" in known        # from a relation's range
    assert "oeo:OEO_00020247" in known        # harvested from the prompt examples
    assert "obo:BFO_0000051" in known


def test_find_unknown_flags_hallucinated_term():
    known = {"oeo:OEO_00000365", "obo:BFO_0000051"}
    query = "SELECT ?s WHERE { ?s a oeo:OEO_99999999 . ?s obo:BFO_0000051 ?o }"
    unknown = find_unknown_uris(query, known)
    assert "oeo:OEO_99999999" in unknown
    assert "obo:BFO_0000051" not in unknown
    assert "oeo:OEO_00000365" not in unknown


def test_find_unknown_ignores_oekg_instance_ids():
    known = {"oeo:OEO_00000365"}
    query = "SELECT ?s WHERE { ?s a oeo:OEO_00000365 . FILTER(?s = oekg:59cf408f-8ab1-7fe1) }"
    assert find_unknown_uris(query, known) == {}


def test_find_unknown_ignores_curies_inside_string_literals():
    query = 'SELECT ?s WHERE { ?s rdfs:label "oeo:OEO_00000365"^^XSD:string }'
    assert find_unknown_uris(query, set()) == {}


def test_find_unknown_suggests_closest_with_label():
    known = {"oeo:OEO_00000365"}
    documents = {"oeo:OEO_00000365": {"label": "scenario factsheet"}}
    unknown = find_unknown_uris("?s a oeo:OEO_00000366 .", known, documents=documents)
    suggestion, label = unknown["oeo:OEO_00000366"]
    assert suggestion == "oeo:OEO_00000365"
    assert label == "scenario factsheet"


# -- output normalisation + syntax validation (#3) --------------------------


def test_normalise_strips_sparql_fence():
    assert normalise_sparql("```sparql\nSELECT ?s WHERE {?s ?p ?o}\n```") == "SELECT ?s WHERE {?s ?p ?o}"


def test_normalise_strips_bare_fence_and_label():
    assert normalise_sparql("```\nSELECT 1\n```") == "SELECT 1"
    assert normalise_sparql("SPARQL: SELECT ?x {}") == "SELECT ?x {}"
    assert normalise_sparql("Query:\nASK {}") == "ASK {}"


def test_normalise_preserves_bot_information():
    assert normalise_sparql("<bot-information> Hello") == "<bot-information> Hello"


def test_validate_syntax_accepts_valid_query():
    query = OEPClient.build_full_query(
        "SELECT ?s ?name WHERE { ?s oeo:OEO_00020439 ?o . "
        "?o a*/rdfs:subClassOf* oeo:OEO_00000158 . ?s rdfs:label ?name }"
    )
    assert validate_syntax(query) is None


def test_validate_syntax_rejects_broken_query():
    query = OEPClient.build_full_query("SELECT ?s WHERE { ?s ?p }")
    assert validate_syntax(query) is not None


def test_validate_syntax_graceful_without_rdflib(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "rdflib.plugins.sparql", None)
    assert validate_syntax("SELECT ?s WHERE { ?s ?p ?o }") is None


# -- schema (domain/range) validation (#4) ----------------------------------


class _SchemaRes:
    relation_ids = frozenset({"oeo:PRED", "oeo:LIT"})
    documents_dict = {
        "oeo:PRED": {
            "uri": "oeo:PRED",
            "label": "has thing",
            "domain": [{"uri": "oeo:D", "label": "dom"}],
            "range": [{"uri": "oeo:A", "label": "a"}, {"uri": "oeo:B", "label": "b"}],
        },
        "oeo:LIT": {"uri": "oeo:LIT", "label": "has value", "range": [{"uri": "Literal"}]},
    }


def test_build_relation_schema_closed_vs_literal():
    schema = build_relation_schema(_SchemaRes())
    assert schema["oeo:PRED"]["closed"] is True
    assert schema["oeo:PRED"]["range_uris"] == frozenset({"oeo:A", "oeo:B"})
    assert schema["oeo:LIT"]["closed"] is False


def test_find_range_violations_flags_out_of_range():
    schema = build_relation_schema(_SchemaRes())
    violations = find_range_violations("SELECT ?s WHERE { ?s oeo:PRED oeo:Z . }", schema)
    assert "oeo:PRED" in violations
    assert violations["oeo:PRED"]["object"] == "oeo:Z"


def test_find_range_violations_ignores_valid_and_indirect():
    schema = build_relation_schema(_SchemaRes())
    assert find_range_violations("SELECT ?s WHERE { ?s oeo:PRED oeo:A . }", schema) == {}
    assert find_range_violations("SELECT ?s WHERE { ?s oeo:PRED ?o . ?o a oeo:A }", schema) == {}
    assert find_range_violations('SELECT ?s WHERE { ?s oeo:LIT "x" }', schema) == {}


def test_find_range_violations_ignores_property_paths_and_values():
    schema = build_relation_schema(_SchemaRes())
    # oeo:PRED is the second step of a property path, not a triple predicate.
    assert find_range_violations("SELECT ?s WHERE { ?s oeo:FIRST/oeo:PRED oeo:Z . }", schema) == {}
    # Two adjacent CURIEs inside a VALUES list must not be read as a triple.
    assert find_range_violations("SELECT ?p WHERE { VALUES ?p { oeo:PRED oeo:Z } }", schema) == {}


def test_summarise_predicate_schema():
    schema = build_relation_schema(_SchemaRes())
    out = summarise_predicate_schema("SELECT ?s WHERE { ?s oeo:PRED oeo:A . }", schema)
    assert "oeo:PRED" in out
    assert "a" in out["oeo:PRED"]["range_labels"]


# -- literal extraction for near-miss repair (#5) ---------------------------


def test_extract_literal_mentions():
    query = (
        'SELECT ?s WHERE { ?s rdfs:label "Germany"^^XSD:string . '
        'FILTER(CONTAINS(LCASE(STR(?l)),"wind")) }'
    )
    assert extract_literal_mentions(query) == ["Germany", "wind"]


def test_extract_literal_mentions_dedup_and_none():
    assert extract_literal_mentions("?s rdfs:label 'A' . ?t rdfs:label 'A' . ?u rdfs:label \"B\"") == ["A", "B"]
    assert extract_literal_mentions("SELECT ?s WHERE { ?s ?p ?o }") == []
