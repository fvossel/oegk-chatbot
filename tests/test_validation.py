from oekg.validation import build_known_vocabulary, find_unknown_uris


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
