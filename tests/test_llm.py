from oekg.llm import _compact_documents


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
