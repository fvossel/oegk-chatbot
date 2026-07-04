from oekg.ontology import term_links


def test_term_links_extracts_and_labels():
    docs = {"oeo:OEO_00000064": {"label": "author"}}
    values = ["https://openenergyplatform.org/ontology/oeo/OEO_00000064", "plain label", None]
    assert term_links(values, docs) == [
        {
            "id": "OEO_00000064",
            "label": "author",
            "url": "https://openenergyplatform.org/ontology/oeo/OEO_00000064/",
        }
    ]


def test_term_links_dedups_and_falls_back_to_id():
    iri = "https://openenergyplatform.org/ontology/oeo/OEO_00000365"
    links = term_links([iri, iri], {})
    assert len(links) == 1
    assert links[0]["label"] == "OEO_00000365"  # no doc -> id as label


def test_term_links_ignores_non_oeo_values():
    assert term_links(["https://openenergyplatform.org/database/tables/foo", "x"], {}) == []
