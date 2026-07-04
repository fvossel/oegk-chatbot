from oekg.grounding import (
    build_grounding_query,
    extract_mentions,
    ground_question,
    parse_grounding_results,
    render_grounding_prompt,
)


def test_extract_mentions_quoted_and_proper():
    out = extract_mentions('Give me bundles for "Fraunhofer ISI" and MESSAGEix')
    assert "Fraunhofer ISI" in out
    assert "MESSAGEix" in out
    assert "Give" not in out  # stopword dropped


def test_extract_mentions_drops_years_and_stopwords():
    assert extract_mentions("Show all scenarios for 2050 in Germany") == ["Germany"]


def test_extract_mentions_respects_max():
    assert extract_mentions("Alpha or Beta or Gamma or Delta", max_mentions=2) == ["Alpha", "Beta"]


def test_build_grounding_query_none_when_empty():
    assert build_grounding_query([]) is None


def test_build_grounding_query_single_select_and_escapes():
    query = build_grounding_query(['A"B'])
    assert query.count("SELECT") == 1
    assert "VALUES" in query and "LIMIT" in query
    assert 'a\\"b' in query  # lowercased + escaped literal


def test_parse_and_render():
    raw = {
        "results": {
            "bindings": [
                {
                    "needle": {"value": "messageix"},
                    "label": {"value": "MESSAGEix v1.2"},
                    "s": {"value": "http://x"},
                    "acronym": {"value": "MSG"},
                    "types": {"value": "http://t/Framework"},
                }
            ]
        }
    }
    entities = parse_grounding_results(raw, ["MESSAGEix"])
    assert len(entities) == 1
    assert entities[0].label == "MESSAGEix v1.2"
    assert entities[0].mention == "MESSAGEix"
    assert entities[0].types == ("Framework",)

    block = render_grounding_prompt(entities)
    assert "MESSAGEix v1.2" in block and "MESSAGEix" in block


def test_parse_caps_per_mention():
    raw = {
        "results": {
            "bindings": [
                {"needle": {"value": "x"}, "label": {"value": "L1"}, "s": {"value": "u1"}},
                {"needle": {"value": "x"}, "label": {"value": "L2"}, "s": {"value": "u2"}},
            ]
        }
    }
    assert len(parse_grounding_results(raw, ["X"], limit_per_mention=1)) == 1


def test_parse_and_render_empty():
    assert parse_grounding_results({}, ["X"]) == []
    assert render_grounding_prompt([]) == ""


def test_ground_question_happy_path():
    called = []

    def run_query(query):
        called.append(query)
        return {
            "results": {
                "bindings": [
                    {
                        "needle": {"value": "messageix"},
                        "label": {"value": "MESSAGEix v1.2"},
                        "s": {"value": "u"},
                    }
                ]
            }
        }

    block = ground_question("bundles for MESSAGEix", run_query)
    assert "MESSAGEix v1.2" in block
    assert len(called) == 1


def test_ground_question_no_mentions_skips_query():
    called = []
    block = ground_question("list all bundles", lambda q: called.append(q) or {})
    assert block == ""
    assert called == []


def test_ground_question_degrades_on_exception():
    def boom(query):
        raise RuntimeError("endpoint down")

    assert ground_question("bundles for MESSAGEix", boom) == ""
