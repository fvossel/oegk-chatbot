from pandas import DataFrame

from oekg.transforms import convert_to_df, get_scenarios

_SCENARIO_URI = "https://openenergyplatform.org/ontology/oekg/scenario/123"


def test_convert_to_df_fills_missing_bindings_with_none():
    data = {
        "head": {"vars": ["s", "label"]},
        "results": {
            "bindings": [
                {"s": {"value": "x"}, "label": {"value": "L"}},
                {"s": {"value": "y"}},  # 'label' missing for this row
            ]
        },
    }
    df = convert_to_df(data)
    assert list(df.columns) == ["s", "label"]
    assert df.iloc[0]["label"] == "L"
    assert df.iloc[1]["label"] is None


def test_convert_to_df_empty_bindings():
    data = {"head": {"vars": ["nothing"]}, "results": {"bindings": []}}
    assert convert_to_df(data).empty


def test_get_scenarios_detects_and_puts_scenario_first():
    df = DataFrame([{"a": "foo", "scenario": _SCENARIO_URI, "year": "2030"}])
    cols = get_scenarios(df)
    assert cols[0] == "scenario"
    assert set(cols) == {"a", "scenario", "year"}


def test_get_scenarios_stops_at_empty_neighbour():
    df = DataFrame([{"a": None, "scenario": _SCENARIO_URI, "year": "2030"}])
    cols = get_scenarios(df)
    assert "a" not in cols
    assert cols[0] == "scenario"


def test_get_scenarios_returns_empty_when_absent():
    df = DataFrame([{"a": "foo", "b": "bar"}])
    assert get_scenarios(df) == []
