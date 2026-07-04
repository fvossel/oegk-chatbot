from types import SimpleNamespace

import oekg.dataapi as dataapi_module
from oekg.config import get_config
from oekg.dataapi import OEPDataClient, summarise_meta

_META = {
    "title": "",
    "description": "GHG projections",
    "resources": [
        {
            "spatial": {"extent": {"name": "EU", "resolutionValue": "national", "crs": ""}},
            "temporal": {"timeseries": [{"start": "2015-09-09", "end": "2050-09-09"}], "referenceDate": ""},
            "licenses": [
                {
                    "name": "CC BY 4.0",
                    "path": "https://creativecommons.org/licenses/by/4.0/",
                    "attribution": "(c) Öko-Institut",
                }
            ],
            "sources": [{"title": "Reportnet 3", "path": "https://reportnet.europa.eu/"}],
            "schema": {"fields": [{"name": "value", "unit": "kt"}, {"name": "id", "unit": "null"}]},
            "review": {"badge": None},
        }
    ],
}


def _response(payload):
    return SimpleNamespace(status_code=200, json=lambda: payload, raise_for_status=lambda: None)


def test_table_from_url_forms_and_non_match():
    assert OEPDataClient.table_from_url(
        "https://openenergyplatform.org/database/tables/eu_leg_data_2021"
    ) == "eu_leg_data_2021"
    assert OEPDataClient.table_from_url(
        "https://openenergyplatform.org/dataedit/view/supply/wind_turbine_library"
    ) == "wind_turbine_library"
    assert OEPDataClient.table_from_url("https://openenergyplatform.org/scenario-bundles/id/abc") is None
    assert OEPDataClient.table_from_url(None) is None


def test_summarise_meta_extracts_fields():
    card = summarise_meta(_META, "tbl")
    assert card["region"] == "EU"
    assert card["resolution"] == "national"
    assert card["temporal"] == "2015-09-09 – 2050-09-09"
    assert card["license"]["name"] == "CC BY 4.0"
    assert card["license"]["attribution"] == "(c) Öko-Institut"
    assert card["sources"][0]["title"] == "Reportnet 3"
    assert card["units"] == {"value": "kt"}  # the "null" unit is filtered out
    assert card["badge"] is None
    assert card["n_fields"] == 2


def test_fetch_rows_and_meta_with_cache(monkeypatch):
    client = OEPDataClient("token", get_config())

    def fake_get(url, params=None, headers=None, timeout=None):
        if url.endswith("/rows/"):
            return _response([{"a": 1}, {"a": 2}])
        return _response(_META)

    monkeypatch.setattr(dataapi_module.requests, "get", fake_get)
    df = client.fetch_rows("tbl", 5)
    assert df is not None and list(df["a"]) == [1, 2]
    assert client.metadata_card("tbl")["region"] == "EU"

    calls = []
    monkeypatch.setattr(dataapi_module.requests, "get", lambda *a, **k: calls.append(1))
    client.fetch_meta("tbl")  # served from cache
    assert calls == []


def test_data_client_is_tolerant_on_errors(monkeypatch):
    client = OEPDataClient("token", get_config())

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(dataapi_module.requests, "get", boom)
    assert client.fetch_rows("tbl") is None
    assert client.metadata_card("tbl") is None
