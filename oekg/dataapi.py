"""Read-only client for the OEP relational data REST API.

Lets the app go beyond graph metadata: when a result links to an OEP dataset
table, fetch its actual rows (a preview) and its OEMetadata (units, coverage,
license, sources, review badge). All calls are tolerant -- any failure returns
``None`` so the chat never breaks.

Endpoints (verified):
- GET {base}/tables/{table}/rows/?limit=N   -> JSON list of row dicts
- GET {base}/tables/{table}/meta/           -> OEMetadata (OEMetadata-2.0.x)
Public tables need no token; the same ``Authorization: Token`` header used for
SPARQL is sent when a token is configured.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from pandas import DataFrame

from oekg.config import AppConfig, get_config

logger = logging.getLogger(__name__)

# A resolved OEP dataset URL looks like .../database/tables/<table> or
# .../dataedit/view/<schema>/<table>; capture the table identifier.
_TABLE_URL_RE = re.compile(r"/(?:database/tables|dataedit/view/[^/]+)/([A-Za-z0-9_]+)")


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", "null", "None"):
            return value
    return None


def summarise_meta(meta: dict[str, Any], table: str = "") -> dict[str, Any]:
    """Extract the human-relevant fields from a full OEMetadata document."""
    resources = meta.get("resources") or []
    resource = resources[0] if resources else {}

    card: dict[str, Any] = {
        "table": table,
        "title": _first_nonempty(meta.get("title"), resource.get("title"), table),
        "description": _first_nonempty(meta.get("description"), resource.get("description")),
    }

    extent = (resource.get("spatial") or {}).get("extent") or {}
    card["region"] = _first_nonempty(extent.get("name"))
    card["resolution"] = _first_nonempty(extent.get("resolutionValue"))

    temporal = resource.get("temporal") or {}
    series = (temporal.get("timeseries") or [{}])[0]
    start, end = _first_nonempty(series.get("start")), _first_nonempty(series.get("end"))
    card["temporal"] = f"{start} – {end}" if start and end else _first_nonempty(temporal.get("referenceDate"))

    licenses = resource.get("licenses") or []
    if licenses:
        lic = licenses[0]
        card["license"] = {
            "name": _first_nonempty(lic.get("name"), lic.get("title")),
            "path": _first_nonempty(lic.get("path")),
            "attribution": _first_nonempty(lic.get("attribution")),
        }

    sources = resource.get("sources") or []
    card["sources"] = [
        {"title": s.get("title"), "path": _first_nonempty(s.get("path"))}
        for s in sources[:3]
        if isinstance(s, dict) and s.get("title")
    ]

    card["badge"] = _first_nonempty((resource.get("review") or {}).get("badge"))

    fields = (resource.get("schema") or {}).get("fields") or []
    card["units"] = {
        f.get("name"): f.get("unit")
        for f in fields
        if isinstance(f, dict) and _first_nonempty(f.get("unit"))
    }
    card["n_fields"] = len(fields)
    return card


class OEPDataClient:
    """Tolerant read client for OEP table rows + metadata."""

    def __init__(self, api_token: str | None, config: AppConfig | None = None) -> None:
        self._api_token = api_token
        self._config = config or get_config()
        self._meta_cache: dict[str, dict[str, Any] | None] = {}

    @staticmethod
    def table_from_url(url: str) -> str | None:
        """Extract the OEP table identifier from a resolved dataset URL."""
        if not isinstance(url, str):
            return None
        match = _TABLE_URL_RE.search(url)
        return match.group(1) if match else None

    def _get(self, path: str, params: dict | None = None) -> Any:
        headers = {"Authorization": f"Token {self._api_token}"} if self._api_token else {}
        try:
            response = requests.get(
                self._config.oep_api_base + path,
                params=params,
                headers=headers,
                timeout=self._config.sparql_timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception:  # noqa: BLE001 - data preview is best-effort
            logger.warning("OEP data API GET failed: %s", path, exc_info=True)
            return None

    def fetch_rows(self, table: str, limit: int = 50) -> DataFrame | None:
        """Return the first ``limit`` rows of ``table`` as a DataFrame (or None)."""
        data = self._get(f"/tables/{table}/rows/", params={"limit": limit})
        if not isinstance(data, list) or not data:
            return None
        return DataFrame(data)

    def fetch_meta(self, table: str) -> dict[str, Any] | None:
        """Return the raw OEMetadata document for ``table`` (memoised)."""
        if table in self._meta_cache:
            return self._meta_cache[table]
        data = self._get(f"/tables/{table}/meta/")
        meta = data if isinstance(data, dict) else None
        self._meta_cache[table] = meta
        return meta

    def metadata_card(self, table: str) -> dict[str, Any] | None:
        """Return the summarised metadata card for ``table`` (or None)."""
        meta = self.fetch_meta(table)
        return summarise_meta(meta, table) if meta else None
