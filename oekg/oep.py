"""Open Energy Platform SPARQL client and result post-processing.

Encapsulates all interaction with the OEP SPARQL endpoint: running queries
(with retries), resolving internal URIs to their public counterparts, grouping
scenario rows by bundle, and assembling a fully-prefixed query for execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from json import loads
from time import sleep
from typing import Any, Callable

import requests
from pandas import DataFrame

from oekg.config import AppConfig, get_config
from oekg.transforms import convert_to_df, get_scenarios
from oekg.validation import validate_syntax

logger = logging.getLogger(__name__)

# Prefixes prepended to every generated query.
_QUERY_PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oeo: <https://openenergyplatform.org/ontology/oeo/>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX oekg: <https://openenergyplatform.org/ontology/oekg/>
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX XSD: <http://www.w3.org/2001/XMLSchema#>


"""

# Returned when a query ultimately fails so callers always get a valid shape.
_EMPTY_RESULT: dict[str, Any] = {"head": {"vars": ["nothing"]}, "results": {"bindings": []}}


class OEPError(Exception):
    """Raised by :meth:`OEPClient.run_sparql` in strict mode when a query errors.

    Distinguishes a genuine endpoint error (bad query, server failure) from a
    syntactically-valid query that simply returned no rows, so the pipeline can
    feed the error back to the LLM for a repair attempt instead of silently
    showing "No results found.".
    """


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running a SPARQL query against the OEP endpoint."""

    results_df: DataFrame
    full_query: str  # prefixed query that was executed (for display)
    sparql_used: str  # the bare query that was actually executed
    repaired: bool = False  # True if the LLM had to repair an endpoint error


class OEPClient:
    """Client for the OEKG SPARQL endpoint of the Open Energy Platform."""

    def __init__(self, api_token: str | None, config: AppConfig | None = None) -> None:
        self._api_token = api_token
        self._config = config or get_config()
        # Memoise URI resolution so a URI repeated across cells is only looked
        # up once (the heaviest source of redundant endpoint round-trips).
        self._uri_cache: dict[str, str] = {}

    # -- low-level query execution ---------------------------------------
    def run_sparql(self, query: str, strict: bool = False) -> dict[str, Any]:
        """Run a SPARQL query with retries.

        In tolerant mode (default) any failure yields :data:`_EMPTY_RESULT` so
        callers can keep going. In ``strict`` mode the final failure (or a 4xx
        client error, which retrying cannot fix) raises :class:`OEPError`.
        """
        headers = {"Authorization": f"Token {self._api_token}"}
        payload = {"query": query, "format": "json"}

        retries = self._config.sparql_retries
        last_error = "no attempt was made"
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    url=self._config.sparql_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._config.sparql_timeout,
                )
                response.raise_for_status()
                # The OEP endpoint returns double-encoded JSON: the body is a
                # JSON string whose content is the actual JSON object. Parse once
                # via .json(); if that still yields a string, parse it again.
                parsed = response.json()
                return loads(parsed) if isinstance(parsed, str) else parsed
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                body = exc.response.text[:300] if exc.response is not None else str(exc)
                last_error = f"HTTP {status}: {body}"
                logger.warning("SPARQL request failed (attempt %s/%s): %s", attempt, retries, last_error)
                # A client error means the query itself is bad; retrying is futile.
                if status is not None and 400 <= status < 500:
                    if strict:
                        raise OEPError(last_error) from exc
                    return dict(_EMPTY_RESULT)
            except Exception as exc:  # noqa: BLE001 - network/parse errors are retried
                last_error = str(exc)
                logger.warning("SPARQL request failed (attempt %s/%s): %s", attempt, retries, last_error)
            if attempt < retries:
                sleep(self._config.sparql_retry_delay)

        if strict:
            raise OEPError(last_error)
        return dict(_EMPTY_RESULT)

    @staticmethod
    def build_full_query(sparql_query: str) -> str:
        """Prepend the standard PREFIX block to a generated query."""
        return _QUERY_PREFIXES + sparql_query

    def find_near_miss_labels(self, mentions: list[str], limit: int) -> list[str]:
        """Fetch real labels whose text contains a token of any mention.

        Used to enrich the empty-result retry with labels that actually exist in
        the graph. One tolerant query; returns ``[]`` on no tokens / no results.
        """
        if not mentions or limit <= 0:
            return []
        min_len = self._config.near_miss_min_token_len
        tokens: list[str] = []
        for mention in mentions:
            qualifying = [t.strip().lower() for t in mention.split() if len(t.strip()) >= min_len]
            if qualifying:
                tokens.extend(qualifying)
            else:
                whole = mention.strip().lower()
                if len(whole) >= min_len:
                    tokens.append(whole)
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return []

        escaped = [t.replace("\\", "\\\\").replace('"', '\\"') for t in tokens]
        filters = " || ".join(f'CONTAINS(LCASE(STR(?label)), "{t}")' for t in escaped)
        query = (
            "SELECT DISTINCT ?label WHERE { ?s rdfs:label ?label . FILTER("
            + filters
            + f") }} LIMIT {int(limit)}"
        )
        df = convert_to_df(self.run_sparql(self.build_full_query(query)))
        if df.empty or "label" not in df.columns:
            return []
        labels = [v for v in df["label"].tolist() if isinstance(v, str) and v.strip()]
        return list(dict.fromkeys(labels))

    # -- URI resolution ---------------------------------------------------
    def resolve_uri(self, uri: str) -> str:
        """Resolve internal URIs to their public OEP resources (memoised)."""
        if not isinstance(uri, str):
            return uri
        if uri in self._uri_cache:
            return self._uri_cache[uri]
        resolved = self._resolve_uri_uncached(uri)
        self._uri_cache[uri] = resolved
        return resolved

    def _resolve_uri_uncached(self, uri: str) -> str:
        # Resolve datasets.
        if uri.startswith(
            "https://openenergyplatform.org/ontology/oekg/input_datasets/"
        ) or uri.startswith("https://openenergyplatform.org/ontology/oekg/output_datasets/"):
            query = (
                """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?literal
                WHERE {
                <"""
                + uri
                + """>
                <https://openenergyplatform.org/ontology/oeo/OEO_00390094> ?literal .
                }"""
            )
            results = self.run_sparql(query)
            bindings = results["results"]["bindings"]
            if not bindings:
                return uri
            literal = bindings[0]["literal"]["value"]
            if not literal.startswith("https://"):
                literal = "https://openenergyplatform.org" + literal
            return literal

        # Resolve scenario bundles.
        if uri.startswith("https://openenergyplatform.org/ontology/oekg"):
            query = (
                """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT ?type ?typeLabel
                WHERE {
                <"""
                + uri
                + """> rdf:type ?type .
                OPTIONAL { ?type rdfs:label ?typeLabel }
                FILTER (?type = <https://openenergyplatform.org/ontology/oeo/OEO_00020227>)
                }"""
            )
            results = self.run_sparql(query)
            bindings = results["results"]["bindings"]
            # ``typeLabel`` is OPTIONAL, so use .get to avoid a KeyError when unbound.
            if bindings and bindings[0].get("typeLabel", {}).get("value") == "scenario bundle":
                return uri.replace(
                    "https://openenergyplatform.org/ontology/oekg/",
                    "https://openenergyplatform.org/scenario-bundles/id/",
                )

        return uri

    # -- scenario bundling ------------------------------------------------
    def _bundles_for_scenarios(self, scenario_uris: list[str]) -> dict[str, tuple[str, str]]:
        """Map each scenario URI to its ``(bundle_uri, bundle_label)`` in one query."""
        if not scenario_uris:
            return {}
        query = (
            """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT DISTINCT ?scenario ?bundle ?bundleLabel
WHERE {
  VALUES ?scenario { """
            + " ".join(scenario_uris)
            + """ }
  ?bundle obo:BFO_0000051 ?scenario .
  ?bundle rdfs:label ?bundleLabel .
}
"""
        )
        result = self.run_sparql(query)
        mapping: dict[str, tuple[str, str]] = {}
        for binding in result["results"]["bindings"]:
            scenario = binding.get("scenario", {}).get("value")
            if scenario is None or scenario in mapping:
                continue
            mapping[scenario] = (
                binding.get("bundle", {}).get("value", ""),
                binding.get("bundleLabel", {}).get("value", ""),
            )
        return mapping

    def _labels_for_scenarios(self, scenario_uris: list[str]) -> dict[str, tuple[str, str]]:
        """Map each scenario URI to its ``(label, acronym)`` in one query."""
        if not scenario_uris:
            return {}
        query = (
            """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/terms/>

SELECT DISTINCT ?scenario ?scenarioLabel ?scenarioAcronym
WHERE {
  VALUES ?scenario { """
            + " ".join(scenario_uris)
            + """ }
  ?scenario rdfs:label ?scenarioLabel .
  OPTIONAL { ?scenario dc:acronym ?scenarioAcronym . }
}
"""
        )
        result = self.run_sparql(query)
        mapping: dict[str, tuple[str, str]] = {}
        for binding in result["results"]["bindings"]:
            scenario = binding.get("scenario", {}).get("value")
            if scenario is None or scenario in mapping:
                continue
            mapping[scenario] = (
                binding.get("scenarioLabel", {}).get("value", ""),
                binding.get("scenarioAcronym", {}).get("value", ""),
            )
        return mapping

    def _bundle_scenarios_by_bundle(
        self, results_df: DataFrame, scenario_cols: list[str]
    ) -> list[dict[str, Any]]:
        scenario_values = [
            v for v in results_df[scenario_cols[0]] if isinstance(v, str) and v
        ]
        unique_uris = list(dict.fromkeys("<" + v + ">" for v in scenario_values))
        bundle_map = self._bundles_for_scenarios(unique_uris)
        label_map = self._labels_for_scenarios(unique_uris)

        bundle_to_scenarios: dict[str, dict[str, Any]] = {}
        for _, row in results_df[scenario_cols].iterrows():
            scenario_uri = row[scenario_cols[0]]
            bundle_uri, bundle_label = bundle_map.get(scenario_uri, (scenario_uri, ""))
            scenario_label, scenario_acronym = label_map.get(scenario_uri, ("", ""))

            bundle = bundle_to_scenarios.setdefault(
                bundle_uri, {"label": bundle_label, "scenarios": []}
            )

            scenario_entry = {"acronym": scenario_acronym, "label": scenario_label}
            scenario_entry.update({col: row[col] for col in scenario_cols[1:]})
            bundle["scenarios"].append(scenario_entry)

        grouped_rows: list[dict[str, Any]] = []
        for uri, data in bundle_to_scenarios.items():
            scenarios_sorted = sorted(
                data["scenarios"], key=lambda s: (s["acronym"], s["label"])
            )
            for i, scenario in enumerate(scenarios_sorted):
                row = {
                    "Bundle URI": uri if i == 0 else "",
                    "Bundle Label": data["label"] if i == 0 else "",
                    "Scenario Acronym": scenario["acronym"],
                    "Scenario Label": scenario["label"],
                }
                row.update(
                    {k: v for k, v in scenario.items() if k not in ("acronym", "label")}
                )
                grouped_rows.append(row)
        return grouped_rows

    # -- high-level orchestration ----------------------------------------
    def execute(
        self,
        sparql_query: str,
        on_empty: Callable[[str], str] | None = None,
        on_error: Callable[[str, str], str] | None = None,
    ) -> ExecutionResult:
        """Execute a query and return an :class:`ExecutionResult`.

        ``on_error`` receives ``(query, error_message)`` when the endpoint
        rejects the query and returns a revised query; it is retried up to
        ``sparql_repair_rounds`` times (LLM self-correction).

        ``on_empty`` receives the query and returns a revised one; it is invoked
        once when an otherwise-successful query yields no rows (constraint
        relaxation).
        """
        sparql_used = sparql_query
        rounds = max(0, self._config.sparql_repair_rounds)
        repaired = False

        raw: dict[str, Any] = dict(_EMPTY_RESULT)
        for attempt in range(rounds + 1):
            full_query = self.build_full_query(sparql_used)
            # Advisory local syntax pre-flight (first attempt only): repair a
            # malformed query inline (no wasted endpoint round-trip) and execute
            # the repaired query in this same iteration. Never blocks execution
            # -- rdflib may reject queries the OEP would accept.
            if attempt == 0 and on_error is not None and self._config.syntax_validation_enabled:
                syntax_error = validate_syntax(full_query)
                if syntax_error is not None:
                    logger.info("local SPARQL syntax check flagged: %s", syntax_error)
                    try:
                        sparql_used = on_error(sparql_used, syntax_error)
                        repaired = True
                        full_query = self.build_full_query(sparql_used)
                    except Exception:  # noqa: BLE001 - advisory; execute the original
                        logger.exception("local syntax repair via on_error failed")
            try:
                raw = self.run_sparql(full_query, strict=True)
                break
            except OEPError as err:
                if on_error is None or attempt == rounds:
                    logger.warning("query failed%s: %s", " after repair" if attempt else "", err)
                    raw = dict(_EMPTY_RESULT)
                    break
                try:
                    sparql_used = on_error(sparql_used, str(err))
                    repaired = True
                except Exception:  # noqa: BLE001 - fall back to empty results
                    logger.exception("SPARQL repair via on_error callback failed")
                    raw = dict(_EMPTY_RESULT)
                    break

        results_df = convert_to_df(raw)

        if results_df.empty and on_empty is not None:
            try:
                revised = on_empty(sparql_used)
                sparql_used = revised
                results_df = convert_to_df(self.run_sparql(self.build_full_query(revised)))
            except Exception:  # noqa: BLE001 - fall back to empty results
                logger.exception("SPARQL retry via on_empty callback failed")
                results_df = DataFrame()

        if not results_df.empty:
            scenario_cols = get_scenarios(results_df)
            if scenario_cols:
                results_df = DataFrame(
                    self._bundle_scenarios_by_bundle(results_df, scenario_cols)
                )

        results_df = results_df.map(self.resolve_uri)
        # Drop duplicate column *labels* only (SELECT vars are unique, so this is
        # a safety net); never drop columns just because their values coincide.
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        return ExecutionResult(
            results_df=results_df,
            full_query=self.build_full_query(sparql_used),
            sparql_used=sparql_used,
            repaired=repaired,
        )
