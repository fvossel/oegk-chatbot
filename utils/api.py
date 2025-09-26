import requests
from typing import Any
from json import loads
from time import sleep
from openai import OpenAI
from utils.functions import convert_to_df, get_scenarios
from utils.gpt import request_sparql_query
from pandas import DataFrame


def run_sparql(api_key: str, query: str, retries: int = 3, delay: int = 2) -> Any:
    """Runs a SPARQL query against the Open Energy Platform Endpoint with retries"""
    HEADER = {"Authorization": f"Token {api_key}"}
    sparql_endpoint = "https://openenergyplatform.org/api/v0/oekg/sparql/"
    payload = {
        "query": query,
        "format": "json"
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url=sparql_endpoint, json=payload, headers=HEADER, timeout=30)
            r.raise_for_status()
            return loads(r.json())
        except Exception as e:
            if attempt == retries:
                return loads("""{
  "head": {
    "vars": [
      "nothing"
    ]
  },
  "results": {
    "bindings": []
  }
}""")
            sleep(delay)

def get_bundle_uri_and_label(api_key: str, scenario_uri: str) -> tuple[str, str]:
    """Extracts the URI and the label from corresponding scenario bundle of a scenario factsheet"""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT DISTINCT ?bundle ?bundleLabel
WHERE {
  ?bundle obo:BFO_0000051  """ + scenario_uri + """.
  ?bundle rdfs:label ?bundleLabel .
}
"""
    result = run_sparql(api_key, query)

    return result["results"]["bindings"][0]["bundle"]["value"], result["results"]["bindings"][0]["bundleLabel"]["value"]

def get_label_and_acronym(api_key: str,  scenario_uri: str) -> tuple[str, str]:
    """Extracts the label and the acronym from a given scenario factsheet"""
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/terms/>

SELECT ?scenario ?scenarioLabel ?scenarioAcronym
WHERE {
  VALUES ?scenario { """ + scenario_uri + """ }
  ?scenario rdfs:label ?scenarioLabel .
  ?scenario dc:acronym ?scenarioAcronym .
}
"""
    result = run_sparql(api_key, query)

    return result["results"]["bindings"][0]["scenarioLabel"]["value"], result["results"]["bindings"][0]["scenarioAcronym"]["value"]


def execute_sparql(sparql_query: str, nl_query: str, top_k_documents: list, client: OpenAI, system_prompt: str, oep_api_token: str) -> tuple[DataFrame, str]:
    """
    Execute a SPARQL query and returns the result as a DataFrame and the executed query text.
    Has multiple fallbacks if the query fails or returns no results.
    """

    full_query =f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oeo: <https://openenergyplatform.org/ontology/oeo/>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX oekg: <https://openenergyplatform.org/ontology/oekg/>
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX XSD: <http://www.w3.org/2001/XMLSchema#>


""" + sparql_query
   
    results_df = convert_to_df(run_sparql(oep_api_token, full_query))

    if results_df.empty:
        try:
            full_query = full_query.replace(sparql_query, "")
            sparql_query = request_sparql_query(nl_query + f"The privious SPARQL-Query has not returned any results: \n{sparql_query}\n Please check the query again and also check if your constraints were to harsh. But also take into account that maybe you have to search for the pattern in the subclasses labels of the mentioned entity. So try again and maybe use some fancy tricks.", top_k_documents, client, system_prompt)
            full_query = full_query + sparql_query
            results_df = convert_to_df(run_sparql(oep_api_token, full_query))
        except:
            results_df = DataFrame()
  
    if not results_df.empty:
        scenarios = get_scenarios(results_df)
        if scenarios:
            bundle_to_scenarios = {}
            for scenario in results_df[scenarios]:
                uri = "<" + scenario + ">"
                bundle_uri, bundle_label = get_bundle_uri_and_label(oep_api_token, uri)
                scenario_label, sneario_acronym = get_label_and_acronym(oep_api_token, uri)
                if bundle_uri not in bundle_to_scenarios:
                    bundle_to_scenarios[bundle_uri] = {"label": bundle_label, "scenarios": []}
                
                bundle_to_scenarios[bundle_uri]["scenarios"].append({"acronym": sneario_acronym, "label": scenario_label})
            
            grouped_rows = []
            for uri, data in bundle_to_scenarios.items():
                scenarios_sorted = sorted(
                    data["scenarios"],
                    key=lambda s: (s["acronym"], s["label"])
                )
                for i, scenario in enumerate(scenarios_sorted):
                    grouped_rows.append({
                        "Bundle URI": uri if i == 0 else "",
                        "Bundle Label": data["label"] if i == 0 else "",
                        "Scenario Acronym": scenario["acronym"],
                        "Scenario Label": scenario["label"]
                    })
            results_df = DataFrame(grouped_rows)


    return results_df, f"Generated Query:\n```sparql\n{full_query}\n```\n\n"