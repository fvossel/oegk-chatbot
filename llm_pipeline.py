import json
import os
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from openai import OpenAI
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS

# Load API key and system prompts
API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_SYSTEM_PROMPT = ""
SPARQL_SYSTEM_PROMPT = ""


with open("summary_system_prompt.txt", "r") as f:
    SUMMARY_SYSTEM_PROMPT = f.read()

with open("sparql_system_prompt.txt", "r") as f:
    SPARQL_SYSTEM_PROMPT = f.read()

# Load ontology resources
with open("classes.json", "r", encoding="utf-8") as f:
    classes_list = json.load(f)

with open("relations_final.json", "r", encoding="utf-8") as f:
    relations_list = json.load(f)

with open("ids.json", "r", encoding="utf-8") as f:
    ids = json.load(f)

# Load vector store for semantic retrieval
faiss_index = faiss.read_index("vector_store.faiss")
documents_dict = {doc["class"]: doc for doc in classes_list}
documents_dict.update({rel["uri"]: rel for rel in relations_list})

def shorten_uri(uri: str) -> str:
    """
    Shortens full URIs to prefixed forms for readability.
    """
    prefixes = [
        ("https://openenergyplatform.org/ontology/oeo/", "oeo:"),
        ("https://openenergyplatform.org/ontology/oekg/", "oekg:"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
        ("http://purl.org/dc/terms/", "dc:"),
        ("http://purl.obolibrary.org/obo/", "obo:"),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
        ("http://www.w3.org/2001/XMLSchema#", "XSD:")
    ]
    for full, prefix in prefixes:
        if uri.startswith(full):
            return uri.replace(full, prefix)
    return uri

def retrieve_top_k_similar_documents(nl_query: str, client: OpenAI, k: int = 5) -> list:
    """
    Retrieve top-k documents most semantically similar to the natural language query using embeddings.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=nl_query
    )
    query_embedding = response.data[0].embedding

    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, k)
    retrieved_docs = []
    for idx in indices[0]:
        retrieved_docs.append(documents_dict[ids[idx]])
    return retrieved_docs


def request_sparql_query(query: str, documents: list, client: OpenAI) ->str:
    """
    Generate a SPARQL query from a natural language request and context documents using the LLM.
    """   
    user_prompt ="Request: " + query + "\n\nContext with classes and their allowed relations and properties:\n" + json.dumps(documents, ensure_ascii=False, indent=2)
    messages = [
        {"role": "developer", "content": SPARQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.create(
        model="gpt-5",
        input=messages,
        reasoning={"effort": "minimal"}
    )

    sparql = response.output_text
    return sparql

def summarise_sparql_results(sparql_results: str, nl_query: str, client: OpenAI) ->str:
    """
    Get a text summary of the SPARQL query results using the LLM.
    """
    try:   
        messages = [
            {"role": "developer", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "The users question:" + nl_query + "\nThe formatted results from the SPARQL query:" + sparql_results}
        ]
        
        response = client.responses.create(
            model="gpt-5-nano",
            input=messages,
        )

        summary = response.output_text
    except:
        summary = ""
    return summary




def get_query(nl_query: str, top_k_documents: list, client: OpenAI) ->str:
    """
    Generate a SPARQL query with error fallback. Returns query or an error message.
    """
    error = ""
    try:
        return request_sparql_query(nl_query, top_k_documents, client)
    except Exception as e:
        error = str(e)
        return request_sparql_query(nl_query, top_k_documents, client)
    finally:
        return error
        

def execute_sparql(sparql_query: str, nl_query: str, top_k_documents: list, g: Graph, client: OpenAI) -> tuple[pd.DataFrame, str]:
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

{sparql_query}
"""
    results = None
    try:
        # 3. Execute the SPARQL query
        results = g.query(full_query)
    except Exception as e:
        try:
            full_query = full_query.replace(sparql_query, "")
            sparql_query = request_sparql_query(nl_query + f"The privious SPARQL-Query was not correct:\n{sparql_query}\n The following error occured:\n{e}", top_k_documents, client)
            full_query = full_query + sparql_query
            results = g.query(full_query)
        except:
            results = None

    if not results:
        try:
            full_query = full_query.replace(sparql_query, "")
            sparql_query = request_sparql_query(nl_query + f"The privious SPARQL-Query has not returned any results: \n{sparql_query}\n Please check the query again and also check if your constraints were to harsh. But also take into account that maybe you have to search for the pattern in the subclasses labels of the mentioned entity. So try again and maybe use some fancy tricks.", top_k_documents, client)
            full_query = full_query + sparql_query
            results = g.query(full_query)
        except:
            results = None


        
    # Namespaces
    oeo = Namespace("https://openenergyplatform.org/ontology/oeo/")
    dc = Namespace("http://purl.org/dc/terms/")
    obo = Namespace("http://purl.obolibrary.org/obo/")
    oekg_prefix = "https://openenergyplatform.org/ontology/oekg/"
    bundle_prefix = "https://openenergyplatform.org/scenario-bundles/id/"

    rows = []
    bundle_to_scenarios = {}  # Map: bundle_uri -> {label: str, scenarios: list}

    if results:
        for row in results:
            mapped_row = {}
            for i, var in enumerate(results.vars):
                val = row[i]

                # Check if value is a scenario factsheet
                if (val, RDF.type, oeo.OEO_00000365) in g:
                    scenario_label = g.value(val, RDFS.label)
                    scenario_acronym = g.value(val, dc.acronym)
                    for bundle in g.subjects(obo.BFO_0000051, val):
                        if (bundle, RDF.type, oeo.OEO_00020227) in g:
                            bundle_label = g.value(bundle, RDFS.label)
                            bundle_uri = str(bundle)
                            if bundle_uri not in bundle_to_scenarios:
                                bundle_to_scenarios[bundle_uri] = {
                                    "label": str(bundle_label) if bundle_label else "",
                                    "scenarios": []
                                }
                            if scenario_label and scenario_acronym:
                                bundle_to_scenarios[bundle_uri]["scenarios"].append({
                                    "acronym": str(scenario_acronym),
                                    "label": str(scenario_label)
                                })
                mapped_row[str(var)] = (
                    bundle_prefix + str(val).rsplit("/", 1)[1]
                    if (val, RDF.type, oeo.OEO_00020227) in g and str(val).startswith(oekg_prefix)
                    else str(val)
                )
            rows.append(mapped_row)

    # Prepare output DataFrame
    if bundle_to_scenarios:
        grouped_rows = []
        for uri, data in bundle_to_scenarios.items():
            scenarios_sorted = sorted(
                data["scenarios"],
                key=lambda s: (s["acronym"], s["label"])
            )
            for i, scenario in enumerate(scenarios_sorted):
                grouped_rows.append({
                    "Bundle URI": bundle_prefix + uri.rsplit("/", 1)[1] if i == 0 else "",
                    "Bundle Label": data["label"] if i == 0 else "",
                    "Scenario Acronym": scenario["acronym"],
                    "Scenario Label": scenario["label"]
                })
        df = pd.DataFrame(grouped_rows)
    else:
        df = pd.DataFrame(rows)

    return df, f"Generated Query:\n```sparql\n{full_query}\n```\n\n"


def call_rag_pipeline(nl_query : str, streamlit_module: st, graph: Graph) -> str:
    """
    End-to-end pipeline: NL→Retrieve→SPARQL→Execute→Summarize.
    Interacts with Streamlit for UI feedback.
    Returns results markdown for display in app.
    """
    history_dialogue = ""
    for role, msg in st.session_state.chat_history:
        prefix = "User:" if role == "user" else "Assistant:"
        history_dialogue += f"{prefix} {msg}\n"
    nl_query = history_dialogue + f"User: {nl_query}\n"

    client = OpenAI(api_key=API_KEY)
    with streamlit_module.spinner("Retrieving context information..."):
        top_k_documents = retrieve_top_k_similar_documents(nl_query, client, k=10)

    with streamlit_module.spinner("Generating SPARQL query..."):
        sparql_query = get_query(nl_query, top_k_documents, client)

    with streamlit_module.spinner("Extracting requested information..."):
        df, final_query = execute_sparql(sparql_query, nl_query, top_k_documents, graph, client)
        
    with streamlit_module.spinner("Finalizing result formatting..."):
        sparql_results = "Query Results:\n" + (df.to_markdown(index=False) if not df.empty else "No results found.")
        summary = "\n\nSummary of results:\n" + (summarise_sparql_results(sparql_results, nl_query, client) if "No results found." not in sparql_results and len(sparql_results) < 10000 else "")

    client.close()  # Make sure OpenAI client is closed if possible
    return final_query + sparql_results + summary


