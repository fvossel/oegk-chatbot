import streamlit as st
from openai import OpenAI
from utils.api import execute_sparql
from utils.load_rag_data import retrieve_top_k_similar_documents
from utils.gpt import summarise_sparql_results, get_query



def call_rag_pipeline(nl_query : str, streamlit_module: st, openai_api_key: str, faiss_index, documents_dict, ids, sparql_system_prompt: str, summary_system_prompt: str, oep_api_token: str) -> str:
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
    client = OpenAI(api_key=openai_api_key)

    with streamlit_module.spinner("Retrieving context information..."):
        top_k_documents = retrieve_top_k_similar_documents(nl_query, client, faiss_index, documents_dict, ids, k=10)

    with streamlit_module.spinner("Generating SPARQL query..."):
        sparql_query = get_query(nl_query, top_k_documents, client, sparql_system_prompt)

    if "<bot-information>" in sparql_query:
        client.close()  # Make sure OpenAI client is closed if possible
        return sparql_query[sparql_query.find("<bot-information>") + len("<bot-information>"):]
    else:
        with streamlit_module.spinner("Extracting requested information..."):
            df, final_query = execute_sparql(sparql_query, nl_query, top_k_documents, client, sparql_system_prompt, oep_api_token)
            
        with streamlit_module.spinner("Finalizing result formatting..."):
            sparql_results = "Query Results:\n" + (df.to_markdown(index=False) if not df.empty else "No results found.")
            summary = "\n\nSummary of results:\n" + (summarise_sparql_results(sparql_results, nl_query, client, summary_system_prompt) if "No results found." not in sparql_results and len(sparql_results) < 10000 else "")

        client.close()  # Make sure OpenAI client is closed if possible
        return final_query + sparql_results + summary


