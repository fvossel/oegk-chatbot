from faiss import Index, read_index
from typing import Any
from json import load
from os import getenv
from openai import OpenAI
import numpy as np

def load_data() -> tuple[str, Index, dict[str, Any], dict[str, Any], str, str]:
    """Loads the necessary data for the RAG system including documents and prompts."""
    openai_api_key = getenv("OPENAI_API_KEY")

    with open("summary_system_prompt.txt", "r") as f:
        summary_system_prompt = f.read()

    with open("sparql_system_prompt.txt", "r") as f:
        sparql_system_prompt = f.read()

    # Load ontology resources
    with open("classes.json", "r", encoding="utf-8") as f:
        classes_list = load(f)

    with open("relations_final.json", "r", encoding="utf-8") as f:
        relations_list = load(f)

    with open("ids.json", "r", encoding="utf-8") as f:
        ids = load(f)

    # Load vector store for semantic retrieval
    faiss_index = read_index("vector_store.faiss")
    documents_dict = {doc["class"]: doc for doc in classes_list}
    documents_dict.update({rel["uri"]: rel for rel in relations_list})

    return openai_api_key, faiss_index, documents_dict, ids, sparql_system_prompt, summary_system_prompt

def retrieve_top_k_similar_documents(nl_query: str, client: OpenAI, faiss_index, documents_dict: dict[str,str], ids, k: int = 5) -> list:
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