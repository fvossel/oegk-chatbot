from openai import OpenAI
from json import dumps


def request_sparql_query(query: str, documents: list, client: OpenAI, system_prompt: str) ->str:
    """
    Generate a SPARQL query from a natural language request and context documents using the LLM.
    """   
    user_prompt ="Request: " + query + "\n\nContext with classes and their allowed relations and properties:\n" + dumps(documents, ensure_ascii=False, indent=2)
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.create(
        model="gpt-5",
        input=messages,
        reasoning={"effort": "minimal"}
    )

    sparql = response.output_text
    return sparql

def summarise_sparql_results(sparql_results: str, nl_query: str, client: OpenAI, system_prompt: str) ->str:
    """
    Get a text summary of the SPARQL query results using the LLM.
    """
    try:   
        messages = [
            {"role": "developer", "content": system_prompt},
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

def get_query(nl_query: str, top_k_documents: list, client: OpenAI, system_prompt: str) -> str:
    try:
        sparql = request_sparql_query(nl_query, top_k_documents, client, system_prompt)
        return sparql
    except Exception as e:
        return str(e)