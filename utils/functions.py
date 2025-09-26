from typing import Any
from pandas import DataFrame

def convert_to_df(json_object: Any) -> DataFrame:
    """Converts a response from the OEP-API into a pandas dataframe."""
    rows = []
    for item in json_object["results"]["bindings"]:
        row = {}
        for var in json_object["head"]["vars"]:
            row[var] = item.get(var, {"value": None})["value"]
        rows.append(row)
        
    return DataFrame(rows)


def get_scenarios(df: DataFrame):
    """Returns the scenario URIs if there are some in the dataframe."""
    first_row = df.iloc[0]
    for col, val in zip(df.columns, first_row):
        if isinstance(val, str) and "/ontology/oekg/scenario/" in val:
            return col
    return None