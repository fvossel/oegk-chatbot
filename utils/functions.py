from typing import Any
from pandas import DataFrame, notna


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
    """Returns the scenario column and its neighboring columns on the same level.
       Ensures that 'scenario' column is first in the returned list.
       Returns an empty list if no scenario column is found."""
    
    # Find the index of the scenario column in the first row
    scenario_idx = None
    first_row = df.iloc[0]

    for i, val in enumerate(first_row):
        if isinstance(val, str) and "/ontology/oekg/scenario/" in val:
            scenario_idx = i
            break

    # If no scenario found, return empty list
    if scenario_idx is None:
        return []

    # Initialize columns list with scenario as the first element
    cols = [df.columns[scenario_idx]]

    # Scan to the left of scenario column for non-empty neighbor columns
    i = scenario_idx - 1
    while i >= 0:
        val = first_row.iloc[i]
        if notna(val):
            cols.insert(0, df.columns[i])  # insert before scenario
            i -= 1
        else:
            break

    # Scan to the right of scenario column for non-empty neighbor columns
    i = scenario_idx + 1
    while i < len(df.columns):
        val = first_row.iloc[i]
        if notna(val):
            cols.append(df.columns[i])
            i += 1
        else:
            break

    # Make sure scenario column is first in the final list
    if cols[0] != df.columns[scenario_idx]:
        cols.remove(df.columns[scenario_idx])
        cols.insert(0, df.columns[scenario_idx])

    return cols