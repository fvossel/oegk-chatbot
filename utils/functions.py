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
       Ensures that 'scenario' column is first in the returned list."""
    first_row = df.iloc[0]

    scenario_idx = None
    for i, val in enumerate(first_row):
        if isinstance(val, str) and "/ontology/oekg/scenario/" in val:
            scenario_idx = i
            break

    if scenario_idx is None:
        return []

    cols = []

    # Scan to the left from scenario_idx
    i = scenario_idx - 1
    while i >= 0:
        val = first_row.iloc[i]
        if notna(val):
            cols.insert(0, df.columns[i])
            i -= 1
        else:
            break

    # Add the scenario column first
    cols.append(df.columns[scenario_idx])

    # Scan to the right from scenario_idx
    i = scenario_idx + 1
    while i < len(df.columns):
        val = first_row.iloc[i]
        if notna(val):
            cols.append(df.columns[i])
            i += 1
        else:
            break

    return cols