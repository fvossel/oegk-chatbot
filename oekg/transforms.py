"""Pure helpers for turning OEP SPARQL responses into DataFrames."""

from __future__ import annotations

from typing import Any

from pandas import DataFrame, notna


def convert_to_df(json_object: Any) -> DataFrame:
    """Convert a SPARQL JSON response from the OEP API into a DataFrame."""
    variables = json_object["head"]["vars"]
    rows = [
        {var: binding.get(var, {"value": None})["value"] for var in variables}
        for binding in json_object["results"]["bindings"]
    ]
    return DataFrame(rows)


def get_scenarios(df: DataFrame) -> list[str]:
    """Return the scenario column and its non-empty neighbours.

    Ensures the ``scenario`` column comes first. Returns an empty list when no
    scenario column is present.
    """
    first_row = df.iloc[0]

    scenario_idx = next(
        (
            i
            for i, val in enumerate(first_row)
            if isinstance(val, str) and "/ontology/oekg/scenario/" in val
        ),
        None,
    )
    if scenario_idx is None:
        return []

    cols = [df.columns[scenario_idx]]

    # Walk left while neighbouring cells are populated.
    i = scenario_idx - 1
    while i >= 0 and notna(first_row.iloc[i]):
        cols.insert(0, df.columns[i])
        i -= 1

    # Walk right while neighbouring cells are populated.
    i = scenario_idx + 1
    while i < len(df.columns) and notna(first_row.iloc[i]):
        cols.append(df.columns[i])
        i += 1

    # Guarantee the scenario column is first.
    scenario_col = df.columns[scenario_idx]
    if cols[0] != scenario_col:
        cols.remove(scenario_col)
        cols.insert(0, scenario_col)

    return cols
