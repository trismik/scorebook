"""Utility functions for transforming and manipulating data structures."""

from itertools import product


def expand_dict(data: dict) -> list[dict]:
    """Expand a dictionary with list values into multiple dictionaries.

    Takes a dictionary that may contain list values and expands it into a list of dictionaries,
    where each dictionary represents one possible combination of values from the lists.
    Non-list values remain constant across all generated dictionaries.

    Args:
        data: A dictionary potentially containing list values to be expanded

    Returns:
        A list of dictionaries representing all possible combinations of the input values
    """
    fixed = {k: v for k, v in data.items() if not isinstance(v, list)}
    expandables = {k: v for k, v in data.items() if isinstance(v, list)}

    keys, values = zip(*expandables.items()) if expandables else ([], [])
    combinations = product(*values) if values else [()]

    return [{**fixed, **dict(zip(keys, combo))} for combo in combinations]
