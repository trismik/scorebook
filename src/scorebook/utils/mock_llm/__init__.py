"""Mock LLM utilities for testing and demonstrations."""

import json
import random
from pathlib import Path
from typing import Any, List

# Load the mock data once at module initialization
_DATA_PATH = Path(__file__).parent / "data" / "mock_llm_data.json"
with open(_DATA_PATH, "r", encoding="utf-8") as f:
    _MOCK_DATA = json.load(f)


def mock_llm(inputs: List[Any], **hyperparameters: Any) -> List[str]:
    """Mock LLM that returns answers based on pre-recorded accuracy data."""

    results = []
    all_choices = ["A", "B", "C", "D", "E"]

    for item in inputs:
        item_id = item["id"]

        # Look up the item in our mock data
        if item_id not in _MOCK_DATA:
            # If item not found, return random answer
            results.append(random.choice(all_choices))
            continue

        item_data = _MOCK_DATA[item_id]
        correct_answer = item_data["answer"]
        was_accurate = item_data["accuracy"]

        if was_accurate:
            # Return the correct answer
            results.append(correct_answer)
        else:
            # Return a random incorrect answer
            incorrect_choices = [choice for choice in all_choices if choice != correct_answer]
            results.append(random.choice(incorrect_choices))

    return results
