import json
import random
from pathlib import Path
from typing import Any, List


def mock_llm(inputs: List[Any], **hyperparameters: Any) -> List[str]:
    """
    Mock LLM that returns answers based on pre-recorded accuracy data.

    Uses temp_0.7 results from mock_llm_data.json:
    - If accuracy was true for that item: return the correct answer
    - If accuracy was false for that item: return a random incorrect answer

    Args:
        inputs: List of items, each containing an "id" field
        **hyperparameters: Additional parameters (currently unused)

    Returns:
        List of answer letters (A, B, C, D, or E)
    """
    # Load the mock data
    data_path = Path(__file__).parent / "mock_llm_data" / "mock_llm_data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        mock_data = json.load(f)

    results = []
    all_choices = ['A', 'B', 'C', 'D', 'E']

    for item in inputs:
        item_id = item["id"]

        # Look up the item in our mock data
        if item_id not in mock_data:
            # If item not found, return random answer
            results.append(random.choice(all_choices))
            continue

        item_data = mock_data[item_id]
        correct_answer = item_data["answer"]
        was_accurate = item_data["temp_0.7"]

        if was_accurate:
            # Return the correct answer
            results.append(correct_answer)
        else:
            # Return a random incorrect answer
            incorrect_choices = [choice for choice in all_choices if choice != correct_answer]
            results.append(random.choice(incorrect_choices))

    return results