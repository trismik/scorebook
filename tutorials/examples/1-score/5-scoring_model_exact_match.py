"""Tutorials - Score - Example 5 - Scoring Models with Exact Match."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import score
from scorebook.metrics.exactmatch import ExactMatch


def main() -> Any:
    """Score text predictions using Exact Match metric.

    This example demonstrates how to compare model outputs against
    reference labels using exact string matching with configurable
    preprocessing options.
    """

    # Prepare a list of items with model outputs and expected labels
    # Note: outputs may have different casing or extra whitespace
    model_predictions = [
        {"output": "Paris", "label": "Paris"},           # Exact match
        {"output": "LONDON", "label": "London"},         # Different case
        {"output": "  Berlin  ", "label": "Berlin"},     # Extra whitespace
        {"output": " NEW YORK ", "label": "new york"},   # Both case and whitespace
        {"output": "Tokyo", "label": "Kyoto"},           # No match
    ]

    print(f"Scoring {len(model_predictions)} predictions\n")

    # Score with default settings (case_insensitive=True, strip=True)
    print("Default settings (case_insensitive=True, strip=True):")
    results_default = score(
        items=model_predictions,
        metrics=ExactMatch(),
        upload_results=False,
    )
    pprint(results_default["aggregate_results"])
    print(f"Item matches: {[item['exact_match'] for item in results_default['item_results']]}")

    # Score with case-sensitive matching
    print("\nCase-sensitive matching (case_insensitive=False, strip=True):")
    results_case_sensitive = score(
        items=model_predictions,
        metrics=ExactMatch(case_insensitive=False),
        upload_results=False,
    )
    pprint(results_case_sensitive["aggregate_results"])
    print(f"Item matches: {[item['exact_match'] for item in results_case_sensitive['item_results']]}")

    # Score without stripping whitespace
    print("\nWithout stripping (case_insensitive=True, strip=False):")
    results_no_strip = score(
        items=model_predictions,
        metrics=ExactMatch(strip=False),
        upload_results=False,
    )
    pprint(results_no_strip["aggregate_results"])
    print(f"Item matches: {[item['exact_match'] for item in results_no_strip['item_results']]}")

    # Score with strict matching (no preprocessing)
    print("\nStrict matching (case_insensitive=False, strip=False):")
    results_strict = score(
        items=model_predictions,
        metrics=ExactMatch(case_insensitive=False, strip=False),
        upload_results=False,
    )
    pprint(results_strict["aggregate_results"])
    print(f"Item matches: {[item['exact_match'] for item in results_strict['item_results']]}")

    return results_default


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="5-scoring_model_exact_match", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "5-scoring_model_exact_match_output.json")