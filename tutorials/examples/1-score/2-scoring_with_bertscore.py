"""Tutorials - Score - Example 2 - Scoring with BertScore."""

import sys
from pathlib import Path
from pprint import pprint
from typing import Any

from scorebook.metrics.bertscore import BertScore

from tutorials.utils import save_results_to_json, setup_logging
from scorebook import score


def main() -> Any:
    """Score pre-computed model predictions using Scorebook.

    This example demonstrates how to score generated model predictions.
    """

    # Prepare a list of items with generated summaries and reference summaries
    model_predictions = [
        {
            "output": "A woman donated her kidney to a stranger. This sparked a chain of six kidney transplants.",
            "label": "Zully Broussard decided to give a kidney to a stranger. A new computer program helped her donation spur transplants for six kidney patients.",
        },
        {
            "output": "Scientists discovered a new species of frog in the Amazon rainforest. The frog has unique markings that distinguish it from other species.",
            "label": "A new frog species with distinctive blue and yellow stripes was found in the Amazon. Researchers say this discovery highlights the biodiversity of the region.",
        },
        {
            "output": "The technology company released its quarterly earnings report showing strong growth.",
            "label": "Tech giant announces record quarterly revenue driven by cloud services and AI products.",
        },
    ]

    # Score the predictions against labels using the BertScore metric
    results = score(
        items=model_predictions,
        metrics=BertScore,
        upload_results=False,  # Disable uploading for this example
    )

    print("\nResults:")
    pprint(results)

    return results


if __name__ == "__main__":

    log_file = setup_logging(experiment_id="2-scoring_model_bertscore", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "2-scoring_model_bertscore_output.json")

