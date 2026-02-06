"""Tutorials - Score - Example 4 - Scoring Models with ROUGE."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging

from scorebook import score
from scorebook.metrics.rouge1 import Rouge1
from scorebook.metrics.rougel import RougeL


def main() -> Any:
    """Score text generation predictions using ROUGE metrics.

    This example demonstrates how to score generated summaries
    against reference summaries using ROUGE-1 and ROUGE-L scores.
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

    # Score the predictions using multiple ROUGE metrics
    # You can evaluate multiple metrics in a single call
    results = score(
        items=model_predictions,
        metrics=[Rouge1(use_stemmer=True), RougeL(use_stemmer=True)],
        upload_results=False,  # Disable uploading for this example
    )

    print("\nResults:")
    pprint(results)

    # Display individual item scores
    print("\n\nIndividual ROUGE Scores:")
    for i, item_score in enumerate(results["item_results"]):
        print(f"\nItem {i+1}:")
        print(f"  ROUGE-1 F1: {item_score['rouge1']:.4f}")
        print(f"  ROUGE-L F1: {item_score['rougeL']:.4f}")

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="4-scoring_model_rouge", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "4-scoring_model_rouge_output.json")
