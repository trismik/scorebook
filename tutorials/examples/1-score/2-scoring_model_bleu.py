"""Tutorials - Score - Example 2 - Scoring Models with BLEU."""

from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv

from tutorials.utils import save_results_to_json, setup_logging
from scorebook import score
from scorebook.metrics.bleu import BLEU


def main() -> Any:
    """Score pre-computed model predictions using Scorebook.

    This example demonstrates how to score generated model predictions.
    """

    # Prepare a list of items with generated outputs and labels
    model_predictions = [
        {"output": "28-jähriger Koch wurde in San Francisco Mall entdeckt.", "label": "28-jähriger Koch in San Francisco Mall tot aufgefunden"},
        {"output": "Ein 28-jähriger Koch, der kürzlich nach San Francisco gezogen war, wurde in der Treppe eines lokalen Einkaufszentrums dieser Woche ermordet.", "label": "Ein 28-jähriger Koch, der vor kurzem nach San Francisco gezogen ist, wurde im Treppenhaus eines örtlichen Einkaufzentrums tot aufgefunden."},
        {"output": 'Der Bruder des Opfers sagt, er könne sich nicht vorstellen, wer ihm schaden wolle, und sagt: "Die Dinge waren endlich gut für ihn."', "label": 'Der Bruder des Opfers sagte aus, dass er sich niemanden vorstellen kann, der ihm schaden wollen würde, "Endlich ging es bei ihm wieder bergauf."'},
    ]

    # Score the predictions against labels using the accuracy metric
    results = score(
        items=model_predictions,
        metrics=BLEU(compact=False),
        upload_results=False,  # Disable uploading for this example
    )

    print("\nResults:")
    pprint(results)

    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = setup_logging(experiment_id="2-scoring_model_bleu", base_dir=Path(__file__).parent)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_dict = main()
    save_results_to_json(results_dict, output_dir, "2-scoring_model_bleu_output.json")
