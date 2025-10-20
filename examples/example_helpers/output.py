"""
Utility functions for saving Scorebook evaluation results.

This module provides common helper functions used across multiple Scorebook examples
for saving evaluation results to files.
"""

import json
from pathlib import Path
from typing import Any


def save_results_to_json(results: Any, output_dir: Path, filename: str) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: The evaluation results to save
        output_dir: Directory to save the file in
        filename: Name of the output file (should include .json extension)
    """
    output_path = output_dir / filename
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=4)
