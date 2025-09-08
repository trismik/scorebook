"""
Utility functions for setting up Scorebook examples.

This module provides common helper functions used across multiple Scorebook examples
for output directory setup and logging configuration.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_output_directory() -> Path:
    """Parse command line arguments and setup output directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation and save results.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.cwd() / "examples/example_results"),
        help=(
            "Directory to save evaluation outputs (CSV and JSON). "
            "Defaults to ./examples/example_results in the current working directory."
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")
    return output_dir


def setup_logging(log_dir: str = "logs", experiment_id: Optional[str] = None) -> Path:
    """Configure logging for evaluation runs."""
    log_dir_path: Path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_id:
        log_file = log_dir_path / f"evaluation_{experiment_id}_{timestamp}.log"
    else:
        log_file = log_dir_path / f"evaluation_{timestamp}.log"

    # Configure root logger with INFO level (affects all libraries)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
        force=True,
    )

    # Set scorebook loggers to DEBUG level to capture all scorebook logs
    scorebook_logger = logging.getLogger("scorebook")
    scorebook_logger.setLevel(logging.DEBUG)

    print(f"Logging to {log_file}")
    return log_file
