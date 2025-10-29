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


def setup_logging(
    log_dir: str = "logs",
    experiment_id: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Configure logging for evaluation runs.

    Args:
        log_dir: Name of the log directory (default: "logs")
        experiment_id: Optional identifier for the experiment
        base_dir: Base directory where log_dir should be created.
                  If None, uses current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    log_dir_path: Path = base_dir / log_dir
    log_dir_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_id:
        log_file = log_dir_path / f"evaluation_{experiment_id}_{timestamp}.log"
    else:
        log_file = log_dir_path / f"evaluation_{timestamp}.log"

    # Create file handler for all logs (same as before)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    # Create console handler for warnings and errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))

    # Configure root logger with both handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )

    # Set scorebook loggers to DEBUG level to capture all scorebook logs
    scorebook_logger = logging.getLogger("scorebook")
    scorebook_logger.setLevel(logging.DEBUG)

    # Ensure trismik_services logs are captured at DEBUG level
    trismik_services_logger = logging.getLogger("scorebook.trismik_services")
    trismik_services_logger.setLevel(logging.DEBUG)

    # Ensure evaluate logs are captured at DEBUG level
    evaluate_logger = logging.getLogger("scorebook.evaluate._sync.evaluate")
    evaluate_logger.setLevel(logging.DEBUG)
    evaluate_logger = logging.getLogger("scorebook.evaluate._async.evaluate_async")
    evaluate_logger.setLevel(logging.DEBUG)

    # Exclude OpenAI inference logs to reduce noise
    openai_logger = logging.getLogger("scorebook.inference.openai")
    openai_logger.setLevel(logging.WARNING)  # Only log warnings and errors

    print(f"Logging to {log_file}")
    return log_file
