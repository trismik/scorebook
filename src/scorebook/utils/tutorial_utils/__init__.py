"""
Helper utilities for Scorebook examples.

This module provides common helper functions used across multiple Scorebook examples
for setup, output handling, and argument parsing.
"""

# Argument parsing utilities
from .args_parser import (
    add_model_selection_arg,
    create_parser,
    parse_args_with_config,
    setup_batch_model_parser,
    setup_openai_model_parser,
)

# Output utilities
from .output import save_results_to_json

# Setup utilities
from .setup import setup_logging, setup_output_directory

__all__ = [
    # Setup
    "setup_logging",
    "setup_output_directory",
    # Output
    "save_results_to_json",
    # Argument parsing
    "create_parser",
    "add_model_selection_arg",
    "setup_openai_model_parser",
    "setup_batch_model_parser",
    "parse_args_with_config",
]
