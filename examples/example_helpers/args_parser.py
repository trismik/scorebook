"""
Generic argument parsing utilities for Scorebook examples.

This module provides reusable argument parsing functions that can be used
across multiple Scorebook examples for consistent command-line interfaces.
"""

import argparse
from typing import Any, Dict, List, Optional


def create_parser(description: str) -> argparse.ArgumentParser:
    """Create a basic argument parser with a description.

    Args:
        description: Description for the argument parser

    Returns:
        Configured ArgumentParser instance
    """
    return argparse.ArgumentParser(description=description)


def add_model_selection_arg(
    parser: argparse.ArgumentParser,
    default: str = "gpt-4o-mini",
    help_text: Optional[str] = None,
    supported_models: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    """Add model selection argument to parser.

    Args:
        parser: ArgumentParser to add the argument to
        default: Default model name
        help_text: Custom help text for the argument
        supported_models: List of supported models for validation

    Returns:
        The modified parser
    """
    if help_text is None:
        help_text = f"OpenAI model to use for inference (default: {default})"
        if supported_models:
            help_text += f". Supported models: {', '.join(supported_models)}"

    parser.add_argument(
        "--model",
        type=str,
        default=default,
        help=help_text,
    )
    return parser


def setup_openai_model_parser(
    description: str = "Select OpenAI model for evaluation.",
    default: str = "gpt-4o-mini",
    supported_models: Optional[List[str]] = None,
) -> str:
    """Set up and parse OpenAI model selection arguments.

    Args:
        description: Description for the argument parser
        default: Default model name
        supported_models: List of supported models for help text

    Returns:
        Selected model name
    """
    parser = create_parser(description)
    add_model_selection_arg(parser, default=default, supported_models=supported_models)
    args = parser.parse_args()
    return str(args.model)


def setup_batch_model_parser(
    description: str = "Select OpenAI model for batch evaluation.", default: str = "gpt-4o-mini"
) -> str:
    """Set up and parse OpenAI model selection arguments for batch inference.

    Args:
        description: Description for the argument parser
        default: Default model name

    Returns:
        Selected model name
    """
    supported_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    help_text = (
        f"OpenAI model to use for batch inference. "
        f"Note: Only select models support the Batch API. "
        f"Supported models include: {', '.join(supported_models)}. "
        f"Default: {default}"
    )

    parser = create_parser(description)
    add_model_selection_arg(parser, default=default, help_text=help_text)
    args = parser.parse_args()
    return str(args.model)


def parse_args_with_config(config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse arguments using a configuration dictionary.

    Args:
        config: Dictionary defining arguments to add. Format:
            {
                "arg_name": {
                    "type": str,
                    "default": "default_value",
                    "help": "Help text",
                    "required": False  # optional
                }
            }

    Returns:
        Dictionary of parsed argument values
    """
    parser = argparse.ArgumentParser()

    for arg_name, arg_config in config.items():
        kwargs = {"type": arg_config.get("type", str), "help": arg_config.get("help", "")}

        if "default" in arg_config:
            kwargs["default"] = arg_config["default"]
        if "required" in arg_config:
            kwargs["required"] = arg_config["required"]

        parser.add_argument(f"--{arg_name.replace('_', '-')}", **kwargs)

    args = parser.parse_args()
    return vars(args)
