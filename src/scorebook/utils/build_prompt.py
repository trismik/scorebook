"""
Module for building prompt strings using Jinja2 templating.

Provides functionality to render prompts from templates with custom filters
and global variables, using strict undefined handling for better error detection.
"""

from typing import Any, Dict, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined

from scorebook.utils.jinja_helpers import default_jinja_filters, default_jinja_globals


def build_prompt(
    prompt_template: str,
    prompt_args: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    globals_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a prompt string from a template and arguments.

    Args:
        prompt_template: Jinja2 template string
        prompt_args: Dictionary of arguments to pass to the template
        filters: Dictionary of Jinja2 filters. Defaults to default_jinja_filters().
        globals_dict: Dictionary of global functions/variables. Defaults to default_jinja_globals().

    Returns:
        str: Rendered prompt string
    """

    # Use defaults if not provided
    filters = filters or default_jinja_filters()
    globals_dict = globals_dict or default_jinja_globals()

    # Create a Jinja2 environment with strict undefined handling
    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add filters and globals
    env.filters.update(filters)
    env.globals.update(globals_dict)

    # Render the template
    template = env.from_string(prompt_template)
    return str(template.render(**prompt_args))
