"""
Module for building prompt strings using Jinja2 templating.

Provides functionality to render prompts from templates with custom filters
and global variables, using strict undefined handling for better error detection.
"""

from typing import Any, Dict, Optional

from jinja2 import BaseLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from scorebook.utils.jinja_helpers import default_jinja_filters, default_jinja_globals


def render_template(
    template: str,
    args: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    globals_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Render a Jinja2 template string with the provided arguments.

    Args:
        template: Jinja2 template string
        args: Dictionary of arguments to pass to the template
        filters: Dictionary of Jinja2 filters. Defaults to default_jinja_filters().
        globals_dict: Dictionary of global functions/variables. Defaults to default_jinja_globals().

    Returns:
        str: Rendered template string
    """

    # Use defaults if not provided
    filters = filters or default_jinja_filters()
    globals_dict = globals_dict or default_jinja_globals()

    # Create a sandboxed Jinja2 environment with strict undefined handling
    env = SandboxedEnvironment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add filters and globals
    env.filters.update(filters)
    env.globals.update(globals_dict)

    # Render the template
    jinja_template = env.from_string(template)
    return str(jinja_template.render(**args))
