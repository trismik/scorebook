"""Jinja2 template helper functions for Scorebook."""

from typing import Any, Callable, Dict, Mapping, Optional

from jinja2 import Environment, StrictUndefined


def jinja_template_to_fn(
    template_str: str,
    *,
    autoescape: bool = False,
    filters: Optional[Dict[str, Any]] = None,
    tests: Optional[Dict[str, Any]] = None,
    globals: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Callable[[Mapping[str, Any]], str]:
    """
    Compile a Jinja template string to a function: fn(eval_item) -> str.

    `strict=True` raises if a variable is missing.
    """
    env = Environment(
        autoescape=autoescape,
        undefined=StrictUndefined if strict else None,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Useful defaults
    env.filters["chr"] = chr
    if filters:
        env.filters.update(filters)
    if tests:
        env.tests.update(tests)
    if globals:
        env.globals.update(globals)

    template = env.from_string(template_str)

    def render(eval_item: Mapping[str, Any]) -> str:
        return str(template.render(**eval_item))

    return render
