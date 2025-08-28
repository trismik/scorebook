"""Jinja2 template helper functions for Scorebook."""

import json
import re
from typing import Any, Dict, List

# Helper functions for use in Jinja templates


def number_to_letter(index: int, uppercase: bool = True) -> str:
    """Convert a number to a letter (0->A, 1->B, etc.).

    Args:
        index: The number to convert to a letter (0-based index)
        uppercase: If True, returns uppercase letter; if False, returns lowercase
    """
    letter = chr(65 + index)
    return letter if uppercase else letter.lower()


def letter_to_number(letter: str) -> int:
    """Convert a letter to a number (A->0, B->1, etc.)."""
    return ord(letter.upper()) - 65


def format_list(items: List[Any], separator: str = ", ", last_separator: str = " and ") -> str:
    """Format a list with proper separators and conjunction.

    Examples:
        format_list(["a", "b", "c"]) -> "a, b and c"
        format_list(["a", "b"]) -> "a and b"
        format_list(["a"]) -> "a"
    """
    if not items:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]}{last_separator}{items[1]}"
    return f"{separator.join(str(item) for item in items[:-1])}{last_separator}{items[-1]}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to a maximum length with optional suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_number(number: float, precision: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{number:.{precision}f}"


def extract_initials(text: str) -> str:
    """Extract initials from a text string.

    Examples:
        extract_initials("John Doe") -> "JD"
        extract_initials("Machine Learning Model") -> "MLM"
    """
    words = re.findall(r"\b[A-Za-z]", text)
    return "".join(words).upper()


def json_pretty(obj: Any, indent: int = 2) -> str:
    """Pretty-print an object as JSON."""
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def percentage(value: float, total: float, precision: int = 1) -> str:
    """Calculate and format a percentage.

    Examples:
        percentage(25, 100) -> "25.0%"
        percentage(1, 3, 2) -> "33.33%"
    """
    if total == 0:
        return "0.0%"
    pct = (value / total) * 100
    return f"{pct:.{precision}f}%"


def ordinal(n: int) -> str:
    """Convert number to ordinal format like 1st, 2nd, 3rd, etc."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def default_jinja_globals() -> Dict[str, Any]:
    """Get default global functions for Jinja templates."""
    return {
        "number_to_letter": number_to_letter,
        "letter_to_number": letter_to_number,
        "format_list": format_list,
        "truncate_text": truncate_text,
        "format_number": format_number,
        "extract_initials": extract_initials,
        "json_pretty": json_pretty,
        "percentage": percentage,
        "ordinal": ordinal,
        "max": max,
        "min": min,
        "len": len,
        "abs": abs,
        "round": round,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "range": range,
    }


def default_jinja_filters() -> Dict[str, Any]:
    """Get default filters for Jinja templates."""
    return {
        "chr": chr,
        "ord": ord,
        "abs": abs,
        "round": round,
        "len": len,
    }
