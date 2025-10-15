"""Tests for render_template function."""

from pathlib import Path

import pytest
import yaml
from jinja2.exceptions import UndefinedError

from scorebook.utils.render_template import render_template


def test_simple_template():
    """Test basic variable substitution."""
    template = "Hello {{ name }}!"
    args = {"name": "World"}
    result = render_template(template, args)
    print(f"Simple template result:\n{result}")
    assert result == "Hello World!"


def test_multiple_variables():
    """Test template with multiple variables."""
    template = "{{ greeting }} {{ name }}, you are {{ age }} years old."
    args = {"greeting": "Hi", "name": "Alice", "age": 25}
    result = render_template(template, args)
    print(f"Multiple variables result:\n{result}")
    assert result == "Hi Alice, you are 25 years old."


def test_for_loop_with_options():
    """Test template with for loop over options."""
    template = """Question: {{ question }}
Options:
{% for option in options %}
{{ loop.index }}. {{ option }}
{% endfor %}"""
    args = {
        "question": "What is the capital of France?",
        "options": ["London", "Paris", "Berlin", "Madrid"],
    }
    result = render_template(template, args)
    print(f"For loop with options result:\n{result}")
    expected = """Question: What is the capital of France?
Options:
1. London
2. Paris
3. Berlin
4. Madrid
"""
    assert result == expected


def test_number_to_letter_function():
    """Test the custom number_to_letter function."""
    template = """{{ question }}
Options:
{% for option in options %}
{{ number_to_letter(loop.index0) }}: {{ option }}
{% endfor %}"""
    args = {"question": "What is 2+2?", "options": ["3", "4", "5", "6"]}
    result = render_template(template, args)
    print(f"Number to letter function result:\n{result}")
    expected = """What is 2+2?
Options:
A: 3
B: 4
C: 5
D: 6
"""
    assert result == expected


def test_mmlu_template_format():
    """Test using the exact template format from dataset_template.yaml."""
    template = """{{ question }}
Options:
{% for option in options %}
{{ number_to_letter(loop.index0) }} : {{ option }}
{% endfor %}"""
    args = {
        "question": "Which of the following is a programming language?",
        "options": ["HTML", "Python", "CSS", "JSON"],
    }
    result = render_template(template, args)
    print(f"MMLU template format result:\n{result}")
    expected = """Which of the following is a programming language?
Options:
A : HTML
B : Python
C : CSS
D : JSON
"""
    assert result == expected


def test_empty_options_list():
    """Test template with empty options list."""
    template = """{{ question }}
Options:
{% for option in options %}
{{ number_to_letter(loop.index0) }}: {{ option }}
{% endfor %}"""
    args = {"question": "Test question", "options": []}
    result = render_template(template, args)
    expected = """Test question
Options:
"""
    assert result == expected


def test_missing_variable_raises_error():
    """Test that missing variables raise an error in strict mode."""
    template = "Hello {{ missing_var }}!"
    args = {}
    with pytest.raises(UndefinedError):
        render_template(template, args)


def test_with_yaml_template_loading():
    """Test loading template from YAML file and using it."""
    # Load the actual template from the test data
    yaml_path = Path(__file__).parent.parent / "data" / "dataset_template.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    template = config["templates"]["input"]
    args = {
        "question": "What is the speed of light?",
        "options": ["299,792,458 m/s", "300,000,000 m/s", "299,000,000 m/s", "301,000,000 m/s"],
    }

    result = render_template(template, args)
    print(f"YAML template loading result:\n{result}")
    expected = """What is the speed of light?
Options:
A : 299,792,458 m/s
B : 300,000,000 m/s
C : 299,000,000 m/s
D : 301,000,000 m/s
"""
    assert result == expected


def test_custom_filters():
    """Test custom filters functionality."""
    template = "{{ name | upper }}"
    args = {"name": "hello"}
    filters = {"upper": str.upper}
    result = render_template(template, args, filters=filters)
    assert result == "HELLO"


def test_custom_globals():
    """Test custom globals functionality."""
    template = "Result: {{ custom_func(5) }}"
    args = {}
    globals_dict = {"custom_func": lambda x: x * 2}
    result = render_template(template, args, globals_dict=globals_dict)
    assert result == "Result: 10"


def test_chr_filter():
    """Test built-in chr filter."""
    template = "{{ 65 | chr }}"
    args = {}
    result = render_template(template, args)
    assert result == "A"


def test_helper_functions():
    """Test various helper functions from jinja_helpers."""
    template = """{{ letter_to_number('B') }}
{{ format_list(['apple', 'banana', 'cherry']) }}
{{ ordinal(3) }}
{{ percentage(25, 100) }}"""
    args = {}
    result = render_template(template, args)
    expected = """1
apple, banana and cherry
3rd
25.0%"""
    print(f"Helper functions result:\n{result}")
    assert result == expected


def test_format_list_helper():
    """Test format_list helper with different cases."""
    template = "{{ format_list(items) }}"

    # Test single item
    result = render_template(template, {"items": ["apple"]})
    assert result == "apple"

    # Test two items
    result = render_template(template, {"items": ["apple", "banana"]})
    assert result == "apple and banana"

    # Test three items
    result = render_template(template, {"items": ["apple", "banana", "cherry"]})
    assert result == "apple, banana and cherry"


def test_truncate_helper():
    """Test truncate_text helper."""
    template = "{{ truncate_text(text, 10) }}"
    args = {"text": "This is a very long text that needs truncation"}
    result = render_template(template, args)
    assert result == "This is..."
