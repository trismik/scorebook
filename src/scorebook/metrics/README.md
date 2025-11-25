# Adding Metrics to Scorebook

This guide explains how to add new metrics to Scorebook.

## Quick Start

1. Create a metric file: `src/scorebook/metrics/yourmetric.py`
2. Implement the metric class
3. Add tests
4. Submit PR for review

### Where to Put Tests

Tests go in one of two directories:

- **`tests/unit/test_metrics/`** - For fast tests using mocked data. These run on every commit.
- **`tests/extended/test_metrics/`** - For tests that require external dependencies, large datasets, or are computationally expensive.

Most metrics only need unit tests. Use extended tests when your metric relies on external APIs, models, or takes significant time to run.

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for instructions on running tests.

---

## Requirements

Your metric must:

- Use the `@scorebook_metric` decorator
- Inherit from `MetricBase`
- Implement the `score()` static method

The `score()` method returns a tuple of `(aggregate_scores, item_scores)`:

- **aggregate_scores**: A `Dict[str, float]` with overall metric values (e.g., `{"accuracy": 0.85}`)
- **item_scores**: A `List` of per-item scores. Supported types: `int`, `float`, `bool`, `str`, or `Dict` containing these types.

---

## File Naming

Metric files use normalized names (lowercase, no underscores/spaces):
- Class: `F1Score` → File: `f1score.py`
- Class: `MeanSquaredError` → File: `meansquarederror.py`

---

## Implementation Template

Create your metric file in `src/scorebook/metrics/yourmetric.py`:

```python
"""Brief description of the metric."""

from typing import Any, Dict, List, Tuple

from scorebook.metrics import MetricBase, scorebook_metric


@scorebook_metric
class YourMetric(MetricBase):
    """One-line description of what this metric measures.

    Formula or explanation (e.g., Accuracy = correct / total).
    """

    def score(outputs: List[Any], labels: List[Any]) -> Tuple[Dict[str, Any], List[Any]]:
        """Calculate metric score between outputs and labels.

        Args:
            outputs: A list of model inference outputs.
            labels: A list of ground truth labels.

        Returns:
            Tuple containing:
                - Aggregate scores dict (e.g., {"your_metric": 0.85})
                - List of per-item scores

        Raises:
            ValueError: If outputs and labels have different lengths.
        """
        # Input validation
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:
            return {"your_metric": 0.0}, []

        # Calculate per-item scores
        item_scores = [calculate_score(out, lab) for out, lab in zip(outputs, labels)]

        # Calculate aggregate score
        aggregate_score = sum(item_scores) / len(item_scores)

        return {"your_metric": aggregate_score}, item_scores
```

---

## Documentation

Each metric should have:

1. **Module-level docstring**: Brief description at the top of the file
2. **Class docstring**: What the metric measures, formula, and any limitations
3. **Method docstring**: Args, Returns, and Raises sections

---

## Example

See `src/scorebook/metrics/accuracy.py` for a complete reference implementation.
