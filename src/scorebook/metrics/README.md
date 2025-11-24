# Adding Metrics to Scorebook

This guide explains how to add new metrics to Scorebook in a standardized, repeatable way.

## Quick Start

1. Create a metric file: `src/scorebook/metrics/yourmetric.py`
2. Implement the metric class
3. Add unit tests: `tests/unit/test_metrics/test_yourmetric.py`
4. Add optional tests (if needed): `tests/optional/test_metrics/test_yourmetric_optional.py`
5. Submit PR for review

---

## 1. Adding a Metric

### File Naming Convention

Metric files must follow the normalized naming convention (lowercase, no underscores/spaces):
- Class: `F1Score` → File: `f1score.py`
- Class: `MeanSquaredError` → File: `meansquarederror.py`
- Class: `Accuracy` → File: `accuracy.py`

### Implementation Template

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
        if len(outputs) != len(labels):
            raise ValueError("Number of outputs must match number of labels")

        if not outputs:  # Handle empty lists
            return {"your_metric": 0.0}, []

        # Calculate per-item scores
        item_scores = [calculate_score(out, lab) for out, lab in zip(outputs, labels)]

        # Calculate aggregate score
        aggregate_score = sum(item_scores) / len(item_scores)
        aggregate_scores = {"your_metric": aggregate_score}

        return aggregate_scores, item_scores
```

### Key Requirements

- **Decorator**: Must use `@scorebook_metric` to register the metric
- **Inheritance**: Must inherit from `MetricBase`
- **Method**: Must implement the `score()` static method
- **Return Type**: Must return `(Dict[str, Any], List[Any])` - aggregate scores and item scores
- **Error Handling**: Validate input lengths and handle edge cases (e.g., empty lists)
- **Naming**: Class name will be normalized (lowercase, no underscores) for user access

---

## 2. Writing Documentation

### Docstring Requirements

Each metric must have:

1. **Module-level docstring**: Brief description at the top of the file
2. **Class docstring**:
   - What the metric measures
   - Formula or calculation method
   - Usage examples
   - Any important notes or limitations
3. **Method docstring**:
   - Clear Args, Returns, and Raises sections
   - Type hints for all parameters

---

## 3. Adding Unit Tests

### Unit Tests

Create unit tests in `tests/unit/test_metrics/test_yourmetric.py`.

**These tests should**:
- Use mocked data (no external dependencies)
- Run quickly (< 1 second per test)
- Test core functionality
- Cover edge cases

### Optional Tests

Create optional tests in `tests/optional/metrics/test_yourmetric.py`.

**Mark tests as optional when they**:
- Require external dependencies (APIs, datasets, models)
- Take significant time to run (> 5 seconds)
- Test integration with external services
- Use large datasets or computationally expensive operations

---

## Example: Complete Metric Implementation

See `src/scorebook/metrics/accuracy.py` for a complete reference implementation.

---
