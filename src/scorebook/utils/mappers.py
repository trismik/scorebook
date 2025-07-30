"""Utility functions for mapping and converting data types in Scorebook."""

from typing import Any, Literal

ClassificationResult = Literal["true_positive", "false_positive", "true_negative", "false_negative"]


def to_binary(value: Any) -> int:
    """Transform various input types to binary (0/1) classification value."""
    if value is None:
        return 0
    if isinstance(value, str):
        if value.upper() in ["A", "1", "TRUE", "YES", "Y"]:
            return 1
        return 0
    return 1 if value else 0


def to_binary_classification(prediction: Any, reference: Any) -> ClassificationResult:
    """
    Determine classification result based on prediction and reference values.

    Args:
        prediction: Predicted value (will be converted to binary)
        reference: Reference/true value (will be converted to binary)

    Returns:
        Classification result as one of: "true_positive", "false_positive",
                                       "true_negative", "false_negative"
    """
    pred_binary = to_binary(prediction)
    ref_binary = to_binary(reference)

    if pred_binary == 1:
        return "true_positive" if ref_binary == 1 else "false_positive"
    return "false_negative" if ref_binary == 1 else "true_negative"
