"""Tests for RougeL metric."""

from scorebook.metrics.rougel import RougeL


def test_rougel_perfect_match() -> None:
    """Test RougeL with identical outputs and labels."""
    rougel = RougeL(use_stemmer=True)
    outputs = ["Scientists discovered a new species of frog in the Amazon rainforest."]
    labels = ["Scientists discovered a new species of frog in the Amazon rainforest."]

    agg, items = rougel.score(outputs, labels)

    assert agg["rougeL"] == 1.0
    assert items[0]["rougeL"] == 1.0
    assert len(items) == len(outputs)
    # Should only have one key (1:1 mapping)
    assert len(agg) == 1
    assert len(items[0]) == 1


def test_rougel_no_match() -> None:
    """Test RougeL with completely different texts."""
    rougel = RougeL(use_stemmer=True)
    outputs = ["The company reported record quarterly earnings."]
    labels = ["Researchers found a cure for a rare disease."]

    agg, items = rougel.score(outputs, labels)

    assert agg["rougeL"] == 0.0
    assert items[0]["rougeL"] == 0.0
    assert len(agg) == 1
    assert len(items[0]) == 1


def test_rougel_multiple_items() -> None:
    """Test RougeL with multiple output-label pairs."""
    rougel = RougeL(use_stemmer=True)
    outputs = [
        "New study shows exercise improves mental health.",
        "The music festival was very successful yesterday.",
    ]
    labels = [
        "New study shows exercise improves mental health.",
        "Scientists explore quantum physics in laboratories.",
    ]

    agg, items = rougel.score(outputs, labels)

    # First item perfect match
    assert items[0]["rougeL"] == 1.0
    # Second item no match
    assert items[1]["rougeL"] == 0.0
    # Aggregate is average
    assert agg["rougeL"] == 0.5


def test_rougel_argument_order() -> None:
    """Test that scorer passes (reference, prediction) in the correct order.

    When the output is a subset of the label, precision should be high
    and recall should be low. With swapped arguments, these values
    would be inverted.
    """
    from unittest.mock import patch

    rougel = RougeL(use_stemmer=False)
    outputs = ["the cat"]
    labels = ["the cat sat on the mat"]

    with patch.object(rougel.scorer, "score", wraps=rougel.scorer.score) as mock_score:
        rougel.score(outputs, labels)
        # rouge_score expects score(target/reference, prediction)
        mock_score.assert_called_once_with("the cat sat on the mat", "the cat")


def test_rougel_empty_lists() -> None:
    """Test RougeL with empty inputs."""
    rougel = RougeL(use_stemmer=True)
    outputs = []
    labels = []

    agg, items = rougel.score(outputs, labels)

    assert agg == {"rougeL": 0.0}
    assert items == []
