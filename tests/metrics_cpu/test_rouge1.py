"""Tests for Rouge1 metric."""

from scorebook.metrics.rouge1 import Rouge1


def test_rouge1_perfect_match() -> None:
    """Test Rouge1 with identical outputs and labels."""
    rouge1 = Rouge1(use_stemmer=True)
    outputs = ["Scientists discovered a new species of frog in the Amazon rainforest."]
    labels = ["Scientists discovered a new species of frog in the Amazon rainforest."]

    agg, items = rouge1.score(outputs, labels)

    # Perfect match should have score of 1.0
    assert agg["rouge1"] == 1.0
    assert items[0]["rouge1"] == 1.0
    assert len(items) == len(outputs)
    # Should only have one key (1:1 mapping)
    assert len(agg) == 1
    assert len(items[0]) == 1


def test_rouge1_no_match() -> None:
    """Test Rouge1 with completely different texts."""
    rouge1 = Rouge1(use_stemmer=True)
    outputs = ["The company reported record quarterly earnings."]
    labels = ["Researchers found a cure for a rare disease."]

    agg, items = rouge1.score(outputs, labels)

    assert agg["rouge1"] == 0.0
    assert items[0]["rouge1"] == 0.0
    assert len(agg) == 1
    assert len(items[0]) == 1


def test_rouge1_multiple_items() -> None:
    """Test Rouge1 with multiple output-label pairs."""
    rouge1 = Rouge1(use_stemmer=True)
    outputs = [
        "New study shows exercise improves mental health.",
        "The music festival was very successful yesterday.",
    ]
    labels = [
        "New study shows exercise improves mental health.",
        "Scientists explore quantum physics in laboratories.",
    ]

    agg, items = rouge1.score(outputs, labels)

    # First item perfect match
    assert items[0]["rouge1"] == 1.0
    # Second item no match
    assert items[1]["rouge1"] == 0.0
    # Aggregate is average
    assert agg["rouge1"] == 0.5


def test_rouge1_argument_order() -> None:
    """Test that scorer passes (reference, prediction) in the correct order.

    When the output is a subset of the label, precision should be high
    (all predicted words appear in reference) and recall should be low
    (only a fraction of reference words were predicted). With swapped
    arguments, these values would be inverted.
    """
    from unittest.mock import patch

    rouge1 = Rouge1(use_stemmer=False)
    outputs = ["the cat"]
    labels = ["the cat sat on the mat"]

    with patch.object(rouge1.scorer, "score", wraps=rouge1.scorer.score) as mock_score:
        rouge1.score(outputs, labels)
        # rouge_score expects score(target/reference, prediction)
        mock_score.assert_called_once_with("the cat sat on the mat", "the cat")


def test_rouge1_empty_lists() -> None:
    """Test Rouge1 with empty inputs."""
    rouge1 = Rouge1(use_stemmer=True)
    outputs = []
    labels = []

    agg, items = rouge1.score(outputs, labels)

    assert agg == {"rouge1": 0.0}
    assert items == []
