"""Tests for ROUGE metric."""

from scorebook.metrics.rouge import ROUGE


def test_rouge_perfect_match() -> None:
    """Test ROUGE with identical outputs and labels."""
    rouge = ROUGE()
    outputs = ["The quick brown fox jumps over the lazy dog"]
    labels = ["The quick brown fox jumps over the lazy dog"]

    agg, items = rouge.score(outputs, labels)

    # Perfect match should have scores of 1.0
    assert agg["rouge1"] == 1.0
    assert agg["rougeL"] == 1.0
    assert items[0]["rouge1"] == 1.0
    assert items[0]["rougeL"] == 1.0
    assert len(items) == len(outputs)


def test_rouge_no_match() -> None:
    """Test ROUGE with completely different texts."""
    rouge = ROUGE()
    outputs = ["apple orange banana"]
    labels = ["car truck motorcycle"]

    agg, items = rouge.score(outputs, labels)

    # No overlap should have scores of 0.0
    assert agg["rouge1"] == 0.0
    assert agg["rougeL"] == 0.0
    assert items[0]["rouge1"] == 0.0
    assert items[0]["rougeL"] == 0.0
    assert len(items) == len(outputs)


def test_rouge_partial_match() -> None:
    """Test ROUGE with partial overlap."""
    rouge = ROUGE()
    outputs = ["The quick brown dog"]
    labels = ["The quick brown fox"]

    agg, items = rouge.score(outputs, labels)

    # Partial overlap should have scores between 0 and 1
    assert 0.0 < agg["rouge1"] < 1.0
    assert 0.0 < agg["rougeL"] < 1.0
    assert 0.0 < items[0]["rouge1"] < 1.0
    assert 0.0 < items[0]["rougeL"] < 1.0
    assert len(items) == len(outputs)


def test_rouge_empty_lists() -> None:
    """Test ROUGE with empty inputs."""
    rouge = ROUGE()
    outputs = []
    labels = []

    agg, items = rouge.score(outputs, labels)

    assert agg == {"rouge1": 0.0, "rougeL": 0.0}
    assert items == []


def test_rouge_multiple_items() -> None:
    """Test ROUGE with multiple output-label pairs."""
    rouge = ROUGE()
    outputs = [
        "A woman donated her kidney to a stranger.",
        "Scientists discovered a new frog species.",
    ]
    labels = [
        "A woman donated her kidney to a stranger.",
        "Completely different text here.",
    ]

    agg, items = rouge.score(outputs, labels)

    # First item should have high scores (perfect match)
    assert items[0]["rouge1"] == 1.0
    assert items[0]["rougeL"] == 1.0

    # Second item should have low scores (no match)
    assert items[1]["rouge1"] == 0.0
    assert items[1]["rougeL"] == 0.0

    # Aggregate should be average of both
    assert agg["rouge1"] == 0.5
    assert agg["rougeL"] == 0.5
    assert len(items) == len(outputs)


def test_rouge_with_none_values() -> None:
    """Test ROUGE handles None values correctly."""
    rouge = ROUGE()
    outputs = [None, "Some text"]
    labels = ["Some text", None]

    agg, items = rouge.score(outputs, labels)

    # Should convert None to empty string and score accordingly
    assert items[0]["rouge1"] == 0.0
    assert items[0]["rougeL"] == 0.0
    assert items[1]["rouge1"] == 0.0
    assert items[1]["rougeL"] == 0.0
    assert len(items) == len(outputs)


def test_rouge_mismatched_lengths() -> None:
    """Test ROUGE handles mismatched input lengths by truncating to shorter length."""
    rouge = ROUGE()
    outputs = ["text1", "text2"]
    labels = ["text1"]

    agg, items = rouge.score(outputs, labels)

    # Should only score the first pair (truncates to shorter length)
    assert len(items) == 1
    assert items[0]["rouge1"] == 1.0
    assert items[0]["rougeL"] == 1.0


def test_rouge_custom_kwargs() -> None:
    """Test ROUGE accepts custom rouge_types and kwargs."""
    # Test with custom rouge_types
    rouge = ROUGE(rouge_types=["rouge1", "rouge2", "rougeL"])
    outputs = ["The quick brown fox"]
    labels = ["The quick brown fox"]

    agg, items = rouge.score(outputs, labels)

    # Should have all three rouge types
    assert "rouge1" in agg
    assert "rouge2" in agg
    assert "rougeL" in agg
    assert agg["rouge1"] == 1.0
    assert agg["rouge2"] == 1.0
    assert agg["rougeL"] == 1.0

    # Test with use_stemmer=False
    rouge_no_stem = ROUGE(use_stemmer=False)
    outputs2 = ["running quickly"]
    labels2 = ["run quick"]

    agg2, items2 = rouge_no_stem.score(outputs2, labels2)

    # Without stemmer, these should have lower overlap
    assert 0.0 <= agg2["rouge1"] <= 1.0
    assert len(items2) == 1
