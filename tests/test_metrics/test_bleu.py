"""Tests for BLEU metric."""

import pytest

from scorebook.metrics.bleu import BLEU


def test_bleu_perfect_match() -> None:
    """Test BLEU with identical outputs and labels."""
    bleu = BLEU()
    outputs = ["The quick brown fox jumps over the lazy dog"]
    labels = ["The quick brown fox jumps over the lazy dog"]

    agg, items = bleu.score(outputs, labels)

    # Perfect match should have BLEU score of 100
    assert agg["BLEU"] == pytest.approx(100.0)
    assert items[0]["BLEU"] == pytest.approx(100.0)
    assert len(items) == len(outputs)


def test_bleu_no_match() -> None:
    """Test BLEU with completely different texts."""
    bleu = BLEU()
    outputs = ["apple orange banana"]
    labels = ["car truck motorcycle"]

    agg, items = bleu.score(outputs, labels)

    # No overlap should have BLEU score of 0.0
    assert agg["BLEU"] == 0.0
    assert items[0]["BLEU"] == 0.0
    assert len(items) == len(outputs)


def test_bleu_partial_match() -> None:
    """Test BLEU with partial overlap."""
    bleu = BLEU()
    outputs = ["The quick brown dog"]
    labels = ["The quick brown fox"]

    agg, items = bleu.score(outputs, labels)

    # Partial overlap should have BLEU score between 0 and 100
    assert 0.0 < agg["BLEU"] < 100.0
    assert 0.0 < items[0]["BLEU"] < 100.0
    assert len(items) == len(outputs)


def test_bleu_empty_lists() -> None:
    """Test BLEU with empty inputs."""
    bleu = BLEU()
    outputs = []
    labels = []

    agg, items = bleu.score(outputs, labels)

    assert agg == {"accuracy": 0.0}
    assert items == []


def test_bleu_mismatched_lengths() -> None:
    """Test BLEU raises error with mismatched input lengths."""
    bleu = BLEU()
    outputs = ["text1", "text2"]
    labels = ["text1"]

    try:
        bleu.score(outputs, labels)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Number of outputs must match number of labels" in str(e)


def test_bleu_multiple_items() -> None:
    """Test BLEU with multiple output-label pairs."""
    bleu = BLEU()
    outputs = [
        "A woman donated her kidney to a stranger.",
        "Scientists discovered a new frog species.",
    ]
    labels = [
        "A woman donated her kidney to a stranger.",
        "Completely different text here.",
    ]

    agg, items = bleu.score(outputs, labels)

    # First item should have high BLEU score (perfect match)
    assert items[0]["BLEU"] == pytest.approx(100.0)

    # Second item should have low BLEU score (poor match)
    assert items[1]["BLEU"] < 10.0

    # Aggregate should be between the two
    assert 0.0 < agg["BLEU"] < 100.0
    assert len(items) == len(outputs)


def test_bleu_compact_mode() -> None:
    """Test BLEU in compact mode (default)."""
    bleu = BLEU(compact=True)
    outputs = ["The quick brown fox"]
    labels = ["The quick brown fox"]

    agg, items = bleu.score(outputs, labels)

    # Compact mode should only have BLEU score
    assert "BLEU" in agg
    assert "1-gram" not in agg
    assert "2-gram" not in agg
    assert "3-gram" not in agg
    assert "4-gram" not in agg
    assert "BP" not in agg
    assert "ratio" not in agg
    assert "hyp_len" not in agg
    assert "ref_len" not in agg

    # Same for items
    assert "BLEU" in items[0]
    assert "1-gram" not in items[0]
    assert len(items) == 1


def test_bleu_non_compact_mode() -> None:
    """Test BLEU in non-compact mode with detailed metrics."""
    bleu = BLEU(compact=False)
    outputs = ["The quick brown fox"]
    labels = ["The quick brown fox"]

    agg, items = bleu.score(outputs, labels)

    # Non-compact mode should have all detailed metrics
    assert "BLEU" in agg
    assert "1-gram" in agg
    assert "2-gram" in agg
    assert "3-gram" in agg
    assert "4-gram" in agg
    assert "BP" in agg
    assert "ratio" in agg
    assert "hyp_len" in agg
    assert "ref_len" in agg

    # Same for items
    assert "BLEU" in items[0]
    assert "1-gram" in items[0]
    assert "2-gram" in items[0]
    assert "3-gram" in items[0]
    assert "4-gram" in items[0]
    assert "BP" in items[0]
    assert "ratio" in items[0]
    assert "hyp_len" in items[0]
    assert "ref_len" in items[0]

    # Perfect match should have perfect n-gram precisions
    assert agg["1-gram"] == 100.0
    assert agg["2-gram"] == 100.0
    assert agg["3-gram"] == 100.0
    assert agg["4-gram"] == 100.0
    assert len(items) == 1
