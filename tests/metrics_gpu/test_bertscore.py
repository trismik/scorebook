"""Tests for BertScore metric (requires bert-score package).

These tests verify BertScore functionality without asserting exact values
since BertScore is non-deterministic.
"""

from scorebook.metrics.bertscore import BertScore


def test_bertscore_returns_valid_structure() -> None:
    """Test that BertScore returns the expected structure."""
    bertscore = BertScore()
    outputs = ["The quick brown fox jumps over the lazy dog"]
    labels = ["The quick brown fox jumps over the lazy dog"]

    agg, items = bertscore.score(outputs, labels)

    # Check aggregate structure
    assert "precision" in agg
    assert "recall" in agg
    assert "F1" in agg

    # Check items structure
    assert len(items) == len(outputs)
    assert "precision" in items[0]
    assert "recall" in items[0]
    assert "F1" in items[0]


def test_bertscore_scores_in_valid_range() -> None:
    """Test that BertScore returns scores in valid range [0, 1]."""
    bertscore = BertScore()
    outputs = ["The quick brown fox jumps over the lazy dog"]
    labels = ["The quick brown fox jumps over the lazy dog"]

    agg, items = bertscore.score(outputs, labels)

    # Aggregate scores should be approximately in [0, 1]
    # Using small epsilon for floating point tolerance
    eps = 1e-6
    assert -eps <= agg["precision"] <= 1.0 + eps
    assert -eps <= agg["recall"] <= 1.0 + eps
    assert -eps <= agg["F1"] <= 1.0 + eps

    # Item scores should be approximately in [0, 1]
    assert -eps <= items[0]["precision"] <= 1.0 + eps
    assert -eps <= items[0]["recall"] <= 1.0 + eps
    assert -eps <= items[0]["F1"] <= 1.0 + eps


def test_bertscore_similar_texts_high_score() -> None:
    """Test that similar texts produce relatively high scores."""
    bertscore = BertScore()
    outputs = ["The quick brown fox jumps over the lazy dog"]
    labels = ["The quick brown fox jumps over the lazy dog"]

    agg, items = bertscore.score(outputs, labels)

    # Identical texts should have high scores (> 0.9 is reasonable)
    assert agg["F1"] > 0.9
    assert items[0]["F1"] > 0.9


def test_bertscore_dissimilar_texts_lower_score() -> None:
    """Test that dissimilar texts produce lower scores than similar texts."""
    bertscore = BertScore()

    # Similar texts
    similar_outputs = ["The cat sat on the mat"]
    similar_labels = ["The cat sat on the mat"]
    similar_agg, _ = bertscore.score(similar_outputs, similar_labels)

    # Dissimilar texts
    dissimilar_outputs = ["The cat sat on the mat"]
    dissimilar_labels = ["Quantum physics describes subatomic particles"]
    dissimilar_agg, _ = bertscore.score(dissimilar_outputs, dissimilar_labels)

    # Similar texts should have higher score than dissimilar
    assert similar_agg["F1"] > dissimilar_agg["F1"]


def test_bertscore_multiple_items() -> None:
    """Test BertScore with multiple output-label pairs."""
    bertscore = BertScore()
    outputs = [
        "A woman donated her kidney to a stranger.",
        "Scientists discovered a new species of frog.",
        "The weather today is sunny and warm.",
    ]
    labels = [
        "A woman donated her kidney to a stranger.",
        "Scientists found a new type of frog.",
        "Completely unrelated text about quantum mechanics.",
    ]

    agg, items = bertscore.score(outputs, labels)

    # Should return correct number of items
    assert len(items) == 3

    # All scores should be approximately in valid range
    eps = 1e-6
    for item in items:
        assert -eps <= item["precision"] <= 1.0 + eps
        assert -eps <= item["recall"] <= 1.0 + eps
        assert -eps <= item["F1"] <= 1.0 + eps

    # First pair (identical) should have higher score than third pair (dissimilar)
    assert items[0]["F1"] > items[2]["F1"]


def test_bertscore_empty_lists() -> None:
    """Test BertScore with empty inputs."""
    bertscore = BertScore()
    outputs: list[str] = []
    labels: list[str] = []

    agg, items = bertscore.score(outputs, labels)

    assert agg == {"precision": 0.0, "recall": 0.0, "F1": 0.0}
    assert items == []


def test_bertscore_custom_kwargs() -> None:
    """Test BertScore with custom kwargs."""
    bertscore = BertScore(lang="en", verbose=False)
    outputs = ["Hello world"]
    labels = ["Hello world"]

    agg, items = bertscore.score(outputs, labels)

    # Should work without errors and return valid structure
    assert "F1" in agg
    assert len(items) == 1
