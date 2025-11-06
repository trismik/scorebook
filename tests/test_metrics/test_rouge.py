

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
