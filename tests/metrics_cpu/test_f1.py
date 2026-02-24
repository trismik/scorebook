import pytest

from scorebook.metrics.f1 import F1


def _key(method: str) -> str:
    """Generate F1 metric key for given averaging method."""
    return f"F1 ({method})"


def test_f1_perfect_score():
    """Test F1 with all correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    metric = F1()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 1.0}
    assert items == []


def test_f1_zero_score():
    """Test F1 with all incorrect predictions."""
    outputs = ["A", "B", "C"]
    labels = ["X", "Y", "Z"]

    metric = F1()
    agg, items = metric.score(outputs, labels)

    assert agg[_key("macro")] == 0.0
    assert items == []


def test_f1_partial_score():
    """Test F1 with some correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "X"]

    metric = F1()
    agg, items = metric.score(outputs, labels)

    assert 0.0 < agg[_key("macro")] < 1.0
    assert items == []


def test_f1_empty_lists():
    """Test F1 with empty inputs."""
    outputs = []
    labels = []

    metric = F1()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 0.0}
    assert items == []


def test_f1_macro_averaging():
    """Test F1 with explicit macro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = F1(average="macro")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert 0.0 <= agg[_key("macro")] <= 1.0
    assert items == []


def test_f1_micro_averaging():
    """Test F1 with micro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = F1(average="micro")
    agg, items = metric.score(outputs, labels)

    assert _key("micro") in agg
    assert agg[_key("micro")] == 0.5  # 2 correct out of 4


def test_f1_weighted_averaging():
    """Test F1 with weighted averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = F1(average="weighted")
    agg, items = metric.score(outputs, labels)

    assert _key("weighted") in agg
    assert 0.0 <= agg[_key("weighted")] <= 1.0


def test_f1_all_averaging():
    """Test F1 with 'all' averaging returns all three methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = F1(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert len(agg) == 3


def test_f1_list_of_methods():
    """Test F1 with a list of averaging methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = F1(average=["macro", "micro"])
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") not in agg
    assert len(agg) == 2


def test_f1_invalid_average_method():
    """Test F1 raises error for invalid averaging method."""
    with pytest.raises(ValueError, match="Invalid average method"):
        F1(average="invalid")


def test_f1_invalid_average_method_in_list():
    """Test F1 raises error for invalid method in list."""
    with pytest.raises(ValueError, match="Invalid average method"):
        F1(average=["macro", "invalid"])


def test_f1_all_combined_with_others():
    """Test F1 raises error when 'all' is combined with other methods."""
    with pytest.raises(ValueError, match="'all' cannot be combined"):
        F1(average=["all", "macro"])


def test_f1_binary_classification():
    """Test F1 on binary classification task."""
    outputs = [1, 1, 0, 0, 1]
    labels = [1, 0, 0, 1, 1]

    metric = F1()
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert items == []


def test_f1_multiclass_classification():
    """Test F1 on multi-class classification task."""
    outputs = ["cat", "dog", "bird", "cat", "dog"]
    labels = ["cat", "dog", "cat", "bird", "dog"]

    metric = F1(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert items == []


def test_f1_custom_kwargs():
    """Test F1 passes custom kwargs to sklearn."""
    outputs = ["A", "B"]
    labels = ["A", "C"]

    # Test with zero_division parameter override
    metric = F1(average="macro", zero_division=1)
    agg, items = metric.score(outputs, labels)

    # Should not raise and should use custom zero_division
    assert _key("macro") in agg
