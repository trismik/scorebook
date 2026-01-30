import pytest

from scorebook.metrics.recall import Recall


def _key(method: str) -> str:
    """Generate Recall metric key for given averaging method."""
    return f"Recall ({method})"


def test_recall_perfect_score():
    """Test Recall with all correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    metric = Recall()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 1.0}
    assert all(items)
    assert len(items) == len(outputs)


def test_recall_zero_score():
    """Test Recall with all incorrect predictions."""
    outputs = ["A", "B", "C"]
    labels = ["X", "Y", "Z"]

    metric = Recall()
    agg, items = metric.score(outputs, labels)

    assert agg[_key("macro")] == 0.0
    assert all(not x for x in items)
    assert len(items) == len(outputs)


def test_recall_partial_score():
    """Test Recall with some correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "X"]

    metric = Recall()
    agg, items = metric.score(outputs, labels)

    assert 0.0 < agg[_key("macro")] < 1.0
    assert items == [True, True, False]


def test_recall_empty_lists():
    """Test Recall with empty inputs."""
    outputs = []
    labels = []

    metric = Recall()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 0.0}
    assert items == []


def test_recall_macro_averaging():
    """Test Recall with explicit macro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Recall(average="macro")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert 0.0 <= agg[_key("macro")] <= 1.0
    assert items == [True, False, True, False]


def test_recall_micro_averaging():
    """Test Recall with micro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Recall(average="micro")
    agg, items = metric.score(outputs, labels)

    assert _key("micro") in agg
    assert agg[_key("micro")] == 0.5  # 2 correct out of 4


def test_recall_weighted_averaging():
    """Test Recall with weighted averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Recall(average="weighted")
    agg, items = metric.score(outputs, labels)

    assert _key("weighted") in agg
    assert 0.0 <= agg[_key("weighted")] <= 1.0


def test_recall_all_averaging():
    """Test Recall with 'all' averaging returns all three methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Recall(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert len(agg) == 3


def test_recall_list_of_methods():
    """Test Recall with a list of averaging methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Recall(average=["macro", "micro"])
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") not in agg
    assert len(agg) == 2


def test_recall_invalid_average_method():
    """Test Recall raises error for invalid averaging method."""
    with pytest.raises(ValueError, match="Invalid average method"):
        Recall(average="invalid")


def test_recall_invalid_average_method_in_list():
    """Test Recall raises error for invalid method in list."""
    with pytest.raises(ValueError, match="Invalid average method"):
        Recall(average=["macro", "invalid"])


def test_recall_all_combined_with_others():
    """Test Recall raises error when 'all' is combined with other methods."""
    with pytest.raises(ValueError, match="'all' cannot be combined"):
        Recall(average=["all", "macro"])


def test_recall_binary_classification():
    """Test Recall on binary classification task."""
    outputs = [1, 1, 0, 0, 1]
    labels = [1, 0, 0, 1, 1]

    metric = Recall()
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert items == [True, False, True, False, True]


def test_recall_multiclass_classification():
    """Test Recall on multi-class classification task."""
    outputs = ["cat", "dog", "bird", "cat", "dog"]
    labels = ["cat", "dog", "cat", "bird", "dog"]

    metric = Recall(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert items == [True, True, False, False, True]


def test_recall_custom_kwargs():
    """Test Recall passes custom kwargs to sklearn."""
    outputs = ["A", "B"]
    labels = ["A", "C"]

    # Test with zero_division parameter override
    metric = Recall(average="macro", zero_division=1)
    agg, items = metric.score(outputs, labels)

    # Should not raise and should use custom zero_division
    assert _key("macro") in agg
