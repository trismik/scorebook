import pytest

from scorebook.metrics.precision import Precision


def _key(method: str) -> str:
    """Generate Precision metric key for given averaging method."""
    return f"Precision ({method})"


def test_precision_perfect_score():
    """Test Precision with all correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "C"]

    metric = Precision()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 1.0}
    assert all(items)
    assert len(items) == len(outputs)


def test_precision_zero_score():
    """Test Precision with all incorrect predictions."""
    outputs = ["A", "B", "C"]
    labels = ["X", "Y", "Z"]

    metric = Precision()
    agg, items = metric.score(outputs, labels)

    assert agg[_key("macro")] == 0.0
    assert all(not x for x in items)
    assert len(items) == len(outputs)


def test_precision_partial_score():
    """Test Precision with some correct predictions."""
    outputs = ["A", "B", "C"]
    labels = ["A", "B", "X"]

    metric = Precision()
    agg, items = metric.score(outputs, labels)

    assert 0.0 < agg[_key("macro")] < 1.0
    assert items == [True, True, False]


def test_precision_empty_lists():
    """Test Precision with empty inputs."""
    outputs = []
    labels = []

    metric = Precision()
    agg, items = metric.score(outputs, labels)

    assert agg == {_key("macro"): 0.0}
    assert items == []


def test_precision_mismatched_lengths():
    """Test Precision raises error for mismatched list lengths."""
    outputs = ["A", "B"]
    labels = ["A", "B", "C"]

    metric = Precision()
    with pytest.raises(ValueError, match="Number of outputs must match number of labels"):
        metric.score(outputs, labels)


def test_precision_macro_averaging():
    """Test Precision with explicit macro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Precision(average="macro")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert 0.0 <= agg[_key("macro")] <= 1.0
    assert items == [True, False, True, False]


def test_precision_micro_averaging():
    """Test Precision with micro averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Precision(average="micro")
    agg, items = metric.score(outputs, labels)

    assert _key("micro") in agg
    assert agg[_key("micro")] == 0.5  # 2 correct out of 4


def test_precision_weighted_averaging():
    """Test Precision with weighted averaging."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Precision(average="weighted")
    agg, items = metric.score(outputs, labels)

    assert _key("weighted") in agg
    assert 0.0 <= agg[_key("weighted")] <= 1.0


def test_precision_all_averaging():
    """Test Precision with 'all' averaging returns all three methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Precision(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert len(agg) == 3


def test_precision_list_of_methods():
    """Test Precision with a list of averaging methods."""
    outputs = ["A", "A", "B", "B"]
    labels = ["A", "B", "B", "A"]

    metric = Precision(average=["macro", "micro"])
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") not in agg
    assert len(agg) == 2


def test_precision_invalid_average_method():
    """Test Precision raises error for invalid averaging method."""
    with pytest.raises(ValueError, match="Invalid average method"):
        Precision(average="invalid")


def test_precision_invalid_average_method_in_list():
    """Test Precision raises error for invalid method in list."""
    with pytest.raises(ValueError, match="Invalid average method"):
        Precision(average=["macro", "invalid"])


def test_precision_all_combined_with_others():
    """Test Precision raises error when 'all' is combined with other methods."""
    with pytest.raises(ValueError, match="'all' cannot be combined"):
        Precision(average=["all", "macro"])


def test_precision_binary_classification():
    """Test Precision on binary classification task."""
    outputs = [1, 1, 0, 0, 1]
    labels = [1, 0, 0, 1, 1]

    metric = Precision()
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert items == [True, False, True, False, True]


def test_precision_multiclass_classification():
    """Test Precision on multi-class classification task."""
    outputs = ["cat", "dog", "bird", "cat", "dog"]
    labels = ["cat", "dog", "cat", "bird", "dog"]

    metric = Precision(average="all")
    agg, items = metric.score(outputs, labels)

    assert _key("macro") in agg
    assert _key("micro") in agg
    assert _key("weighted") in agg
    assert items == [True, True, False, False, True]


def test_precision_custom_kwargs():
    """Test Precision passes custom kwargs to sklearn."""
    outputs = ["A", "B"]
    labels = ["A", "C"]

    # Test with zero_division parameter override
    metric = Precision(average="macro", zero_division=1)
    agg, items = metric.score(outputs, labels)

    # Should not raise and should use custom zero_division
    assert _key("macro") in agg
