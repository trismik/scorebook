from scorebook.metrics.exactmatch import ExactMatch


def test_exact_match_perfect_score():
    """Test ExactMatch with all correct predictions."""
    outputs = ["hello", "world", "test"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)
    assert len(items) == len(outputs)


def test_exact_match_zero_score():
    """Test ExactMatch with all incorrect predictions."""
    outputs = ["hello", "world", "test"]
    labels = ["foo", "bar", "baz"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 0.0}
    assert all(not item["exact_match"] for item in items)
    assert len(items) == len(outputs)


def test_exact_match_partial_score():
    """Test ExactMatch with some correct predictions."""
    outputs = ["hello", "world", "test"]
    labels = ["hello", "world", "different"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": True}, {"exact_match": True}, {"exact_match": False}]


def test_exact_match_empty_lists():
    """Test ExactMatch with empty inputs."""
    outputs = []
    labels = []

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 0.0}
    assert items == []


def test_exact_match_case_insensitive_default():
    """Test ExactMatch is case insensitive by default."""
    outputs = ["Hello", "WORLD", "TeSt"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_case_sensitive():
    """Test ExactMatch with case sensitivity enabled."""
    outputs = ["Hello", "world", "TeSt"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(case_insensitive=False)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1 / 3}
    assert items == [{"exact_match": False}, {"exact_match": True}, {"exact_match": False}]


def test_exact_match_strip_default():
    """Test ExactMatch strips whitespace by default."""
    outputs = ["  hello  ", "world\n", "\ttest"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_no_strip():
    """Test ExactMatch without stripping whitespace."""
    outputs = ["  hello  ", "world", "test"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(strip=False)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": False}, {"exact_match": True}, {"exact_match": True}]


def test_exact_match_both_options_disabled():
    """Test ExactMatch with both preprocessing options disabled."""
    outputs = ["  Hello  ", "world", "test"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(case_insensitive=False, strip=False)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": False}, {"exact_match": True}, {"exact_match": True}]


def test_exact_match_non_string_values():
    """Test ExactMatch with non-string values passes through unchanged."""
    outputs = [1, 2, 3]
    labels = [1, 2, 4]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": True}, {"exact_match": True}, {"exact_match": False}]


def test_exact_match_mixed_types():
    """Test ExactMatch with mixed string and non-string values."""
    outputs = ["hello", 42, "test"]
    labels = ["HELLO", 42, "TEST"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_empty_strings():
    """Test ExactMatch with empty strings."""
    outputs = ["", "hello", ""]
    labels = ["", "hello", "world"]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": True}, {"exact_match": True}, {"exact_match": False}]


def test_exact_match_whitespace_only():
    """Test ExactMatch with whitespace-only strings (stripped to empty)."""
    outputs = ["   ", "hello", "  \t\n  "]
    labels = ["", "hello", ""]

    metric = ExactMatch()
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_strip_punctuation():
    """Test ExactMatch with strip_punctuation enabled."""
    outputs = ["hello!", "?world", "...test..."]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(strip_punctuation=True)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_strip_punctuation_disabled():
    """Test ExactMatch with strip_punctuation disabled (default)."""
    outputs = ["hello!", "world", "test"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(strip_punctuation=False)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 2 / 3}
    assert items == [{"exact_match": False}, {"exact_match": True}, {"exact_match": True}]


def test_exact_match_strip_punctuation_only_edges():
    """Test that strip_punctuation only removes leading/trailing punctuation."""
    # Internal punctuation should NOT be removed
    outputs = ["hello, world!", "it's a test", "foo-bar"]
    labels = ["hello, world", "it's a test", "foo-bar"]

    metric = ExactMatch(strip_punctuation=True)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_strip_punctuation_preserves_internal():
    """Test that internal punctuation is preserved with strip_punctuation."""
    outputs = ["hello, world"]
    labels = ["hello world"]

    metric = ExactMatch(strip_punctuation=True)
    agg, items = metric.score(outputs, labels)

    # Should NOT match because comma is internal
    assert agg == {"exact_match": 0.0}
    assert items == [{"exact_match": False}]


def test_exact_match_strip_punctuation_multiple():
    """Test strip_punctuation with multiple punctuation marks at edges."""
    outputs = ["!!!hello???", "...world...", "((test))"]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(strip_punctuation=True)
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)


def test_exact_match_all_preprocessing_options():
    """Test ExactMatch with all preprocessing options enabled."""
    outputs = ["  !!!Hello!!!  ", "  ???WORLD???  ", "  ...Test...  "]
    labels = ["hello", "world", "test"]

    metric = ExactMatch(
        case_insensitive=True,
        strip=True,
        strip_punctuation=True,
    )
    agg, items = metric.score(outputs, labels)

    assert agg == {"exact_match": 1.0}
    assert all(item["exact_match"] for item in items)
