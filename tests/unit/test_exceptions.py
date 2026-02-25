from scorebook.exceptions import AllRunsFailedError, EvaluationError, InferenceError, ScoreBookError


def test_all_runs_failed_error():
    """Verify AllRunsFailedError stores errors, formats message, and has correct inheritance."""
    errors = [
        ("dataset='ds1', hyperparameters={'a': 1}", InferenceError("bad key")),
        ("dataset='ds2', hyperparameters={'a': 2}", RuntimeError("timeout")),
    ]
    exc = AllRunsFailedError(errors)

    # Inheritance chain
    assert isinstance(exc, EvaluationError)
    assert isinstance(exc, ScoreBookError)
    assert isinstance(exc, Exception)

    # Error storage
    assert exc.errors is errors
    assert len(exc.errors) == 2

    # Message format
    msg = str(exc)
    assert msg.startswith("All 2 evaluation runs failed:")
    assert "InferenceError: bad key" in msg
    assert "RuntimeError: timeout" in msg
    assert "dataset='ds1'" in msg
    assert "dataset='ds2'" in msg
