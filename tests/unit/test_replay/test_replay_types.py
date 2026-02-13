"""Unit tests for replay types."""

import pytest

from scorebook.types import AdaptiveReplayRunResult, AdaptiveReplayRunSpec


class TestAdaptiveReplayRunSpec:
    """Tests for AdaptiveReplayRunSpec dataclass."""

    def test_creation_with_required_fields(self):
        """Test AdaptiveReplayRunSpec can be created with required fields."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_123",
            hyperparameter_config={"temperature": 0.7},
            hyperparameters_index=0,
            experiment_id="exp_1",
            project_id="proj_1",
        )
        assert spec.previous_run_id == "run_123"
        assert spec.hyperparameter_config == {"temperature": 0.7}
        assert spec.hyperparameters_index == 0
        assert spec.experiment_id == "exp_1"
        assert spec.project_id == "proj_1"
        assert spec.metadata is None

    def test_creation_with_metadata(self):
        """Test AdaptiveReplayRunSpec can be created with optional metadata."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_456",
            hyperparameter_config={},
            hyperparameters_index=1,
            experiment_id="exp_2",
            project_id="proj_2",
            metadata={"model": "gpt-4", "version": "1.0"},
        )
        assert spec.metadata == {"model": "gpt-4", "version": "1.0"}

    def test_creation_with_empty_hyperparameters(self):
        """Test AdaptiveReplayRunSpec can be created with empty hyperparameters."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_789",
            hyperparameter_config={},
            hyperparameters_index=0,
            experiment_id="exp_3",
            project_id="proj_3",
        )
        assert spec.hyperparameter_config == {}


class TestAdaptiveReplayRunResult:
    """Tests for AdaptiveReplayRunResult dataclass."""

    @pytest.fixture
    def sample_spec(self):
        """Create a sample AdaptiveReplayRunSpec for testing."""
        return AdaptiveReplayRunSpec(
            previous_run_id="run_123",
            hyperparameter_config={"temp": 0.5},
            hyperparameters_index=0,
            experiment_id="exp_1",
            project_id="proj_1",
            metadata={"model": "gpt-4"},
        )

    def test_creation_with_all_fields(self, sample_spec):
        """Test AdaptiveReplayRunResult can be created with all fields."""
        result = AdaptiveReplayRunResult(
            run_spec=sample_spec,
            run_completed=True,
            scores={"score": {"theta": 0.8, "std_error": 0.1}},
            run_id="new_run_456",
            replay_of_run="run_123",
        )
        assert result.run_spec == sample_spec
        assert result.run_completed is True
        assert result.scores == {"score": {"theta": 0.8, "std_error": 0.1}}
        assert result.run_id == "new_run_456"
        assert result.replay_of_run == "run_123"

    def test_creation_with_defaults(self, sample_spec):
        """Test AdaptiveReplayRunResult uses correct defaults."""
        result = AdaptiveReplayRunResult(
            run_spec=sample_spec,
            run_completed=False,
            scores={},
        )
        assert result.run_id is None
        assert result.replay_of_run is None

    def test_aggregate_scores_includes_all_fields(self, sample_spec):
        """Test aggregate_scores property includes all relevant fields."""
        result = AdaptiveReplayRunResult(
            run_spec=sample_spec,
            run_completed=True,
            scores={"overall": {"theta": 0.8, "std_error": 0.1}},
            run_id="new_run_456",
            replay_of_run="run_123",
        )

        agg = result.aggregate_scores
        assert agg["previous_run_id"] == "run_123"
        assert agg["experiment_id"] == "exp_1"
        assert agg["project_id"] == "proj_1"
        assert agg["replay_of_run"] == "run_123"
        assert agg["temp"] == 0.5  # From hyperparameter_config
        assert agg["model"] == "gpt-4"  # From metadata
        assert "overall" in agg  # From scores

    def test_aggregate_scores_with_empty_hyperparameters(self):
        """Test aggregate_scores works with empty hyperparameters."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_123",
            hyperparameter_config={},
            hyperparameters_index=0,
            experiment_id="exp_1",
            project_id="proj_1",
        )
        result = AdaptiveReplayRunResult(
            run_spec=spec,
            run_completed=True,
            scores={"theta": 0.5},
        )

        agg = result.aggregate_scores
        assert agg["previous_run_id"] == "run_123"
        assert agg["theta"] == 0.5

    def test_aggregate_scores_with_no_metadata(self):
        """Test aggregate_scores works without metadata."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_123",
            hyperparameter_config={"temp": 0.7},
            hyperparameters_index=0,
            experiment_id="exp_1",
            project_id="proj_1",
        )
        result = AdaptiveReplayRunResult(
            run_spec=spec,
            run_completed=True,
            scores={},
        )

        agg = result.aggregate_scores
        assert agg["temp"] == 0.7
        assert "model" not in agg

    def test_aggregate_scores_with_empty_scores(self):
        """Test aggregate_scores works with empty scores dict."""
        spec = AdaptiveReplayRunSpec(
            previous_run_id="run_123",
            hyperparameter_config={},
            hyperparameters_index=0,
            experiment_id="exp_1",
            project_id="proj_1",
        )
        result = AdaptiveReplayRunResult(
            run_spec=spec,
            run_completed=False,
            scores={},
        )

        agg = result.aggregate_scores
        assert agg["previous_run_id"] == "run_123"
        assert agg["replay_of_run"] is None

    def test_failed_replay_result(self, sample_spec):
        """Test AdaptiveReplayRunResult for a failed replay."""
        result = AdaptiveReplayRunResult(
            run_spec=sample_spec,
            run_completed=False,
            scores={},
            run_id=None,
            replay_of_run=None,
        )
        assert result.run_completed is False
        assert result.run_id is None
        assert result.replay_of_run is None
        assert result.scores == {}
