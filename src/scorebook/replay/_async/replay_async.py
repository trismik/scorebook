"""Async implementation for adaptive run replays."""

import logging
from typing import Any, Callable, Dict, Optional, Union

from trismik import TrismikAsyncClient
from trismik.types import TrismikRunMetadata

from scorebook.evaluate.evaluate_helpers import (
    create_trismik_async_client,
    get_model_name,
    make_trismik_inference,
)
from scorebook.types import AdaptiveReplayRunResult, AdaptiveReplayRunSpec

logger = logging.getLogger(__name__)


async def replay_async(
    inference: Callable,
    previous_run_id: str,
    experiment_id: str,
    project_id: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    show_progress: Optional[bool] = None,
) -> Union[Dict[str, Any], AdaptiveReplayRunResult]:
    """
    Replay a previous adaptive evaluation run with a new inference function.

    This runs the exact same questions in the exact same order as the original
    run, but with your new model/configuration. This enables fair comparisons
    between different models or hyperparameter settings.

    Args:
        inference: The async inference callable to evaluate
        previous_run_id: The ID of the original adaptive run to replay
        experiment_id: Experiment identifier
        project_id: Project identifier
        hyperparameters: Optional hyperparameters to pass to inference
        metadata: Optional metadata to attach to this replay run
        return_dict: If True, return results as a dict; if False, return
                     AdaptiveReplayRunResult object
        show_progress: If True, show progress during replay

    Returns:
        Replay results containing theta scores and std_error

    Example:
        >>> # Original run
        >>> result = await evaluate_async(
        ...     inference=model_v1,
        ...     datasets="my_test:adaptive",
        ...     experiment_id="exp1",
        ...     project_id="proj1",
        ...     return_dict=False
        ... )
        >>> original_run_id = result.run_results[0].run_id
        >>>
        >>> # Replay with new model
        >>> replay_result = await replay_async(
        ...     inference=model_v2,
        ...     previous_run_id=original_run_id,
        ...     experiment_id="exp1",
        ...     project_id="proj1",
        ... )
    """
    # Validate required parameters
    if not experiment_id or not project_id:
        raise ValueError("experiment_id and project_id are required for replay")

    if not previous_run_id:
        raise ValueError("previous_run_id is required for replay")

    # Build replay spec
    replay_spec = AdaptiveReplayRunSpec(
        previous_run_id=previous_run_id,
        hyperparameter_config=hyperparameters or {},
        hyperparameters_index=0,
        experiment_id=experiment_id,
        project_id=project_id,
        metadata=metadata,
    )

    # Create client and execute replay
    trismik_client = create_trismik_async_client()

    async with trismik_client:
        result = await execute_replay(
            inference=inference,
            replay_spec=replay_spec,
            trismik_client=trismik_client,
            metadata=metadata,
            show_progress=show_progress,
        )

    if return_dict:
        return result.aggregate_scores
    return result


async def execute_replay(
    inference: Callable,
    replay_spec: AdaptiveReplayRunSpec,
    trismik_client: TrismikAsyncClient,
    metadata: Optional[Dict[str, Any]] = None,
    show_progress: Optional[bool] = None,
) -> AdaptiveReplayRunResult:
    """Execute a single replay run."""
    logger.debug("Executing replay for previous run: %s", replay_spec.previous_run_id)

    try:
        # Create inference with bound hyperparameters
        async def inference_with_hyperparams(items: Any) -> Any:
            return await inference(items, **replay_spec.hyperparameter_config)

        # Build progress callback if needed
        on_progress = None
        if show_progress:

            def _on_progress(current: int, total: int) -> None:
                # Could integrate with progress bars here
                logger.debug("Replay progress: %d/%d", current, total)

            on_progress = _on_progress

        # Get model name for metadata
        model_name = get_model_name(inference, metadata)

        # Call trismik's run_replay
        trismik_results = await trismik_client.run_replay(
            previous_run_id=replay_spec.previous_run_id,
            run_metadata=TrismikRunMetadata(
                model_metadata=TrismikRunMetadata.ModelMetadata(name=model_name),
                test_configuration={},
                inference_setup={},
            ),
            item_processor=make_trismik_inference(inference_with_hyperparams),
            on_progress=on_progress,
            return_dict=False,
        )

        # Extract scores from the results
        scores = {}
        if hasattr(trismik_results, "score") and trismik_results.score:
            scores = {"score": trismik_results.score}
        elif hasattr(trismik_results, "scores") and trismik_results.scores:
            scores = trismik_results.scores

        # Make scores JSON serializable
        scores = _make_json_serializable(scores)

        # Extract run IDs
        run_id = trismik_results.run_id
        replay_of_run = getattr(trismik_results, "replay_of_run", None)

        logger.debug("Replay completed for %s, new run_id: %s", replay_spec.previous_run_id, run_id)

        return AdaptiveReplayRunResult(
            run_spec=replay_spec,
            run_completed=True,
            scores=scores,
            run_id=run_id,
            replay_of_run=replay_of_run,
        )

    except Exception as e:
        logger.warning("Failed to complete replay for %s: %s", replay_spec.previous_run_id, str(e))
        return AdaptiveReplayRunResult(
            run_spec=replay_spec,
            run_completed=False,
            scores={},
            run_id=None,
            replay_of_run=None,
        )


def _make_json_serializable(obj: Any) -> Any:
    """Convert AdaptiveTestScore objects to JSON-serializable dicts."""
    if hasattr(obj, "theta") and hasattr(obj, "std_error"):
        return {"theta": obj.theta, "std_error": obj.std_error}
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj
