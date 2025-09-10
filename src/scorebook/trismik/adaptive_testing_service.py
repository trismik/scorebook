"""Trismik adaptive testing service integration."""

import asyncio
import dataclasses
import inspect
import logging
from typing import Any, Callable, Iterable, Mapping

from trismik.adaptive_test import AdaptiveTest
from trismik.client_async import TrismikAsyncClient
from trismik.types import TrismikMultipleChoiceTextItem, TrismikRunMetadata

from scorebook.types import AdaptiveEvalRunResult, AdaptiveEvalRunSpec

from .login import get_token

logger = logging.getLogger(__name__)


async def run_adaptive_evaluation(
    inference: Callable,
    adaptive_run_spec: AdaptiveEvalRunSpec,
    experiment_id: str,
    project_id: str,
    metadata: Any,
) -> AdaptiveEvalRunResult:
    """Run an adaptive evaluation using the Trismik API.

    Args:
        inference: Function to run inference
        adaptive_run_spec: Specification for the adaptive evaluation
        experiment_id: Experiment identifier
        project_id: Trismik project ID
        metadata: Additional metadata
    Returns:
        Results from the adaptive evaluation
    """
    runner = AdaptiveTest(
        make_trismik_inference(inference),
        client=TrismikAsyncClient(
            service_url="https://api-stage.trismik.com/adaptive-testing", api_key=get_token()
        ),
    )

    logger.debug(
        "test_id: %s, project_id: %s, experiment: %s ",
        adaptive_run_spec.dataset,
        project_id,
        experiment_id,
    )
    trismik_results = runner.run(
        adaptive_run_spec.dataset,
        project_id,
        experiment_id,
        run_metadata=TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="unknown"),
            test_configuration={},
            inference_setup={},
        ),
        return_dict=False,
    )

    # Convert TrismikRunResults to AdaptiveEvalRunResult
    # Extract scores from the Trismik results
    scores = {}
    if hasattr(trismik_results, "scores") and trismik_results.scores:
        scores = trismik_results.scores
    elif hasattr(trismik_results, "__dict__"):
        # If scores aren't directly available, include all attributes as scores
        scores = {k: v for k, v in trismik_results.__dict__.items() if not k.startswith("_")}

    # Convert AdaptiveTestScore objects to JSON-serializable dictionaries
    def make_json_serializable(obj: Any) -> Any:
        if hasattr(obj, "theta") and hasattr(obj, "std_error"):
            # This is likely an AdaptiveTestScore object
            return {"theta": obj.theta, "std_error": obj.std_error}
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    # Make scores JSON serializable
    scores = make_json_serializable(scores)

    return AdaptiveEvalRunResult(run_spec=adaptive_run_spec, scores=scores)


def make_trismik_inference(
    inference_function: Callable,
    return_list: bool = False,
) -> Callable[[Any], Any]:
    """Wrap an inference function for flexible input handling.

    Takes a function expecting list[dict] and makes it accept single dict
    or TrismikMultipleChoiceTextItem.
    """

    # Check if the inference function is async
    is_async = inspect.iscoroutinefunction(inference_function) or (
        hasattr(inference_function, "__call__")
        and inspect.iscoroutinefunction(inference_function.__call__)
    )

    def sync_trismik_inference_function(eval_items: Any, **kwargs: Any) -> Any:
        # Single TrismikMultipleChoiceTextItem dataclass
        if isinstance(eval_items, TrismikMultipleChoiceTextItem):
            eval_item_dict = dataclasses.asdict(eval_items)
            results = inference_function([eval_item_dict], **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results if return_list else results[0]

        # Single item (a mapping)
        if isinstance(eval_items, Mapping):
            results = inference_function([eval_items], **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results if return_list else results[0]

        # Iterable of items (but not a string/bytes)
        if isinstance(eval_items, Iterable) and not isinstance(eval_items, (str, bytes)):
            # Convert any TrismikMultipleChoiceTextItem instances to dicts
            converted_items = []
            for item in eval_items:
                if isinstance(item, TrismikMultipleChoiceTextItem):
                    converted_items.append(dataclasses.asdict(item))
                else:
                    converted_items.append(item)
            results = inference_function(converted_items, **kwargs)
            if is_async:
                results = asyncio.run(results)
            return results

        raise TypeError(
            "Expected a single item (Mapping[str, Any] or TrismikMultipleChoiceTextItem) "
            "or an iterable of such items."
        )

    return sync_trismik_inference_function
