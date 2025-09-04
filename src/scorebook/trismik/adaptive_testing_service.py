"""Trismik adaptive testing service integration."""

import asyncio
import dataclasses
import inspect
from typing import Any, Callable, Iterable, Mapping

from trismik.adaptive_test import AdaptiveTest
from trismik.client_async import TrismikAsyncClient
from trismik.types import TrismikMultipleChoiceTextItem, TrismikRunMetadata

from .login import get_token


def run_adaptive_evaluation(
    inference_callable: Callable,
    dataset: str,
    project_id: str,
    experiment_id: str,
    metadata: Any,
) -> Any:
    """Run an adaptive evaluation using the Trismik API.

    Args:
        inference_callable: Function to run inference
        dataset: Dataset identifier
        project_id: Trismik project ID
        experiment_id: Experiment identifier
        metadata: Additional metadata

    Returns:
        Results from the adaptive evaluation
    """
    runner = AdaptiveTest(
        make_trismik_inference(inference_callable),
        client=TrismikAsyncClient(
            service_url="https://api-stage.trismik.com/adaptive-testing", api_key=get_token()
        ),
    )

    results = runner.run(
        dataset,
        project_id,
        experiment_id,
        run_metadata=TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="unknown"),
            test_configuration={},
            inference_setup={},
        ),
        return_dict=False,
    )

    return results


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
