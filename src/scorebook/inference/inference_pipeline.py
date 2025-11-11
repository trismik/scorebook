"""
Inference pipeline implementation for processing items through model inference.

This module provides a pipeline structure for handling model inference tasks,
supporting preprocessing, model inference, and postprocessing steps in a
configurable way.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, cast

from scorebook.utils.async_utils import is_awaitable


class InferencePipeline:
    """A pipeline for processing items through model inference.

    This class implements a three-stage pipeline that handles:
    1. Preprocessing of input items
    2. Model inference
    3. Postprocessing of model outputs

    The pipeline automatically adapts to sync or async execution based on the
    inference function provided during initialization.

    Attributes:
        model: Name or identifier of the model being used
        preprocessor: Function to prepare items for model inference
        inference_function: Function that performs the actual model inference
        postprocessor: Function to process the model outputs
    """

    def __init__(
        self,
        model: str,
        inference_function: Callable,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
    ) -> None:
        """Initialize the inference pipeline.

        The pipeline will automatically become sync or async based on the
        inference_function provided.

        Args:
            model: Name or identifier of the model to use
            inference_function: Function that performs model inference
            preprocessor: Optional function to prepare items for inference.
            postprocessor: Optional function to process model outputs.
        """
        self.model: str = model
        self.inference_function = inference_function
        self.preprocessor: Optional[Callable] = preprocessor
        self.postprocessor: Optional[Callable] = postprocessor

        # Dynamically change the class to provide appropriate sync/async interface
        self.__class__ = (
            _AsyncInferencePipeline if is_awaitable(inference_function) else _SyncInferencePipeline
        )


class _SyncInferencePipeline(InferencePipeline):
    """Synchronous version of InferencePipeline."""

    def run(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Execute the complete inference pipeline synchronously.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        if self.preprocessor:
            input_items = [self.preprocessor(item, **hyperparameters) for item in items]
        else:
            input_items = items

        # Sync inference function - call directly
        inference_outputs = self.inference_function(input_items, **hyperparameters)

        if self.postprocessor:
            return [
                self.postprocessor(inference_output, **hyperparameters)
                for inference_output in inference_outputs
            ]
        else:
            return cast(List[Any], inference_outputs)

    def __call__(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Make the pipeline instance callable synchronously.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        return self.run(items, **hyperparameters)


class _AsyncInferencePipeline(InferencePipeline):
    """Asynchronous version of InferencePipeline."""

    async def run(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Execute the complete inference pipeline asynchronously.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        if self.preprocessor:
            input_items = [self.preprocessor(item, **hyperparameters) for item in items]
        else:
            input_items = items

        # Handle both sync and async inference functions
        if is_awaitable(self.inference_function):
            inference_outputs = await self.inference_function(input_items, **hyperparameters)
        else:
            # Run sync function in thread pool to avoid blocking
            inference_outputs = await asyncio.to_thread(
                self.inference_function, input_items, **hyperparameters
            )

        if self.postprocessor:
            return [
                self.postprocessor(inference_output, **hyperparameters)
                for inference_output in inference_outputs
            ]
        else:
            return cast(List[Any], inference_outputs)

    async def __call__(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Make the pipeline instance callable asynchronously.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        return await self.run(items, **hyperparameters)
