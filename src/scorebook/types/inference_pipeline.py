"""
Inference pipeline implementation for processing items through model inference.

This module provides a pipeline structure for handling model inference tasks,
supporting preprocessing, model inference, and postprocessing steps in a
configurable way.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, cast


class InferencePipeline:
    """A pipeline for processing items through model inference.

    This class implements a three-stage pipeline that handles:
    1. Preprocessing of input items
    2. Model inference
    3. Postprocessing of model outputs


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

    async def run(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Execute the complete inference pipeline on a list of items.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        if self.preprocessor:
            input_items = [self.preprocessor(item) for item in items]
        else:
            input_items = items

        if asyncio.iscoroutinefunction(self.inference_function):
            inference_outputs = await self.inference_function(input_items, **hyperparameters)
        else:
            inference_outputs = self.inference_function(input_items, **hyperparameters)

        if self.postprocessor:
            return [self.postprocessor(inference_output) for inference_output in inference_outputs]
        else:
            return cast(List[Any], inference_outputs)

    async def __call__(self, items: List[Dict[str, Any]], **hyperparameters: Any) -> List[Any]:
        """Make the pipeline instance callable by wrapping the run method.

        Args:
            items: List of items to process through the pipeline
            **hyperparameters: Model-specific parameters for inference

        Returns:
            List of processed outputs after running through the complete pipeline
        """
        return await self.run(items, **hyperparameters)
