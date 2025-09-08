"""
Types package containing data structures and type definitions for the Scorebook framework.

This module provides core data types used throughout the framework for dataset handling
and evaluation results.
"""

from scorebook.types.eval_dataset import EvalDataset
from scorebook.types.eval_result import EvalResult
from scorebook.types.eval_run_spec import EvalRunSpec

__all__ = ["EvalDataset", "EvalResult", "EvalRunSpec"]
