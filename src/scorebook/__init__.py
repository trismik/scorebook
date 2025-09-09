"""
Scorebook package.

A Python project for scorebook functionality.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

from scorebook.eval_dataset import EvalDataset
from scorebook.evaluate import evaluate
from scorebook.inference_pipeline import InferencePipeline
from scorebook.trismik.login import login
from scorebook.utils.build_prompt import build_prompt

__all__ = ["EvalDataset", "evaluate", "build_prompt", "login", "InferencePipeline"]
