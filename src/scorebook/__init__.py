"""
Scorebook package.

A Python project for scorebook functionality.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

from scorebook.eval_datasets import EvalDataset
from scorebook.evaluate import evaluate, evaluate_async
from scorebook.inference.inference_pipeline import InferencePipeline
from scorebook.trismik.credentials import login, whoami
from scorebook.utils.build_prompt import build_prompt

__all__ = [
    "EvalDataset",
    "evaluate",
    "evaluate_async",
    "build_prompt",
    "login",
    "whoami",
    "InferencePipeline",
]
