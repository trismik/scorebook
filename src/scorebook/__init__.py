"""
Scorebook package.

A Python project for scorebook functionality.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

from scorebook.dashboard.create_project import create_project, create_project_async
from scorebook.dashboard.credentials import login, logout, whoami
from scorebook.dashboard.upload_results import upload_result, upload_result_async
from scorebook.eval_datasets.eval_dataset import EvalDataset
from scorebook.evaluate._async.evaluate_async import evaluate_async
from scorebook.evaluate._sync.evaluate import evaluate
from scorebook.inference.inference_pipeline import InferencePipeline
from scorebook.metrics.core.metric_registry import scorebook_metric
from scorebook.score._async.score_async import score_async
from scorebook.score._sync.score import score
from scorebook.utils.render_template import render_template

__all__ = [
    "EvalDataset",
    "evaluate",
    "evaluate_async",
    "score",
    "score_async",
    "render_template",
    "login",
    "logout",
    "whoami",
    "InferencePipeline",
    "create_project",
    "create_project_async",
    "upload_result",
    "upload_result_async",
    "scorebook_metric",
]
