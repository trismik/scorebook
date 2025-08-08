"""Progress bar utilities for evaluation tracking."""

from contextlib import contextmanager
from typing import Any, Generator, List, Optional

from tqdm import tqdm


class EvaluationProgressBars:
    """Manages nested progress bars for evaluation tracking."""

    def __init__(self, datasets: List[Any], hyperparam_count: int) -> None:
        """Initialize progress bar manager.

        Args:
            datasets: List of datasets being evaluated
            hyperparam_count: Number of hyperparameter configurations per dataset
        """
        self.datasets = datasets
        self.hyperparam_count = hyperparam_count
        self.dataset_pbar: Optional[tqdm] = None
        self.hyperparam_pbar: Optional[tqdm] = None

    def start_dataset_progress(self) -> None:
        """Start the outer progress bar for datasets."""
        self.dataset_pbar = tqdm(
            total=len(self.datasets),
            desc="Datasets   ",
            unit="dataset",
            position=0,
            leave=True,
            ncols=80,
            bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt}",
        )

    def update_dataset_progress(self) -> None:
        """Update the dataset progress bar."""
        if self.dataset_pbar:
            self.dataset_pbar.update(1)

    def close_dataset_progress(self) -> None:
        """Close the dataset progress bar."""
        if self.dataset_pbar:
            self.dataset_pbar.close()
            self.dataset_pbar = None

    @contextmanager
    def hyperparam_progress_context(self) -> Generator[tqdm, None, None]:
        """Context manager for hyperparameter progress bar."""
        self.hyperparam_pbar = tqdm(
            total=self.hyperparam_count,
            desc="Hyperparams",
            unit="config",
            position=1,
            leave=False,
            ncols=80,
            bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt}",
        )
        try:
            yield self.hyperparam_pbar
        finally:
            self.hyperparam_pbar.close()
            self.hyperparam_pbar = None

    def update_hyperparam_progress(self) -> None:
        """Update the hyperparameter progress bar."""
        if self.hyperparam_pbar:
            self.hyperparam_pbar.update(1)


@contextmanager
def evaluation_progress(
    datasets: List[Any], hyperparam_count: int
) -> Generator[EvaluationProgressBars, None, None]:
    """Context manager for evaluation progress bars.

    Args:
        datasets: List of datasets being evaluated
        hyperparam_count: Number of hyperparameter configurations per dataset

    Yields:
        EvaluationProgressBars: Progress bar manager instance
    """
    progress_bars = EvaluationProgressBars(datasets, hyperparam_count)
    progress_bars.start_dataset_progress()
    try:
        yield progress_bars
    finally:
        progress_bars.close_dataset_progress()
