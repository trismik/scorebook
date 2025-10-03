"""Progress bar utilities for evaluation tracking."""

from contextlib import contextmanager
from typing import Generator, Optional

from tqdm import tqdm


class EvaluationProgressBars:
    """Manages nested progress bars for evaluation tracking."""

    def __init__(self, dataset_count: int, hyperparam_count: int, total_eval_runs: int) -> None:
        """Initialize progress bar manager.

        Args:
            dataset_count: Number of datasets being evaluated
            hyperparam_count: Number of hyperparameter configurations per dataset
            total_eval_runs: Total number of EvalRunSpecs (dataset_count * hyperparam_count)
        """
        self.dataset_count = dataset_count
        self.hyperparam_count = hyperparam_count
        self.total_eval_runs = total_eval_runs

        self.dataset_pbar: Optional[tqdm] = None
        self.hyperparam_pbar: Optional[tqdm] = None

        # Track progress per dataset
        self.current_dataset_idx = 0
        self.completed_hyperparams_per_dataset: dict[int, int] = {}
        self.completed_eval_runs = 0

    def start_progress_bars(self) -> None:
        """Start both progress bars."""
        # Top level: Datasets
        self.dataset_pbar = tqdm(
            total=self.dataset_count,
            desc="Datasets   ",
            unit="dataset",
            position=0,
            leave=True,
            ncols=80,
            bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt}",
        )

        # Bottom level: Eval runs
        self.hyperparam_pbar = tqdm(
            total=self.total_eval_runs,
            desc="Eval Runs  ",
            unit="run",
            position=1,
            leave=False,
            ncols=80,
            bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt}",
        )

    def on_eval_run_completed(self, dataset_idx: int) -> None:
        """Update progress when an eval run (EvalRunSpec) completes."""
        self.completed_eval_runs += 1
        if self.hyperparam_pbar:
            self.hyperparam_pbar.update(1)

        # Track how many runs completed for this dataset
        self.completed_hyperparams_per_dataset[dataset_idx] = (
            self.completed_hyperparams_per_dataset.get(dataset_idx, 0) + 1
        )

        # Check if this dataset is complete
        if self.completed_hyperparams_per_dataset[dataset_idx] == self.hyperparam_count:
            # Update dataset progress
            if self.dataset_pbar:
                self.dataset_pbar.update(1)

            # Reset hyperparameter progress for next dataset (if any)
            if dataset_idx < self.dataset_count - 1:
                if self.hyperparam_pbar:
                    self.hyperparam_pbar.reset()

    def close_progress_bars(self) -> None:
        """Close both progress bars."""
        if self.hyperparam_pbar:
            self.hyperparam_pbar.close()
            self.hyperparam_pbar = None
        if self.dataset_pbar:
            self.dataset_pbar.close()
            self.dataset_pbar = None


@contextmanager
def evaluation_progress(
    dataset_count: int, hyperparameter_config_count: int, run_count: int
) -> Generator[EvaluationProgressBars, None, None]:
    """Context manager for evaluation progress bars.

    Args:
        dataset_count: Number of datasets being evaluated
        hyperparameter_config_count: Number of hyperparameter configurations per dataset
        run_count: Total number of EvalRunSpecs

    Yields:
        EvaluationProgressBars: Progress bar manager instance
    """
    progress_bars = EvaluationProgressBars(dataset_count, hyperparameter_config_count, run_count)
    progress_bars.start_progress_bars()
    try:
        yield progress_bars
    finally:
        progress_bars.close_progress_bars()
