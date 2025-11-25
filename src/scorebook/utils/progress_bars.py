"""Progress bar utilities for evaluation tracking."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

from tqdm.auto import tqdm


@dataclass
class EvaluationProgressBars:
    """Tracks progress for evaluation runs."""

    _runs_bar: tqdm
    _items_bar: tqdm
    completed_runs: int = field(default=0, init=False)
    failed_runs: int = field(default=0, init=False)
    uploaded_runs: int = field(default=0, init=False)
    upload_failed_runs: int = field(default=0, init=False)

    def on_run_completed(self, items_processed: int, succeeded: bool) -> None:
        """Update progress when an evaluation run completes.

        Args:
            items_processed: Number of items processed in this run.
                Pass 0 for adaptive evals (items tracked via on_item_progress).
            succeeded: Whether the run completed successfully.
        """
        self._runs_bar.update(1)
        if items_processed > 0:
            self._items_bar.update(items_processed)
        if succeeded:
            self.completed_runs += 1
        else:
            self.failed_runs += 1

    def on_item_progress(self, current: int, total: int) -> None:
        """Update progress for individual items (used by adaptive evaluations).

        Args:
            current: Current item count.
            total: Total item count.
        """
        self._items_bar.n = current
        if total != self._items_bar.total:
            self._items_bar.total = total
        self._items_bar.refresh()

    def on_upload_completed(self, succeeded: bool) -> None:
        """Update progress when an upload completes."""
        if succeeded:
            self.uploaded_runs += 1
        else:
            self.upload_failed_runs += 1


@contextmanager
def evaluation_progress_context(
    total_eval_runs: int,
    total_items: int,
    model_display: str,
    enabled: bool = True,
) -> Generator[Optional[EvaluationProgressBars], None, None]:
    """Context manager for evaluation progress bars.

    Args:
        total_eval_runs: Total number of evaluation runs.
        total_items: Total number of items across all runs.
        model_display: Model name to display in progress description.
        enabled: Whether to show progress bars.

    Yields:
        EvaluationProgressBars instance, or None if disabled.
    """
    if not enabled:
        yield None
        return

    runs_bar = tqdm(
        total=total_eval_runs,
        desc=f"Evaluating {model_display}",
        unit="run",
        leave=False,
    )
    items_bar = tqdm(
        total=total_items,
        desc="Items",
        unit="item",
        leave=False,
    )

    progress = EvaluationProgressBars(_runs_bar=runs_bar, _items_bar=items_bar)
    try:
        yield progress
    finally:
        items_bar.close()
        runs_bar.close()


@contextmanager
def scoring_progress_context(
    total_metrics: int,
    enabled: bool = True,
) -> Generator[Optional[tqdm], None, None]:
    """Context manager for scoring progress display.

    Args:
        total_metrics: Total number of metrics to score.
        enabled: Whether to show progress bar.

    Yields:
        tqdm progress bar instance, or None if disabled.
    """
    if not enabled:
        yield None
        return

    progress_bar = tqdm(
        total=total_metrics,
        desc="Scoring",
        unit="metric",
        leave=False,
    )

    try:
        yield progress_bar
    finally:
        progress_bar.close()
