"""Progress bar utilities for evaluation tracking."""

import shutil
import threading
import time
from contextlib import contextmanager
from itertools import cycle
from typing import Generator, Optional

from tqdm import tqdm

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"


BAR_FORMAT = "{desc}|{bar}|"


class EvaluationProgressBars:
    """Manages progress bars for evaluation runs and item processing."""

    def __init__(
        self,
        total_eval_runs: int,
        total_items: int,
        dataset_count: int,
        hyperparam_count: int,
        model_display: str,
    ) -> None:
        """Initialize progress bar manager.

        Args:
            total_eval_runs: Total number of evaluation runs scheduled
            total_items: Total number of evaluation items across all runs
            dataset_count: Number of datasets included in the evaluation
            hyperparam_count: Number of hyperparameter configurations evaluated
            model_display: Human readable model/inference name for the header
        """
        self.total_eval_runs = total_eval_runs
        self.total_items = total_items

        self.header_pbar: Optional[tqdm] = None
        self.eval_runs_pbar: Optional[tqdm] = None
        self.items_pbar: Optional[tqdm] = None

        self._eval_label = "Evaluations"
        self._items_label = "Items"
        self._label_width = max(len(self._eval_label), len(self._items_label))
        self._count_width = max(len(str(self.total_eval_runs)), len(str(self.total_items)), 1)
        self.dataset_count = dataset_count
        self.hyperparam_count = hyperparam_count
        self._dataset_label = "Dataset" if dataset_count == 1 else "Datasets"
        self._dataset_label_short = self._dataset_label
        if hyperparam_count == 1:
            self._hyperparam_label = "Hyperparameter configuration"
            self._hyperparam_label_short = "Hyperparam config"
        else:
            self._hyperparam_label = "Hyperparameter configurations"
            self._hyperparam_label_short = "Hyperparam configs"
        self.model_display = model_display
        self.completed_runs = 0
        self.failed_runs = 0
        self._start_time: Optional[float] = None
        self._spinner_frames = self._build_spinner_frames()
        self._spinner_cycle = cycle(self._spinner_frames) if self._spinner_frames else None
        self._spinner_interval = 0.15
        self._spinner_stop = threading.Event()
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_width = len(self._spinner_frames[0]) if self._spinner_frames else 0

    def start_progress_bars(self) -> None:
        """Start the evaluation progress bars."""
        self._start_time = time.monotonic()
        initial_frame = self._spinner_frames[0] if self._spinner_frames else ""
        header_desc = self._compose_header(initial_frame, 0.0)
        self.header_pbar = tqdm(
            total=0,
            desc=header_desc,
            position=0,
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}",
        )
        eval_desc = self._format_desc(self._eval_label, 0, self.total_eval_runs)
        self.eval_runs_pbar = tqdm(
            total=self.total_eval_runs,
            desc=eval_desc,
            unit="run",
            position=1,
            leave=True,
            dynamic_ncols=True,
            bar_format=BAR_FORMAT,
        )

        items_desc = self._format_desc(self._items_label, 0, self.total_items)
        self.items_pbar = tqdm(
            total=self.total_items,
            desc=items_desc,
            unit="item",
            position=2,
            leave=False,
            dynamic_ncols=True,
            bar_format=BAR_FORMAT,
        )

        self._refresh_descriptions()
        self._start_spinner()

    def on_run_completed(self, items_processed: int, succeeded: bool) -> None:
        """Update progress when an evaluation run completes."""
        if succeeded:
            self.completed_runs += 1
        else:
            self.failed_runs += 1

        if self.eval_runs_pbar is not None:
            self.eval_runs_pbar.update(1)

        if self.items_pbar is not None and items_processed:
            self.items_pbar.update(items_processed)

        self._refresh_descriptions()

    def close_progress_bars(self) -> None:
        """Close both progress bars."""
        self._stop_spinner()
        if self.items_pbar is not None:
            self.items_pbar.close()
            self.items_pbar = None
        if self.eval_runs_pbar is not None:
            self.eval_runs_pbar.close()
            self.eval_runs_pbar = None
        if self.header_pbar is not None:
            self.header_pbar.close()
            self.header_pbar = None
        self._start_time = None

    def _refresh_descriptions(self) -> None:
        """Refresh descriptions so bars stay aligned as counts change."""

        if self.eval_runs_pbar is not None:
            eval_desc = self._format_desc(
                self._eval_label,
                min(self.eval_runs_pbar.n, self.total_eval_runs),
                self.total_eval_runs,
            )
            self.eval_runs_pbar.set_description_str(eval_desc, refresh=False)

        if self.items_pbar is not None:
            items_desc = self._format_desc(
                self._items_label,
                min(self.items_pbar.n, self.total_items),
                self.total_items,
            )
            self.items_pbar.set_description_str(items_desc, refresh=False)

        if self.eval_runs_pbar is not None:
            self.eval_runs_pbar.refresh()
        if self.items_pbar is not None:
            self.items_pbar.refresh()

    def _format_desc(self, label: str, completed: int, total: int) -> str:
        """Return a padded description string with counts and percentage."""

        label_str = label.ljust(self._label_width)
        count_str = f"{completed:>{self._count_width}}/{total:>{self._count_width}}"
        if total > 0:
            percent = int((completed / total) * 100)
            percent_str = f"{percent:>3d}%"
        else:
            percent_str = " --%"
        return f"{label_str} {count_str} {percent_str} "

    def _build_spinner_frames(self) -> list[str]:
        """Return spinner frames that rotate three dots."""

        frames = ["⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"]
        width = max(len(frame) for frame in frames)
        return [frame.ljust(width) for frame in frames]

    def _start_spinner(self) -> None:
        """Start the spinner animation above the progress bars."""

        if (
            self.header_pbar is None
            or self._spinner_thread is not None
            or not self._spinner_frames
            or self._spinner_cycle is None
        ):
            return

        self._spinner_stop.clear()
        self._spinner_cycle = cycle(self._spinner_frames)
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self) -> None:
        """Stop the spinner animation and finalize the header line."""

        if self._spinner_thread is None:
            return

        self._spinner_stop.set()
        self._spinner_thread.join()
        self._spinner_thread = None

        if self.header_pbar is not None:
            elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
            final_frame = " " * self._spinner_width if self._spinner_width else ""
            final_desc = self._compose_header(final_frame, elapsed)
            self.header_pbar.set_description_str(final_desc, refresh=True)

    def _spin(self) -> None:
        """Continuously update the spinner while evaluations are running."""

        while not self._spinner_stop.is_set() and self._spinner_cycle is not None:
            frame = next(self._spinner_cycle)
            if self.header_pbar is not None:
                elapsed = (
                    time.monotonic() - self._start_time if self._start_time is not None else 0.0
                )
                header_desc = self._compose_header(frame, elapsed)
                self.header_pbar.set_description_str(header_desc, refresh=False)
                self.header_pbar.refresh()
            time.sleep(self._spinner_interval)

    def _compose_header(self, frame: str, elapsed_seconds: float) -> str:
        """Compose the header line with spinner and elapsed time."""

        if self._spinner_width:
            frame_str = frame if frame else " " * self._spinner_width
            if len(frame_str) < self._spinner_width:
                frame_str = frame_str.ljust(self._spinner_width)
        else:
            frame_str = frame

        elapsed_str = self._format_elapsed_time(elapsed_seconds)
        if self.failed_runs > 0:
            runs_section_plain = (
                f"[RUNS PASSED: {self.completed_runs}, RUNS FAILED: {self.failed_runs}]"
            )
            runs_section_display = (
                f"["
                f"{GREEN}RUNS PASSED: {self.completed_runs}{RESET}, "
                f"{RED}RUNS FAILED: {self.failed_runs}{RESET}"
                "]"
            )
        else:
            runs_section_plain = f"[RUNS PASSED: {self.completed_runs}]"
            runs_section_display = f"[{GREEN}RUNS PASSED: {self.completed_runs}{RESET}]"
        left_section_prefix = f"{frame_str} ({elapsed_str})" if frame_str else f"({elapsed_str})"
        left_section_plain = (
            f"{left_section_prefix} Evaluating {self.model_display} | "
            f"{self.dataset_count} {self._dataset_label_short} | "
            f"{self.hyperparam_count} {self._hyperparam_label_short}"
        )

        term_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        spacing = term_width - len(left_section_plain) - len(runs_section_plain)
        if spacing < 3:
            spacing = 3

        left_section_display = left_section_plain
        return f"{left_section_display}{' ' * spacing}{runs_section_display}"

    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """Format elapsed time as mm:ss or hh:mm:ss."""

        total_seconds = int(max(elapsed_seconds, 0))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


@contextmanager
def evaluation_progress(
    total_eval_runs: int,
    total_items: int,
    dataset_count: int,
    hyperparam_count: int,
    model_display: str,
) -> Generator[EvaluationProgressBars, None, None]:
    """Context manager for evaluation progress bars.

    Args:
        total_eval_runs: Total number of runs that will be executed
        total_items: Total number of evaluation items across all runs
        dataset_count: Number of datasets included in the evaluation
        hyperparam_count: Number of hyperparameter configurations evaluated
        model_display: Human readable model/inference name for the header

    Yields:
        EvaluationProgressBars: Progress bar manager instance
    """
    progress_bars = EvaluationProgressBars(
        total_eval_runs,
        total_items,
        dataset_count,
        hyperparam_count,
        model_display,
    )
    progress_bars.start_progress_bars()
    try:
        yield progress_bars
    finally:
        progress_bars.close_progress_bars()
