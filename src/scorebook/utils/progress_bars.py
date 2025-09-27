"""Progress bar utilities for evaluation tracking."""

import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Generator, Optional

from tqdm import tqdm

# ANSI Color codes
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
LIGHT_GREEN = "\033[92m"  # Lighter green for upload stats
LIGHT_RED = "\033[91m"  # Lighter red for upload failure stats

# Blue shimmer colors for sweep effect
BLUE_BASE = "\033[34m"  # Standard blue (base color)
BLUE_HIGHLIGHT = "\033[1;34m"  # Bright blue (subtle highlight for sweep)
SHIMMER_WIDTH = 3  # Number of characters in the highlight sweep

# Spinner blue shimmer colors (cycle through these for spinner frames)
SPINNER_BLUE_COLORS = [
    "\033[34m",  # Standard blue
    "\033[1;34m",  # Bright blue
    "\033[94m",  # Light blue
    "\033[36m",  # Cyan
    "\033[1;36m",  # Bright cyan
    "\033[96m",  # Light cyan
]

# Progress bar configuration
PROGRESS_BAR_FORMAT = "{desc}|{bar}|"
HEADER_FORMAT = "{desc}"
SPINNER_INTERVAL_SECONDS = 0.08
TERMINAL_FALLBACK_SIZE = (120, 20)
MINIMUM_HEADER_SPACING = 3

# Spinner animation frames
SPINNER_FRAMES = ["⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓"]

# Progress bar labels
EVALUATIONS_LABEL = "Evaluations"
ITEMS_LABEL = "Items"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation progress tracking."""

    total_eval_runs: int
    total_items: int
    dataset_count: int
    hyperparam_count: int
    model_display: str

    @property
    def dataset_label(self) -> str:
        """Get the appropriate dataset label (singular/plural)."""
        return "Dataset" if self.dataset_count == 1 else "Datasets"

    @property
    def hyperparam_label(self) -> str:
        """Get the appropriate hyperparameter label (singular/plural)."""
        if self.hyperparam_count == 1:
            return "Hyperparam Configuration"
        return "Hyperparam Configurations"


class ProgressBarFormatter:
    """Handles formatting for progress bar descriptions and headers."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the formatter with configuration."""
        self.config = config
        self._label_width = max(len(EVALUATIONS_LABEL), len(ITEMS_LABEL))
        self._count_width = max(len(str(config.total_eval_runs)), len(str(config.total_items)), 1)

    def format_progress_description(self, label: str, completed: int, total: int) -> str:
        """Format a progress bar description with counts and percentage."""
        label_str = label.ljust(self._label_width)
        count_str = f"{completed:>{self._count_width}}/{total:>{self._count_width}}"

        if total > 0:
            percent = int((completed / total) * 100)
            percent_str = f"{percent:>3d}%"
        else:
            percent_str = " --%"

        return f"{label_str} {count_str} {percent_str} "

    @staticmethod
    def format_elapsed_time(elapsed_seconds: float) -> str:
        """Format elapsed time as mm:ss or hh:mm:ss."""
        total_seconds = int(max(elapsed_seconds, 0))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def format_header(
        self,
        spinner_frame: str,
        elapsed_seconds: float,
        completed_runs: int,
        failed_runs: int,
        uploaded_runs: int,
        upload_failed_runs: int,
        shimmer_text: str = "",
    ) -> str:
        """Compose the header line with spinner, elapsed time, and run statistics."""
        elapsed_str = ProgressBarFormatter.format_elapsed_time(elapsed_seconds)
        left_section = self._build_left_section(spinner_frame, elapsed_str, shimmer_text)
        right_section = ProgressBarFormatter._build_run_status_section(
            completed_runs, failed_runs, uploaded_runs, upload_failed_runs
        )

        return ProgressBarFormatter._combine_header_sections(left_section, right_section)

    def _build_left_section(
        self, spinner_frame: str, elapsed_str: str, shimmer_text: str = ""
    ) -> str:
        """Build the left section of the header with spinner and evaluation info."""
        # Apply shimmer effect to the model display name
        evaluating_text = f"Evaluating {self.config.model_display}"
        model_text = shimmer_text if shimmer_text else evaluating_text

        return (
            f"{spinner_frame} {model_text} ({elapsed_str}) | "
            f"{self.config.dataset_count} {self.config.dataset_label} | "
            f"{self.config.hyperparam_count} {self.config.hyperparam_label}"
        )

    @staticmethod
    def _build_run_status_section(
        completed_runs: int, failed_runs: int, uploaded_runs: int, upload_failed_runs: int
    ) -> tuple[str, str]:
        """Build the run status section with plain and colored versions."""
        # Build base run statistics
        run_parts = [f"RUNS PASSED: {completed_runs}"]
        colored_run_parts = [f"{GREEN}RUNS PASSED: {completed_runs}{RESET}"]

        if failed_runs > 0:
            run_parts.append(f"RUNS FAILED: {failed_runs}")
            colored_run_parts.append(f"{RED}RUNS FAILED: {failed_runs}{RESET}")

        # Add upload statistics if any uploads have occurred
        if uploaded_runs > 0 or upload_failed_runs > 0:
            run_parts.append(f"RUNS UPLOADED: {uploaded_runs}")
            colored_run_parts.append(f"{LIGHT_GREEN}RUNS UPLOADED: {uploaded_runs}{RESET}")

            if upload_failed_runs > 0:
                run_parts.append(f"UPLOADS FAILED: {upload_failed_runs}")
                colored_run_parts.append(f"{LIGHT_RED}UPLOADS FAILED: {upload_failed_runs}{RESET}")

        plain = f"[{', '.join(run_parts)}]"
        colored = f"[{', '.join(colored_run_parts)}]"

        return plain, colored

    @staticmethod
    def _combine_header_sections(left_section: str, right_sections: tuple[str, str]) -> str:
        """Combine left and right header sections with appropriate spacing."""
        plain_right, colored_right = right_sections

        # Calculate visual length (without ANSI codes) for proper spacing
        def visual_length(text: str) -> int:
            """Calculate the visual length of text, excluding ANSI escape codes."""
            import re

            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return len(ansi_escape.sub("", text))

        term_width = shutil.get_terminal_size(fallback=TERMINAL_FALLBACK_SIZE).columns
        left_visual_length = visual_length(left_section)
        right_visual_length = len(plain_right)  # plain_right has no ANSI codes

        spacing = term_width - left_visual_length - right_visual_length
        spacing = max(spacing, MINIMUM_HEADER_SPACING)

        return f"{left_section}{' ' * spacing}{colored_right}"


class SpinnerManager:
    """Manages spinner animation for the progress header."""

    def __init__(self) -> None:
        """Initialize the spinner manager."""
        self._frames = SpinnerManager._normalize_spinner_frames()
        self._cycle: Optional[cycle] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frame_width = len(self._frames[0]) if self._frames else 0
        self._shimmer_position = 0  # Position of the shimmer sweep
        self._spinner_color_index = 0  # Index for spinner color cycling

    @staticmethod
    def _normalize_spinner_frames() -> list[str]:
        """Normalize spinner frames to have consistent width."""
        if not SPINNER_FRAMES:
            return []

        width = max(len(frame) for frame in SPINNER_FRAMES)
        return [frame.ljust(width) for frame in SPINNER_FRAMES]

    def start(self, update_callback: Callable[[str], None]) -> None:
        """Start the spinner animation."""
        if self._thread is not None or not self._frames:
            return

        self._stop_event.clear()
        self._cycle = cycle(self._frames)
        self._thread = threading.Thread(target=self._animate, args=(update_callback,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner animation."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def get_initial_frame(self) -> str:
        """Get the first spinner frame with blue shimmer effect."""
        if not self._frames:
            return ""
        frame = self._frames[0]
        color = SPINNER_BLUE_COLORS[self._spinner_color_index % len(SPINNER_BLUE_COLORS)]
        return f"{color}{frame}{RESET}"

    def get_empty_frame(self) -> str:
        """Get an empty frame with the same width as spinner frames."""
        return " " * self.frame_width

    def get_next_spinner_frame(self) -> str:
        """Get the next spinner frame with blue shimmer effect."""
        if not self._frames or not self._cycle:
            return ""

        frame = next(self._cycle)
        color = SPINNER_BLUE_COLORS[self._spinner_color_index % len(SPINNER_BLUE_COLORS)]
        self._spinner_color_index += 1
        return f"{color}{frame}{RESET}"

    def get_shimmer_text(self, text: str) -> str:
        """Apply sweep shimmer effect to text, returning formatted string."""
        if not text:
            return text

        # Build the text in segments to avoid color bleeding
        result = ""
        i = 0

        while i < len(text):
            # Determine if we're in a highlight segment or base segment
            if self._shimmer_position <= i < self._shimmer_position + SHIMMER_WIDTH:
                # Start highlight segment
                highlight_chars = ""
                while (
                    i < len(text)
                    and self._shimmer_position <= i < self._shimmer_position + SHIMMER_WIDTH
                ):
                    highlight_chars += text[i]
                    i += 1
                result += f"{BLUE_HIGHLIGHT}{highlight_chars}{RESET}"
            else:
                # Start base segment
                base_chars = ""
                while i < len(text) and not (
                    self._shimmer_position <= i < self._shimmer_position + SHIMMER_WIDTH
                ):
                    base_chars += text[i]
                    i += 1
                result += f"{BLUE_BASE}{base_chars}{RESET}"

        # Advance shimmer position for next call
        self._shimmer_position += 1
        # Reset to beginning when we've swept past the end
        if self._shimmer_position >= len(text) + SHIMMER_WIDTH:
            self._shimmer_position = -SHIMMER_WIDTH

        return result

    def _animate(self, update_callback: Callable[[str], None]) -> None:
        """Continuously update the spinner animation."""
        while not self._stop_event.is_set() and self._cycle is not None:
            frame = self.get_next_spinner_frame()
            update_callback(frame)
            time.sleep(SPINNER_INTERVAL_SECONDS)


class EvaluationProgressBars:
    """Manages progress bars for evaluation runs and item processing."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize progress bar manager.

        Args:
            config: Configuration for the evaluation progress tracking
        """
        self.config = config
        self.formatter = ProgressBarFormatter(config)
        self.spinner = SpinnerManager()

        # Progress bar instances
        self._header_bar: Optional[tqdm] = None
        self._evaluations_bar: Optional[tqdm] = None
        self._items_bar: Optional[tqdm] = None

        # State tracking
        self.completed_runs = 0
        self.failed_runs = 0
        self.uploaded_runs = 0
        self.upload_failed_runs = 0
        self._start_time: Optional[float] = None

    def start_progress_bars(self) -> None:
        """Start the evaluation progress bars."""
        self._start_time = time.monotonic()

        # Initialize header bar with spinner
        initial_frame = self.spinner.get_initial_frame()
        evaluating_text = f"Evaluating {self.config.model_display}"
        initial_shimmer = self.spinner.get_shimmer_text(evaluating_text)
        header_desc = self.formatter.format_header(initial_frame, 0.0, 0, 0, 0, 0, initial_shimmer)
        self._header_bar = tqdm(
            total=0,
            desc=header_desc,
            position=0,
            leave=True,
            dynamic_ncols=True,
            bar_format=HEADER_FORMAT,
        )

        # Initialize evaluations progress bar
        eval_desc = self.formatter.format_progress_description(
            EVALUATIONS_LABEL, 0, self.config.total_eval_runs
        )
        self._evaluations_bar = tqdm(
            total=self.config.total_eval_runs,
            desc=eval_desc,
            unit="run",
            position=1,
            leave=True,
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
        )

        # Initialize items progress bar
        items_desc = self.formatter.format_progress_description(
            ITEMS_LABEL, 0, self.config.total_items
        )
        self._items_bar = tqdm(
            total=self.config.total_items,
            desc=items_desc,
            unit="item",
            position=2,
            leave=False,
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
        )

        self._refresh_progress_descriptions()
        self.spinner.start(self._update_header_spinner)

    def on_run_completed(self, items_processed: int, succeeded: bool) -> None:
        """Update progress when an evaluation run completes."""
        if succeeded:
            self.completed_runs += 1
        else:
            self.failed_runs += 1

        if self._evaluations_bar is not None:
            self._evaluations_bar.update(1)

        if self._items_bar is not None and items_processed:
            self._items_bar.update(items_processed)

        self._refresh_progress_descriptions()

    def on_upload_completed(self, succeeded: bool) -> None:
        """Update progress when an upload completes."""
        if succeeded:
            self.uploaded_runs += 1
        else:
            self.upload_failed_runs += 1

    def on_upload_success(self) -> None:
        """Update progress when an upload succeeds."""
        self.uploaded_runs += 1

    def on_upload_failed(self) -> None:
        """Update progress when an upload fails."""
        self.upload_failed_runs += 1

    def close_progress_bars(self) -> None:
        """Close all progress bars and cleanup resources."""
        self.spinner.stop()
        self._finalize_header()

        if self._items_bar is not None:
            self._items_bar.close()
            self._items_bar = None
        if self._evaluations_bar is not None:
            self._evaluations_bar.close()
            self._evaluations_bar = None
        if self._header_bar is not None:
            self._header_bar.close()
            self._header_bar = None

        self._start_time = None

    def _refresh_progress_descriptions(self) -> None:
        """Refresh progress bar descriptions to maintain alignment as counts change."""
        if self._evaluations_bar is not None:
            eval_desc = self.formatter.format_progress_description(
                EVALUATIONS_LABEL,
                min(self._evaluations_bar.n, self.config.total_eval_runs),
                self.config.total_eval_runs,
            )
            self._evaluations_bar.set_description_str(eval_desc, refresh=False)

        if self._items_bar is not None:
            items_desc = self.formatter.format_progress_description(
                ITEMS_LABEL,
                min(self._items_bar.n, self.config.total_items),
                self.config.total_items,
            )
            self._items_bar.set_description_str(items_desc, refresh=False)

        # Refresh both bars
        if self._evaluations_bar is not None:
            self._evaluations_bar.refresh()
        if self._items_bar is not None:
            self._items_bar.refresh()

    def _update_header_spinner(self, frame: str) -> None:
        """Update the header with a new spinner frame."""
        if self._header_bar is not None and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            evaluating_text = f"Evaluating {self.config.model_display}"
            shimmer_text = self.spinner.get_shimmer_text(evaluating_text)
            header_desc = self.formatter.format_header(
                frame,
                elapsed,
                self.completed_runs,
                self.failed_runs,
                self.uploaded_runs,
                self.upload_failed_runs,
                shimmer_text,
            )
            self._header_bar.set_description_str(header_desc, refresh=False)
            self._header_bar.refresh()

    def _finalize_header(self) -> None:
        """Finalize the header line without spinner animation."""
        if self._header_bar is not None and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            final_frame = self.spinner.get_empty_frame()
            # No shimmer for final header
            final_desc = self.formatter.format_header(
                final_frame,
                elapsed,
                self.completed_runs,
                self.failed_runs,
                self.uploaded_runs,
                self.upload_failed_runs,
                "",
            )
            self._header_bar.set_description_str(final_desc, refresh=True)


@contextmanager
def evaluation_progress_context(
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
    config = EvaluationConfig(
        total_eval_runs=total_eval_runs,
        total_items=total_items,
        dataset_count=dataset_count,
        hyperparam_count=hyperparam_count,
        model_display=model_display,
    )
    progress_bars = EvaluationProgressBars(config)
    progress_bars.start_progress_bars()
    try:
        yield progress_bars
    finally:
        progress_bars.close_progress_bars()
