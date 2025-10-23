"""Progress bar utilities for evaluation tracking."""

import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Generator, Optional, cast

from tqdm.auto import tqdm

_IS_NOTEBOOK: Optional[bool] = None


def _is_notebook() -> bool:
    """Detect if code is running in a Jupyter notebook environment.

    Uses lazy evaluation with caching for efficiency.
    """
    global _IS_NOTEBOOK
    if _IS_NOTEBOOK is None:
        try:
            shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
            _IS_NOTEBOOK = shell == "ZMQInteractiveShell"
        except NameError:
            _IS_NOTEBOOK = False
    return _IS_NOTEBOOK


# Color codes - ANSI for terminals, plain text for notebooks
RESET = "\033[0m"


def _make_color_func(ansi_code: str) -> Callable[[str], str]:
    """Create a color function that checks notebook status at runtime.

    Args:
        ansi_code: The ANSI escape code for the color (e.g., "32" for green)

    Returns:
        A function that formats text with the color, or returns plain text in notebooks
    """

    def color_func(text: str) -> str:
        if _is_notebook():
            return text
        return f"\033[{ansi_code}m{text}\033[0m"

    return color_func


# Color functions - automatically handle notebook vs terminal rendering
GREEN = _make_color_func("32")  # Green
RED = _make_color_func("31")  # Red
LIGHT_GREEN = _make_color_func("92")  # Light green
LIGHT_RED = _make_color_func("91")  # Light red
BLUE_BASE = _make_color_func("34")  # Blue
BLUE_HIGHLIGHT = _make_color_func("1;34")  # Bright blue


# Shimmer effect width (number of characters highlighted in sweep animation)
# Tested values: 2 (too subtle), 3 (optimal), 5 (too wide)
SHIMMER_WIDTH = 3

# Spinner blue shimmer colors for terminals (cycled for visual effect)
SPINNER_BLUE_COLORS = [
    "\033[34m",  # Standard blue
    "\033[1;34m",  # Bright blue
    "\033[94m",  # Light blue
    "\033[36m",  # Cyan
    "\033[1;36m",  # Bright cyan
    "\033[96m",  # Light cyan
]

# Progress bar configuration
PROGRESS_BAR_FORMAT = "{desc}|{bar}|"  # Compact format for progress bars
HEADER_FORMAT = "{desc}"  # Header shows only description, no bar

# Spinner update interval in seconds
# 0.08s = 12.5 Hz provides smooth animation without excessive CPU usage
# Lower values (0.05) cause flickering, higher values (0.2) appear choppy
SPINNER_INTERVAL_SECONDS = 0.08

# Terminal size fallback if detection fails
# 120 columns: Common wide terminal default
# 20 rows: Not used but required by shutil.get_terminal_size()
TERMINAL_FALLBACK_SIZE = (120, 20)

# Minimum spacing between header left and right sections
# Prevents sections from touching when terminal is narrow
MINIMUM_HEADER_SPACING = 3

# Spinner animation frames
SPINNER_FRAMES_UNICODE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_FRAMES_ASCII = ["|", "/", "-", "\\", "|", "/", "-", "\\"]


def _select_spinner_frames() -> list[str]:
    """Select appropriate spinner frames based on terminal capabilities."""
    import sys

    encoding = sys.stdout.encoding or "ascii"

    if encoding.lower() in ("utf-8", "utf8"):
        return SPINNER_FRAMES_UNICODE
    else:
        return SPINNER_FRAMES_ASCII


# Use Braille characters for smooth rotation (fallback to ASCII if needed)
SPINNER_FRAMES = _select_spinner_frames()

# Progress bar labels
EVALUATIONS_LABEL = "Evaluations"  # Label for run-level progress
ITEMS_LABEL = "Items"  # Label for item-level progress

# Compiled regex pattern for ANSI escape codes (used for calculating visual length)
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _visual_length(text: str) -> int:
    """Calculate the visual length of text, excluding ANSI escape codes."""
    return len(_ANSI_ESCAPE_PATTERN.sub("", text))


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
    """Handles formatting for progress bar descriptions and headers.

    This class is responsible for:
    - Formatting progress descriptions with aligned counts and percentages
    - Building header sections with spinner, timing, and statistics
    - Ensuring proper text alignment accounting for ANSI escape codes

    The formatter maintains consistent column widths based on the maximum
    number of digits needed for counts, ensuring progress bars don't shift
    as numbers increment.
    """

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
        colored_run_parts = [GREEN(f"RUNS PASSED: {completed_runs}")]

        if failed_runs > 0:
            run_parts.append(f"RUNS FAILED: {failed_runs}")
            colored_run_parts.append(RED(f"RUNS FAILED: {failed_runs}"))

        # Add upload statistics if any uploads have occurred
        if uploaded_runs > 0 or upload_failed_runs > 0:
            run_parts.append(f"RUNS UPLOADED: {uploaded_runs}")
            colored_run_parts.append(LIGHT_GREEN(f"RUNS UPLOADED: {uploaded_runs}"))

            if upload_failed_runs > 0:
                run_parts.append(f"UPLOADS FAILED: {upload_failed_runs}")
                colored_run_parts.append(LIGHT_RED(f"UPLOADS FAILED: {upload_failed_runs}"))

        plain = f"[{', '.join(run_parts)}]"
        colored = f"[{', '.join(colored_run_parts)}]"

        return plain, colored

    @staticmethod
    def _combine_header_sections(left_section: str, right_sections: tuple[str, str]) -> str:
        """Combine left and right header sections with appropriate spacing."""
        plain_right, colored_right = right_sections

        term_width = shutil.get_terminal_size(fallback=TERMINAL_FALLBACK_SIZE).columns
        left_visual_length = _visual_length(left_section)
        right_visual_length = len(plain_right)

        # Check for terminal width overflow
        total_content_width = left_visual_length + right_visual_length
        if total_content_width >= term_width - MINIMUM_HEADER_SPACING:
            # Terminal too narrow, truncate left section
            max_left_width = term_width - right_visual_length - MINIMUM_HEADER_SPACING - 3
            if max_left_width < 20:
                # Terminal impossibly narrow, just show right section
                return colored_right

            # Truncate left section (strip ANSI codes for simplicity)
            left_plain = _ANSI_ESCAPE_PATTERN.sub("", left_section)
            left_truncated = left_plain[:max_left_width] + "..."
            left_section = left_truncated
            left_visual_length = len(left_truncated)

        spacing = term_width - left_visual_length - right_visual_length
        spacing = max(spacing, MINIMUM_HEADER_SPACING)

        return f"{left_section}{' ' * spacing}{colored_right}"


class SpinnerManager:
    """Manages spinner animation for the progress header.

    Features:
    - Runs spinner animation in a background daemon thread
    - Applies blue color cycling to spinner frames (terminal only)
    - Provides shimmer sweep effect for text highlighting
    - Thread-safe state management with locks

    The spinner updates at SPINNER_INTERVAL_SECONDS frequency and
    automatically stops when stop() is called. In notebook environments,
    plain text frames are used without ANSI color codes. The daemon thread
    ensures the program can exit cleanly even if the spinner doesn't stop.
    """

    def __init__(self) -> None:
        """Initialize the spinner manager."""
        self._frames = SpinnerManager._normalize_spinner_frames()
        self._cycle: Optional[cycle] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frame_width = len(self._frames[0]) if self._frames else 0
        self._shimmer_position = 0  # Position of the shimmer sweep
        self._spinner_color_index = 0  # Index for spinner color cycling
        self._lock = threading.Lock()  # Protects spinner state

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

    def is_running(self) -> bool:
        """Check if the spinner animation is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Stop the spinner animation."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)

        if self._thread.is_alive():
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Spinner thread did not stop cleanly within 5 seconds")
            # Thread is daemon, so it will be killed on exit anyway

        self._thread = None

    def get_initial_frame(self) -> str:
        """Get the first spinner frame with blue shimmer effect (terminals only)."""
        if not self._frames:
            return ""
        frame = self._frames[0]

        # Return plain frame for notebooks (no ANSI colors)
        if _is_notebook():
            return frame

        # Add color codes for terminals
        color = SPINNER_BLUE_COLORS[self._spinner_color_index % len(SPINNER_BLUE_COLORS)]
        return f"{color}{frame}{RESET}"

    def get_empty_frame(self) -> str:
        """Get an empty frame with the same width as spinner frames."""
        return " " * self.frame_width

    def get_next_spinner_frame(self) -> str:
        """Get the next spinner frame with blue shimmer effect (terminals only)."""
        if not self._frames or not self._cycle:
            return ""

        frame = cast(str, next(self._cycle))

        # Return plain frame for notebooks (no ANSI colors)
        if _is_notebook():
            return frame

        # Add color codes for terminals (thread-safe)
        with self._lock:
            color = SPINNER_BLUE_COLORS[self._spinner_color_index % len(SPINNER_BLUE_COLORS)]
            self._spinner_color_index += 1
        return f"{color}{frame}{RESET}"

    def get_shimmer_text(self, text: str) -> str:
        """Apply sweep shimmer effect to text, returning formatted string."""
        if not text:
            return text

        # Get current shimmer position (thread-safe)
        with self._lock:
            shimmer_pos = self._shimmer_position
            self._shimmer_position += 1
            if self._shimmer_position >= len(text) + SHIMMER_WIDTH:
                self._shimmer_position = -SHIMMER_WIDTH

        # Build the text in segments using list (more efficient than string concat)
        result_parts = []
        i = 0

        while i < len(text):
            # Determine if we're in a highlight segment or base segment
            if shimmer_pos <= i < shimmer_pos + SHIMMER_WIDTH:
                # Start highlight segment
                highlight_start = i
                while i < len(text) and shimmer_pos <= i < shimmer_pos + SHIMMER_WIDTH:
                    i += 1
                result_parts.append(BLUE_HIGHLIGHT(text[highlight_start:i]))
            else:
                # Start base segment
                base_start = i
                while i < len(text) and not (shimmer_pos <= i < shimmer_pos + SHIMMER_WIDTH):
                    i += 1
                result_parts.append(BLUE_BASE(text[base_start:i]))

        return "".join(result_parts)

    def _animate(self, update_callback: Callable[[str], None]) -> None:
        """Continuously update the spinner animation."""
        import logging

        logger = logging.getLogger(__name__)

        while not self._stop_event.is_set() and self._cycle is not None:
            try:
                frame = self.get_next_spinner_frame()
                update_callback(frame)
                time.sleep(SPINNER_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(
                    f"Non-critical: Spinner animation thread encountered an error "
                    f"and will stop. Progress bars will continue without animation. "
                    f"Details: {e}",
                    exc_info=True,
                )
                break  # Exit gracefully rather than crash silently


class EvaluationProgressBars:
    """Manages progress bars for evaluation runs and item processing.

    This class coordinates multiple progress displays:
    - Terminal mode: header bar + evaluations bar + items bar
    - Notebook mode: single simplified evaluations bar

    Thread Safety:
    All state updates (completed_runs, failed_runs, etc.) are protected
    by _state_lock to prevent race conditions with the spinner thread.

    Lifecycle:
    1. __init__: Initialize with configuration
    2. start_progress_bars: Create and display bars
    3. on_run_completed: Update when runs finish
    4. on_upload_completed: Update when uploads finish
    5. close_progress_bars: Clean up and show summary
    """

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
        self._state_lock = threading.Lock()  # Protects run counters

    def start_progress_bars(self) -> None:
        """Start the evaluation progress bars."""
        self._start_time = time.monotonic()

        try:
            self._initialize_progress_bars()
        except Exception:
            # Ensure spinner is stopped if initialization fails
            self.spinner.stop()
            raise

    def _initialize_progress_bars(self) -> None:
        """Initialize progress bars based on environment."""
        if _is_notebook():
            # Simplified notebook version - just one progress bar for evaluation runs
            spinner_frame = SPINNER_FRAMES[0] if SPINNER_FRAMES else ""
            desc = (
                f"{spinner_frame} Evaluating {self.config.model_display} | "
                f"{self.config.dataset_count} {self.config.dataset_label} | "
                f"{self.config.hyperparam_count} {self.config.hyperparam_label}"
            )
            self._evaluations_bar = tqdm(
                total=self.config.total_eval_runs,
                desc=desc,
                unit="run",
                leave=False,
                bar_format="{desc} | {n}/{total} Runs {percentage:3.0f}%|{bar}|",
            )
            # Start spinner animation for notebooks
            self.spinner.start(self._update_notebook_spinner)
        else:
            # Full terminal version with header, spinner, and multiple bars
            initial_frame = self.spinner.get_initial_frame()
            evaluating_text = f"Evaluating {self.config.model_display}"
            initial_shimmer = self.spinner.get_shimmer_text(evaluating_text)
            header_desc = self.formatter.format_header(
                initial_frame, 0.0, 0, 0, 0, 0, initial_shimmer
            )
            self._header_bar = tqdm(
                total=0,
                desc=header_desc,
                leave=False,
                dynamic_ncols=True,
                bar_format=HEADER_FORMAT,
            )

            eval_desc = self.formatter.format_progress_description(
                EVALUATIONS_LABEL, 0, self.config.total_eval_runs
            )
            self._evaluations_bar = tqdm(
                total=self.config.total_eval_runs,
                desc=eval_desc,
                unit="run",
                leave=False,
                dynamic_ncols=True,
                bar_format=PROGRESS_BAR_FORMAT,
            )

            items_desc = self.formatter.format_progress_description(
                ITEMS_LABEL, 0, self.config.total_items
            )
            self._items_bar = tqdm(
                total=self.config.total_items,
                desc=items_desc,
                unit="item",
                leave=False,
                dynamic_ncols=True,
                bar_format=PROGRESS_BAR_FORMAT,
            )

            self._refresh_progress_descriptions()
            self.spinner.start(self._update_header_spinner)

    def on_run_completed(self, items_processed: int, succeeded: bool) -> None:
        """Update progress when an evaluation run completes."""
        with self._state_lock:
            if succeeded:
                self.completed_runs += 1
            else:
                self.failed_runs += 1

        if self._evaluations_bar is not None:
            self._evaluations_bar.update(1)

        if self._items_bar is not None:
            self._items_bar.update(items_processed)

        self._refresh_progress_descriptions()

    def on_upload_completed(self, succeeded: bool) -> None:
        """Update progress when an upload completes."""
        with self._state_lock:
            if succeeded:
                self.uploaded_runs += 1
            else:
                self.upload_failed_runs += 1

        # Trigger header refresh in terminal mode
        if not _is_notebook() and self._header_bar is not None:
            self._refresh_header()

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

        # Print summary after clearing progress bars
        self._print_summary()

    def _refresh_progress_descriptions(self) -> None:
        """Refresh progress bar descriptions to maintain alignment as counts change."""
        # Skip refresh in notebooks (spinner handles description updates)
        if _is_notebook():
            return

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

    def _update_notebook_spinner(self, frame: str) -> None:
        """Update the notebook progress bar spinner (notebooks only)."""
        if self._evaluations_bar is not None:
            desc = (
                f"{frame} Evaluating {self.config.model_display} | "
                f"{self.config.dataset_count} {self.config.dataset_label} | "
                f"{self.config.hyperparam_count} {self.config.hyperparam_label}"
            )
            self._evaluations_bar.set_description_str(desc, refresh=False)
            self._evaluations_bar.refresh()

    def _update_header_spinner(self, frame: str) -> None:
        """Update the header with a new spinner frame (terminals only)."""
        if self._header_bar is not None and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            evaluating_text = f"Evaluating {self.config.model_display}"
            shimmer_text = self.spinner.get_shimmer_text(evaluating_text)

            # Read state with lock
            with self._state_lock:
                completed = self.completed_runs
                failed = self.failed_runs
                uploaded = self.uploaded_runs
                upload_failed = self.upload_failed_runs

            header_desc = self.formatter.format_header(
                frame,
                elapsed,
                completed,
                failed,
                uploaded,
                upload_failed,
                shimmer_text,
            )
            self._header_bar.set_description_str(header_desc, refresh=False)
            self._header_bar.refresh()

    def _refresh_header(self) -> None:
        """Refresh the header bar with current statistics."""
        if self._header_bar is None or self._start_time is None:
            return

        elapsed = time.monotonic() - self._start_time

        # Get current spinner frame (or empty if stopped)
        if self.spinner.is_running():
            # Spinner running, will update via callback soon
            return
        else:
            # Spinner stopped, update manually
            frame = self.spinner.get_empty_frame()

        with self._state_lock:
            completed = self.completed_runs
            failed = self.failed_runs
            uploaded = self.uploaded_runs
            upload_failed = self.upload_failed_runs

        header_desc = self.formatter.format_header(
            frame, elapsed, completed, failed, uploaded, upload_failed, ""
        )
        self._header_bar.set_description_str(header_desc, refresh=True)

    def _finalize_header(self) -> None:
        """Finalize the header line without spinner animation."""
        # Only for terminal mode
        if _is_notebook():
            return

        if self._header_bar is not None and self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            final_frame = self.spinner.get_empty_frame()

            # Read state with lock
            with self._state_lock:
                completed = self.completed_runs
                failed = self.failed_runs
                uploaded = self.uploaded_runs
                upload_failed = self.upload_failed_runs

            # No shimmer for final header
            final_desc = self.formatter.format_header(
                final_frame, elapsed, completed, failed, uploaded, upload_failed, ""
            )
            self._header_bar.set_description_str(final_desc, refresh=True)

    def _print_summary(self) -> None:
        """Print a clean summary after evaluation completes."""
        # Build summary message
        summary_parts = [f"Evaluating {self.config.model_display} Completed"]

        # Add run completion info
        total_runs = self.completed_runs + self.failed_runs
        expected_runs = self.config.total_eval_runs

        # Show if some runs didn't complete (cancelled/interrupted)
        if total_runs < expected_runs:
            summary_parts.append(
                f"{self.completed_runs}/{total_runs} Runs Completed Successfully "
                f"(out of {expected_runs} expected)"
            )
        elif self.failed_runs == 0:
            summary_parts.append(f"{self.completed_runs} Runs Completed Successfully")
        else:
            summary_parts.append(f"{self.completed_runs}/{total_runs} Runs Completed Successfully")

        # Add upload info if any uploads occurred
        if self.uploaded_runs > 0 or self.upload_failed_runs > 0:
            total_uploads = self.uploaded_runs + self.upload_failed_runs
            if self.upload_failed_runs == 0:
                summary_parts.append(f"{self.uploaded_runs} Runs Uploaded Successfully")
            else:
                summary_parts.append(
                    f"{self.uploaded_runs}/{total_uploads} Runs Uploaded Successfully"
                )

        # Join parts with ", " and print
        summary = ", ".join(summary_parts)
        print(summary)


@contextmanager
def evaluation_progress_context(
    total_eval_runs: int,
    total_items: int,
    dataset_count: int,
    hyperparam_count: int,
    model_display: str,
    enabled: bool = True,
) -> Generator[Optional[EvaluationProgressBars], None, None]:
    """Context manager for evaluation progress bars.

    Args:
        total_eval_runs: Total number of runs that will be executed
        total_items: Total number of evaluation items across all runs
        dataset_count: Number of datasets included in the evaluation
        hyperparam_count: Number of hyperparameter configurations evaluated
        model_display: Human readable model/inference name for the header
        enabled: Whether to show progress bars (default: True)

    Yields:
        Optional[EvaluationProgressBars]: Progress bar manager instance (None if disabled)
    """
    if not enabled:
        yield None
        return

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
