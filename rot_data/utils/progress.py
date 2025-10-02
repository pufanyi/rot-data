"""Global progress bar manager for coordinating multiple progress bars."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class SmartProgressColumn(ProgressColumn):
    """
    A smart progress column that displays bytes in human-readable format for
    download tasks, and regular count format for other tasks.
    """

    def render(self, task: Task) -> Text:
        """Render the progress column based on task type."""
        # Check if this is a download task (has is_download metadata)
        is_download = task.fields.get("is_download", False)

        if is_download:
            # Use human-readable byte format for downloads
            completed = task.completed
            total = task.total

            if total is None:
                return Text(self._format_bytes(completed), style="progress.download")

            return Text(
                f"{self._format_bytes(completed)}/{self._format_bytes(total)}",
                style="progress.download",
            )
        else:
            # Use count format for other tasks
            completed = int(task.completed)
            total = task.total

            if total is None:
                return Text(str(completed), style="progress.download")

            return Text(f"{completed}/{int(total)}", style="progress.download")

    @staticmethod
    def _format_bytes(size: float) -> str:
        """Format bytes in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"


class SmartSpeedColumn(ProgressColumn):
    """
    A smart speed column that only shows transfer speed for download tasks.
    For other tasks, it shows nothing.
    """

    def render(self, task: Task) -> Text:
        """Render the speed column only for download tasks."""
        is_download = task.fields.get("is_download", False)

        if not is_download:
            # Don't show speed for non-download tasks
            return Text("")

        # Show transfer speed for download tasks
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.data.speed")

        # Format speed in human-readable format
        return Text(f"{self._format_speed(speed)}", style="progress.data.speed")

    @staticmethod
    def _format_speed(speed: float) -> str:
        """Format speed in human-readable format."""
        for unit in ["B/s", "KB/s", "MB/s", "GB/s"]:
            if speed < 1024.0:
                return f"{speed:.1f} {unit}"
            speed /= 1024.0
        return f"{speed:.1f} TB/s"


class ProgressManager:
    """
    Singleton manager for coordinating progress bars across threads.

    All progress bars are displayed in a single Progress instance at the bottom
    of the terminal, with the overall progress bar shown at the top.
    """

    _instance: ProgressManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ProgressManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._progress: Progress | None = None
        self._overall_task_id: TaskID | None = None
        self._active = False
        self._console = Console(stderr=True)

    @contextmanager
    def managed_progress(self, total_tasks: int | None = None):
        """
        Context manager that creates and manages the shared Progress instance.

        Args:
            total_tasks: Total number of tasks to process (for overall progress bar)
        """
        with self._lock:
            if self._active:
                raise RuntimeError("Progress manager is already active")
            self._active = True

        try:
            # Create a single Progress instance with all columns
            # SmartProgressColumn automatically formats bytes for downloads
            # and counts for other task types.
            # SmartSpeedColumn only shows speed for download tasks
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                SmartProgressColumn(),
                SmartSpeedColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
            )

            with self._progress:
                # Add overall task progress bar at the top
                if total_tasks is not None and total_tasks > 0:
                    self._overall_task_id = self._progress.add_task(
                        "[bold magenta]Overall Progress",
                        total=total_tasks,
                    )
                else:
                    self._overall_task_id = None

                yield self

        finally:
            with self._lock:
                self._progress = None
                self._overall_task_id = None
                self._active = False

    def add_task(self, description: str, **kwargs: Any) -> TaskID:
        """Add a new task to the progress display."""
        if self._progress is None:
            raise RuntimeError("Progress manager is not active")
        return self._progress.add_task(description, **kwargs)

    def add_download_task(self, description: str, **kwargs: Any) -> TaskID:
        """
        Add a download task with byte-formatted progress display.

        The progress will be displayed in human-readable format (e.g., 4.46 GB/18.65 GB)
        instead of raw bytes.
        """
        if self._progress is None:
            raise RuntimeError("Progress manager is not active")
        # Mark this task as a download task for SmartProgressColumn
        kwargs["is_download"] = True
        return self._progress.add_task(description, **kwargs)

    def update(self, task_id: TaskID, **kwargs: Any) -> None:
        """Update a task's progress."""
        if self._progress is None:
            raise RuntimeError("Progress manager is not active")
        self._progress.update(task_id, **kwargs)

    def advance_overall(self, advance: int = 1) -> None:
        """Advance the overall progress bar."""
        if self._progress is None:
            raise RuntimeError("Progress manager is not active")
        if self._overall_task_id is not None:
            self._progress.update(self._overall_task_id, advance=advance)

    def remove_task(self, task_id: TaskID) -> None:
        """Remove a task from the progress display."""
        if self._progress is None:
            raise RuntimeError("Progress manager is not active")
        self._progress.remove_task(task_id)

    @property
    def is_active(self) -> bool:
        """Check if the progress manager is currently active."""
        return self._active

    @property
    def console(self) -> Console:
        """Get the shared console instance."""
        return self._console


# Global instance accessor
_manager = ProgressManager()


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    return _manager
