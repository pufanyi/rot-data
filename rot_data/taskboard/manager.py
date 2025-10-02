"""Core task management for the task board."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single task."""

    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    total: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "metadata": self.metadata,
            "percentage": (
                (self.progress / self.total * 100)
                if self.total and self.total > 0
                else 0.0
            ),
        }


class TaskBoard:
    """
    Thread-safe task board for tracking multiple tasks.

    This can be used across different data loaders to track task progress.
    """

    _instance: TaskBoard | None = None
    _lock = threading.Lock()

    def __new__(cls) -> TaskBoard:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the task board."""
        if self._initialized:
            return
        self._initialized = True
        self._tasks: dict[str, Task] = {}
        self._task_lock = threading.Lock()
        self._subscribers: list[callable] = []
        self._subscriber_lock = threading.Lock()

    def add_task(
        self,
        task_id: str,
        name: str,
        total: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """
        Add a new task to the board.

        Args:
            task_id: Unique identifier for the task
            name: Display name for the task
            total: Total units of work (optional)
            metadata: Additional metadata for the task

        Returns:
            The created Task object
        """
        with self._task_lock:
            task = Task(
                id=task_id,
                name=name,
                total=total,
                metadata=metadata or {},
            )
            self._tasks[task_id] = task
            self._notify_subscribers()
            return task

    def start_task(self, task_id: str) -> None:
        """Mark a task as started."""
        with self._task_lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.RUNNING
                self._tasks[task_id].start_time = datetime.now()
                self._notify_subscribers()

    def update_task(
        self,
        task_id: str,
        progress: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update task progress.

        Args:
            task_id: Task identifier
            progress: Current progress value
            metadata: Additional metadata to update
        """
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if progress is not None:
                    task.progress = progress
                if metadata:
                    task.metadata.update(metadata)
                self._notify_subscribers()

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                if task.total is not None:
                    task.progress = task.total
                self._notify_subscribers()

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
                task.error = error
                self._notify_subscribers()

    def cancel_task(self, task_id: str) -> None:
        """Mark a task as cancelled."""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.end_time = datetime.now()
                self._notify_subscribers()

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the board."""
        with self._task_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._notify_subscribers()

    def clear_all(self) -> None:
        """Clear all tasks from the board."""
        with self._task_lock:
            self._tasks.clear()
            self._notify_subscribers()

    def get_task(self, task_id: str) -> Task | None:
        """Get a specific task by ID."""
        with self._task_lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> dict[str, Task]:
        """Get all tasks."""
        with self._task_lock:
            return self._tasks.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of all tasks."""
        with self._task_lock:
            total = len(self._tasks)
            by_status = {status.value: 0 for status in TaskStatus}
            for task in self._tasks.values():
                by_status[task.status.value] += 1

            return {
                "total": total,
                "by_status": by_status,
                "tasks": [task.to_dict() for task in self._tasks.values()],
            }

    def subscribe(self, callback: callable) -> None:
        """
        Subscribe to task updates.

        Args:
            callback: Function to call when tasks are updated
        """
        with self._subscriber_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: callable) -> None:
        """
        Unsubscribe from task updates.

        Args:
            callback: Function to remove from subscribers
        """
        with self._subscriber_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _notify_subscribers(self) -> None:
        """Notify all subscribers of task updates."""
        with self._subscriber_lock:
            subscribers = self._subscribers.copy()

        for callback in subscribers:
            try:
                callback()
            except Exception:
                pass  # Silently ignore subscriber errors


def get_task_board() -> TaskBoard:
    """Get the global task board instance."""
    return TaskBoard()

