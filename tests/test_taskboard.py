"""Tests for the task board functionality."""

from rot_data.taskboard import TaskBoard, TaskStatus


def test_taskboard_singleton():
    """Test that TaskBoard is a singleton."""
    board1 = TaskBoard()
    board2 = TaskBoard()
    assert board1 is board2


def test_add_task():
    """Test adding a task."""
    board = TaskBoard()
    board.clear_all()
    
    task = board.add_task("test_1", "Test Task", total=100)
    assert task.id == "test_1"
    assert task.name == "Test Task"
    assert task.total == 100
    assert task.status == TaskStatus.PENDING


def test_task_lifecycle():
    """Test the complete task lifecycle."""
    board = TaskBoard()
    board.clear_all()
    
    # Add task
    board.add_task("test_2", "Lifecycle Test", total=100)
    
    # Start task
    board.start_task("test_2")
    task = board.get_task("test_2")
    assert task.status == TaskStatus.RUNNING
    assert task.start_time is not None
    
    # Update progress
    board.update_task("test_2", progress=50)
    task = board.get_task("test_2")
    assert task.progress == 50
    
    # Complete task
    board.complete_task("test_2")
    task = board.get_task("test_2")
    assert task.status == TaskStatus.COMPLETED
    assert task.end_time is not None
    assert task.progress == 100


def test_task_failure():
    """Test task failure handling."""
    board = TaskBoard()
    board.clear_all()
    
    board.add_task("test_3", "Failing Task", total=100)
    board.start_task("test_3")
    board.fail_task("test_3", "Something went wrong")
    
    task = board.get_task("test_3")
    assert task.status == TaskStatus.FAILED
    assert task.error == "Something went wrong"
    assert task.end_time is not None


def test_get_summary():
    """Test getting task summary."""
    board = TaskBoard()
    board.clear_all()
    
    # Add multiple tasks
    board.add_task("task_1", "Task 1")
    board.add_task("task_2", "Task 2")
    board.start_task("task_2")
    
    summary = board.get_summary()
    assert summary["total"] == 2
    assert summary["by_status"]["pending"] == 1
    assert summary["by_status"]["running"] == 1
    assert len(summary["tasks"]) == 2


def test_task_metadata():
    """Test task metadata handling."""
    board = TaskBoard()
    board.clear_all()
    
    metadata = {"category": "test", "index": 42}
    board.add_task("test_4", "Metadata Test", metadata=metadata)
    
    task = board.get_task("test_4")
    assert task.metadata["category"] == "test"
    assert task.metadata["index"] == 42
    
    # Update metadata
    board.update_task("test_4", metadata={"extra": "value"})
    task = board.get_task("test_4")
    assert task.metadata["extra"] == "value"
    assert task.metadata["category"] == "test"  # Should be preserved


def test_remove_task():
    """Test removing a task."""
    board = TaskBoard()
    board.clear_all()
    
    board.add_task("test_5", "Remove Test")
    assert board.get_task("test_5") is not None
    
    board.remove_task("test_5")
    assert board.get_task("test_5") is None


def test_subscriber_notification():
    """Test subscriber notifications."""
    board = TaskBoard()
    board.clear_all()
    
    notifications = []
    
    def callback():
        notifications.append(True)
    
    board.subscribe(callback)
    
    # These operations should trigger notifications
    board.add_task("test_6", "Notification Test")
    board.start_task("test_6")
    board.complete_task("test_6")
    
    board.unsubscribe(callback)
    
    # At least 3 notifications (add, start, complete)
    assert len(notifications) >= 3

