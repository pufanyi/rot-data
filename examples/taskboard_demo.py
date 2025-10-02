"""Simple demo of the task board without requiring full data loading."""

import time
from random import randint, random
from threading import Thread

from loguru import logger

from rot_data.taskboard import TaskBoard
from rot_data.taskboard.server import start_server
from rot_data.utils.logging import setup_logger


def simulate_task(taskboard: TaskBoard, task_id: str, task_name: str) -> None:
    """Simulate a task with random progress."""
    # Add task
    total = randint(50, 200)
    taskboard.add_task(task_id, task_name, total=total)
    
    # Start task
    time.sleep(random() * 2)
    taskboard.start_task(task_id)
    logger.info(f"Started task: {task_name}")
    
    # Simulate progress
    for i in range(total):
        time.sleep(random() * 0.1)
        taskboard.update_task(task_id, progress=i + 1)
    
    # Randomly fail or complete
    if random() < 0.1:  # 10% chance of failure
        taskboard.fail_task(task_id, "Random failure for demo purposes")
        logger.error(f"Failed task: {task_name}")
    else:
        taskboard.complete_task(task_id)
        logger.success(f"Completed task: {task_name}")


def main():
    """Run a demo with simulated tasks."""
    setup_logger(level="INFO", use_rich=True)
    
    # Start task board server
    logger.info("Starting task board server at http://0.0.0.0:8765")
    server = start_server(host="0.0.0.0", port=8765)
    
    logger.success("Task board is now available at http://localhost:8765")
    logger.info("Open the URL in your browser to view task progress")
    
    # Wait for server to start
    time.sleep(2)
    
    # Get task board instance
    taskboard = TaskBoard()
    
    # Create 441 simulated tasks (like the user's _load_one tasks)
    num_tasks = 441
    logger.info(f"Creating {num_tasks} simulated tasks...")
    
    threads = []
    for i in range(num_tasks):
        task_id = f"load_task_{i}"
        task_name = f"Load CO3D Subset {i + 1}/{num_tasks}"
        thread = Thread(target=simulate_task, args=(taskboard, task_id, task_name))
        thread.daemon = True
        threads.append(thread)
    
    # Start all threads with staggered start
    logger.info("Starting all tasks...")
    for thread in threads:
        thread.start()
        time.sleep(0.01)  # Stagger starts slightly
    
    # Wait for all tasks to complete
    logger.info("Tasks are running. Check the task board in your browser!")
    for thread in threads:
        thread.join()
    
    logger.success("All tasks completed!")
    logger.info("Task board server is still running. Press Ctrl+C to exit.")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop()


if __name__ == "__main__":
    main()

