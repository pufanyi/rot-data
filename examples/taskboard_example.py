"""Example script demonstrating the task board functionality."""

import time
from pathlib import Path

from loguru import logger

from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.taskboard.server import start_server
from rot_data.utils.logging import setup_logger


def main():
    """Run the CO3D data loader with task board enabled."""
    # Setup logging
    setup_logger(level="INFO", use_rich=True)
    
    # Start the task board server
    logger.info("Starting task board server at http://0.0.0.0:8765")
    server = start_server(host="0.0.0.0", port=8765)
    
    logger.info("Task board is now available at http://localhost:8765")
    logger.info("Open the URL in your browser to view task progress")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Create data loader with task board enabled
    cache_dir = Path("cache")
    loader = CO3DDataLoader(
        cache_dir=cache_dir,
        num_threads=4,
        use_taskboard=True,  # Enable task board tracking
    )
    
    logger.info("Starting data loading with task board tracking...")
    
    # Load data - tasks will be tracked in the task board
    count = 0
    for _data in loader.load():
        count += 1
        if count % 10 == 0:
            logger.info(f"Loaded {count} data items so far...")
    
    logger.success(f"Completed loading {count} data items")
    logger.info("Task board server is still running. Press Ctrl+C to exit.")
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop()


if __name__ == "__main__":
    main()

