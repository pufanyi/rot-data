"""Logging utilities using loguru."""

import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    use_rich: bool = True,
) -> None:
    """
    Configure loguru logger with custom settings.

    Args:
        level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only logs to console
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 week", "30 days")
        use_rich: Whether to use Rich handler (compatible with Rich progress bars)
    """
    # Remove default handler
    logger.remove()

    if use_rich:
        # Use Rich handler for better compatibility with Rich progress bars
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_level=True,
            show_path=True,
            omit_repeated_times=False,
        )
        logger.add(
            rich_handler,
            level=level,
            format="{message}",
        )
    else:
        # Add console handler with custom format
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Add file handler if log_file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        logger.info(f"Logging to file: {log_file}")


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        The loguru logger instance
    """
    return logger
