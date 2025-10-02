"""Tools to prepare datasets and push them to Hugging Face."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image, Sequence, Value
from loguru import logger

from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.dataloader.data import DataLoader
from rot_data.taskboard.server import TaskBoardServer, start_server
from rot_data.utils.logging import setup_logger

LoaderFactory = Callable[..., DataLoader]

DATASET_LOADERS: dict[str, LoaderFactory] = {
    "co3d": CO3DDataLoader,
}


def iter_dataset_items(loader: DataLoader) -> Iterator[dict[str, Any]]:
    """Transform a ``DataLoader`` stream into dictionaries usable by ``Dataset``."""
    for item in loader.load():
        yield {
            "id": item._id,
            "label": item.label,
            "images": item.images,
            "predict_image": item.predict_image,
        }


def get_loader(dataset_name: str, **loader_kwargs: Any) -> DataLoader:
    """Instantiate a loader for the requested dataset."""
    try:
        loader_factory = DATASET_LOADERS[dataset_name]
    except KeyError as exc:  # pragma: no cover - invalid configuration guard
        available = ", ".join(sorted(DATASET_LOADERS)) or "<none>"
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available options: {available}."
        ) from exc

    try:
        return loader_factory(**loader_kwargs)
    except TypeError as exc:  # pragma: no cover - propagate clearer error
        raise TypeError(
            "Loader initialization failed. Check provided keyword arguments."
        ) from exc


def prepare_dataset(
    dataset_name: str = "co3d",
    *,
    repo_id: str | None = None,
    cache_dir: str | Path = "cache",
    num_threads: int = 32,
    push_to_hub: bool = True,
    num_proc: int = 32,
    use_taskboard: bool = False,
    taskboard_port: int = 8765,
) -> Dataset:
    """
    Build a Hugging Face dataset from the configured loader
    and optionally push it.
    
    Args:
        dataset_name: Name of the dataset to load
        repo_id: Target Hugging Face repo ID
        cache_dir: Cache directory for intermediate artifacts
        num_threads: Number of threads for data loading
        push_to_hub: Whether to push to Hugging Face hub
        num_proc: Number of processes for dataset processing
        use_taskboard: Whether to enable task board visualization
        taskboard_port: Port for the task board server
    
    Returns:
        The prepared Dataset
    """
    server: TaskBoardServer | None = None
    
    # Start task board server if requested
    if use_taskboard:
        logger.info(f"Starting task board server at http://0.0.0.0:{taskboard_port}")
        server = start_server(host="0.0.0.0", port=taskboard_port)
        logger.success(f"Task board available at http://localhost:{taskboard_port}")
        logger.info("Open the URL in your browser to view task progress")
        time.sleep(2)  # Give server time to start
        
        # Force single process when taskboard is enabled to avoid serialization issues
        if num_proc != 1:
            logger.warning(
                f"Taskboard is enabled. Forcing num_proc=1 (was {num_proc}) "
                "to avoid serialization issues with WebSocket connections."
            )
            num_proc = 1
    
    # Create loader with task board enabled if requested
    loader = get_loader(
        dataset_name,
        cache_dir=str(cache_dir),
        num_threads=num_threads,
        use_taskboard=use_taskboard,
    )

    logger.info(f"Building Hugging Face dataset using '{dataset_name}' loader")

    features = Features(
        {
            "id": Value("string"),
            "label": Value("string"),
            "images": Sequence(feature=Image()),
            "predict_image": Image(),
        }
    )

    # When taskboard is enabled, collect all data first to avoid serialization issues
    if use_taskboard:
        logger.info("Collecting all data items (taskboard mode)...")
        data_items = list(iter_dataset_items(loader))
        logger.info(f"Collected {len(data_items)} items, creating dataset...")
        dataset = Dataset.from_list(data_items, features=features)
    else:
        dataset = Dataset.from_generator(
            lambda: iter_dataset_items(loader),
            num_proc=num_proc,
            features=features,
        )

    logger.success(f"Dataset prepared with {dataset.num_rows} records")

    if push_to_hub:
        target_repo = repo_id or f"pufanyi/{dataset_name}"
        logger.info(f"Pushing dataset to Hugging Face hub repo '{target_repo}'")
        dataset.push_to_hub(target_repo)
        logger.success(f"Dataset pushed to '{target_repo}'")

    # Cleanup task board server
    if server:
        logger.info("Task board server will remain active until you stop the program")
    
    return dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets and push to Hugging Face"
    )
    parser.add_argument(
        "--dataset-name",
        default="co3d",
        choices=sorted(DATASET_LOADERS),
        help="Data source to load",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Hugging Face repo (defaults to 'pufanyi/<dataset-name>')",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Cache directory for intermediate artifacts",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of threads for data loading",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=32,
        help="Number of processes for dataset processing",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing the prepared dataset to the Hugging Face hub",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    parser.add_argument(
        "--use-taskboard",
        action="store_true",
        help="Enable task board visualization (opens web interface)",
    )
    parser.add_argument(
        "--taskboard-port",
        type=int,
        default=8765,
        help="Port for the task board server (default: 8765)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level=args.log_level)

    if args.use_taskboard:
        logger.info("=" * 60)
        logger.info("Task Board Enabled!")
        logger.info(f"Open http://localhost:{args.taskboard_port} in your browser")
        logger.info("=" * 60)

    try:
        prepare_dataset(
            dataset_name=args.dataset_name,
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            num_threads=args.num_threads,
            push_to_hub=not args.no_push,
            num_proc=args.num_proc,
            use_taskboard=args.use_taskboard,
            taskboard_port=args.taskboard_port,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if args.use_taskboard:
            logger.info("Task board server stopped")


if __name__ == "__main__":
    main()
