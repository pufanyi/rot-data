"""Tools to prepare datasets and push them to Hugging Face."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image, Sequence, Value
from loguru import logger

from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.dataloader.data import DataLoader
from rot_data.utils.logging import setup_logger
from rot_data.utils.progress import get_progress_manager

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


def collect_with_progress(iterator: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collect items from an iterator with a live progress display.

    This function expects the ProgressManager to be already active
    (typically started in the calling function or by the loader).
    """
    items = []
    manager = get_progress_manager()

    if manager.is_active:
        # Add a collecting task to the existing progress display
        collect_task = manager.add_task(
            "[yellow]Collecting dataset items",
            total=None,
        )
        try:
            for item in iterator:
                items.append(item)
                manager.update(collect_task, advance=1)
        finally:
            manager.remove_task(collect_task)
    else:
        # Fallback: simple collection without progress display
        for item in iterator:
            items.append(item)

    return items


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
    from_generator: bool = False,
) -> Dataset:
    """
    Build a Hugging Face dataset from the configured loader
    and optionally push it.
    """
    loader = get_loader(dataset_name, cache_dir=str(cache_dir), num_threads=num_threads)

    logger.info(f"Building Hugging Face dataset using '{dataset_name}' loader")

    features = Features(
        {
            "id": Value("string"),
            "label": Value("string"),
            "images": Sequence(feature=Image()),
            "predict_image": Image(),
        }
    )

    manager = get_progress_manager()

    # Start progress manager to ensure beautiful progress display
    # The loader and collection will both use this shared progress instance
    with manager.managed_progress():
        if from_generator:
            dataset = Dataset.from_generator(
                lambda: iter_dataset_items(loader),
                num_proc=num_proc,
                features=features,
            )
        else:
            items = collect_with_progress(iter_dataset_items(loader))
            logger.info(f"Collected {len(items)} items")

            dataset = Dataset.from_generator(
                lambda: items,
                features=features,
            )

    logger.success(f"Dataset prepared with {dataset.num_rows} records")

    if push_to_hub:
        target_repo = repo_id or f"pufanyi/{dataset_name}"
        logger.info(f"Pushing dataset to Hugging Face hub repo '{target_repo}'")
        dataset.push_to_hub(target_repo)
        logger.success(f"Dataset pushed to '{target_repo}'")

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
        "--from-generator",
        action="store_true",
        help="Use generator for dataset processing",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level=args.log_level)

    prepare_dataset(
        dataset_name=args.dataset_name,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        from_generator=args.from_generator,
        num_threads=args.num_threads,
        push_to_hub=not args.no_push,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
