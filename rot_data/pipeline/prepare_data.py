"""Tools to prepare datasets and push them to Hugging Face."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

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


def save_dataset_to_jsonl(
    dataset_name: str = "co3d",
    *,
    output_dir: str | Path = "output",
    cache_dir: str | Path = "cache",
    num_threads: int = 32,
) -> None:
    """
    Build a dataset from the configured loader and save to jsonl format.
    Images are saved to a separate directory and referenced by relative paths.
    """
    loader = get_loader(dataset_name, cache_dir=str(cache_dir), num_threads=num_threads)

    logger.info(f"Building dataset using '{dataset_name}' loader")

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path / "data.jsonl"

    manager = get_progress_manager()

    # Start progress manager to ensure beautiful progress display
    # The loader and collection will both use this shared progress instance
    with manager.managed_progress():
        items = collect_with_progress(iter_dataset_items(loader))
        logger.info(f"Collected {len(items)} items")

        # Save images and create jsonl entries
        logger.info(f"Saving images to {images_dir} and metadata to {jsonl_path}")
        
        save_task = manager.add_task(
            "[green]Saving dataset",
            total=len(items),
        )
        
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in items:
                # Save images
                image_paths = []
                for idx, img in enumerate(item["images"]):
                    img_filename = f"{item['id']}_input_{idx}.jpg"
                    img_path = images_dir / img_filename
                    img.save(img_path)
                    # Store relative path from output_dir
                    image_paths.append(f"images/{img_filename}")
                
                # Save predict image
                predict_img_filename = f"{item['id']}_predict.jpg"
                predict_img_path = images_dir / predict_img_filename
                item["predict_image"].save(predict_img_path)
                predict_img_rel_path = f"images/{predict_img_filename}"
                
                # Create jsonl entry
                jsonl_entry = {
                    "id": item["id"],
                    "label": item["label"],
                    "images": image_paths,
                    "predict_image": predict_img_rel_path,
                }
                
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                manager.update(save_task, advance=1)
        
        manager.remove_task(save_task)

    logger.success(f"Dataset saved with {len(items)} records to {output_path}")
    logger.success(f"  - Metadata: {jsonl_path}")
    logger.success(f"  - Images: {images_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets and save to jsonl format"
    )
    parser.add_argument(
        "--dataset-name",
        default="co3d",
        choices=sorted(DATASET_LOADERS),
        help="Data source to load",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for jsonl and images",
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
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level=args.log_level)

    save_dataset_to_jsonl(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
