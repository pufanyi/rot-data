import io
import json
import os
import re
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from random import randint

import numpy as np
from loguru import logger
from PIL import Image

from ..utils.download import DownloadError, download_file
from ..utils.extract import ZipReader
from ..utils.logging import setup_logger
from ..utils.progress import get_progress_manager
from .data import Data, DataLoader


def select_frames[T](frame_list: list[T]) -> tuple[list[T], T]:
    """
    Select evenly spaced frames from a list.

    Args:
        frame_list: List of frames (can be paths, tuples, or any type)

    Returns:
        List of selected frames with the same type as input
    """
    len_frames = len(frame_list)
    num_frame_1_index = randint(0, len_frames - 1)
    num_frame_2_index = num_frame_1_index + len_frames // 2

    # Generate discrete probability distribution using normal distribution
    mu = (num_frame_1_index + num_frame_2_index) / 2
    sigma = (num_frame_2_index - num_frame_1_index) / 6

    # Create array of all possible indices in range
    indices = np.arange(num_frame_1_index, num_frame_2_index + 1)

    # Calculate normal distribution probability density for each index
    probabilities = np.exp(-0.5 * ((indices - mu) / sigma) ** 2)

    # Normalize to sum to 1
    probabilities = probabilities / probabilities.sum()

    # Discrete sampling from the normalized distribution
    result_frame_index = np.random.choice(indices, p=probabilities)

    frame_1 = frame_list[num_frame_1_index]
    frame_2 = frame_list[num_frame_2_index % len_frames]
    predict_frame = frame_list[result_frame_index % len_frames]

    return [frame_1, frame_2], predict_frame


class CO3DDataLoader(DataLoader):
    def __init__(
        self,
        cache_dir: str | os.PathLike = "cache",
        num_threads: int = 4,
    ):
        self.links_file = Path(__file__).parents[1] / "links" / "co3d.json"
        with self.links_file.open() as f:
            self.links = json.load(f)

        self.cache_dir = Path(cache_dir).expanduser()
        self.num_threads = num_threads

    def _load_one(self, link: str, category: str) -> Iterable[Data]:
        task_name = link.split("/")[-1].split(".")[0]
        worker_folder = self.cache_dir / task_name
        worker_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing category '{category}' from {link}")

        try:
            data_zip_path = worker_folder / "data.zip"
            # Use category and task name for clearer progress display
            download_file(link, data_zip_path, description=f"{category} ({task_name})")

            # Read ZIP archive directly without extraction
            logger.info(f"Reading archive {task_name}")
            # Match common image formats: jpg, jpeg, png, bmp, gif, webp, tiff, etc.
            frame_pattern = re.compile(r"^(.+)/([^/]+)/images/(frame\d+\.\w+)$")
            data_groups: dict[str, dict[str, list[tuple[str, bytes]]]] = {}

            manager = get_progress_manager()

            # First pass: count total entries
            logger.debug("Counting archive entries...")
            with ZipReader(data_zip_path) as reader:
                all_entries = reader.list_entries()
                total_entries = sum(1 for entry in all_entries if entry.is_file)

            logger.debug(f"Found {total_entries} files in archive")

            # Second pass: read and group data
            task = None
            if manager.is_active:
                task = manager.add_task(
                    f"[cyan]Reading {task_name} data",
                    total=total_entries,
                )

            files_matched = 0
            with ZipReader(data_zip_path) as reader:
                for entry in all_entries:
                    if not entry.is_file:
                        continue

                    if task is not None and manager.is_active:
                        manager.update(task, advance=1)

                    # Match pattern: category/id/images/frame*.<ext>
                    match = frame_pattern.match(entry.pathname)
                    if not match:
                        continue

                    entry_category, entry_id, frame_name = match.groups()
                    if entry_category != category:
                        continue

                    files_matched += 1
                    # Store entry reference instead of reading content now
                    # Group by category and id
                    if entry_category not in data_groups:
                        data_groups[entry_category] = {}
                    if entry_id not in data_groups[entry_category]:
                        data_groups[entry_category][entry_id] = []

                    data_groups[entry_category][entry_id].append(
                        (frame_name, entry.pathname)
                    )

            if task is not None and manager.is_active:
                manager.remove_task(task)
            logger.info(f"Matched {files_matched} frame files for {link}")

            # Process grouped data
            if category not in data_groups:
                logger.warning(f"Category {category} not found in {link}")
                return

            num_objects = len(data_groups[category])
            logger.info(f"Processing {num_objects} objects in category '{category}'")

            task = None
            if manager.is_active:
                task = manager.add_task(
                    f"[green]Loading {category} objects",
                    total=num_objects,
                )

            # Open ZIP file once for all objects
            with ZipReader(data_zip_path) as reader:
                for entry_id in sorted(data_groups[category].keys()):
                    frame_list = data_groups[category][entry_id]

                    if not frame_list:
                        logger.warning(f"No frames found for ID {entry_id}")
                        if task is not None and manager.is_active:
                            manager.update(task, advance=1)
                        continue

                    # Sort frames by name
                    frame_list.sort(key=lambda x: x[0])
                    logger.debug(f"Object {entry_id}: found {len(frame_list)} frames")

                    # Select frames using the select_frames function
                    selected_frames, predict_frame = select_frames(frame_list)

                    # Now read only the selected frames
                    images = []
                    try:
                        for _frame_name, pathname in selected_frames:
                            content = reader.read_file(pathname)
                            img = Image.open(io.BytesIO(content))
                            images.append(img.copy())
                            img.close()
                        _frame_name, pathname = predict_frame
                        content = reader.read_file(pathname)
                        img = Image.open(io.BytesIO(content))
                        predict_image = img.copy()
                        img.close()
                        yield Data(
                            images=images,
                            predict_image=predict_image,
                            label=category,
                            _id=entry_id,
                        )
                    except Exception as e:
                        logger.warning(f"No valid images loaded for ID {entry_id}: {e}")

                    finally:
                        if task is not None and manager.is_active:
                            manager.update(task, advance=1)

            if task is not None and manager.is_active:
                manager.remove_task(task)

            logger.success(
                f"Completed loading category '{category}': "
                f"{num_objects} objects processed"
            )

        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e

        finally:
            # Clean up the worker folder and all its contents
            if worker_folder.exists():
                shutil.rmtree(worker_folder)
                logger.debug(f"Cleaned up temporary folder: {worker_folder}")

    def load(self) -> Iterable[Data]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[str, str]] = []
        for categories in self.links.values():
            if not isinstance(categories, dict):
                continue
            for category, links in categories.items():
                if category.upper() == "METADATA":
                    continue
                for link in links:
                    jobs.append((category, link))

        if not jobs:
            logger.warning("No CO3D links available to load")
            return

        result_queue: Queue[Data | object] = Queue()
        sentinel = object()
        manager = get_progress_manager()

        def worker(job_category: str, job_link: str) -> None:
            print(f"job_category: {job_category}, job_link: {job_link}")
            try:
                for data in self._load_one(job_link, job_category):
                    result_queue.put(data)
            except DownloadError as exc:
                logger.error(f"Failed to load {job_link}: {exc}")
            except Exception:
                logger.exception(
                    f"Unexpected error while loading {job_category} from {job_link}"
                )
            finally:
                result_queue.put(sentinel)

        with manager.managed_progress(total_tasks=len(jobs)):
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                for job_category, job_link in jobs:
                    executor.submit(worker, job_category, job_link)

                completed_jobs = 0
                while completed_jobs < len(jobs):
                    item = result_queue.get()
                    if item is sentinel:
                        completed_jobs += 1
                        manager.advance_overall()
                        continue
                    yield item


if __name__ == "__main__":
    # Initialize logger with Rich handler to avoid conflicts with progress bars
    setup_logger(level="INFO", use_rich=True)

    loader = CO3DDataLoader()
    for data in loader.load():
        print(data)
