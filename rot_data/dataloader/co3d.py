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
from typing import Any

import numpy as np
from datasets import (
    Dataset,
    Features,
    Value,
    load_from_disk,
)
from datasets import (
    Image as HFImage,
)
from datasets import (
    Sequence as HFSequence,
)
from loguru import logger
from PIL import Image

from ..utils.download import DownloadError, download_file
from ..utils.extract import ZipReader
from ..utils.logging import setup_logger
from ..utils.progress import get_progress_manager
from .data import Data, DataLoader

DATASET_REPO_ID = "pufanyi/co3d-subsets"
SUBSET_FEATURES = Features(
    {
        "id": Value("string"),
        "label": Value("string"),
        "images": HFSequence(feature=HFImage()),
        "predict_image": HFImage(),
    }
)

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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_cache_root = (
            self.cache_dir / "datasets" / DATASET_REPO_ID.replace("/", "__")
        )
        self.dataset_cache_root.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads

    def _subset_name(self, link: str) -> str:
        return Path(link).stem

    def _subset_cache_dir(self, subset_name: str) -> Path:
        return self.dataset_cache_root / subset_name

    def _load_subset_from_cache(self, subset_name: str) -> Dataset | None:
        subset_dir = self._subset_cache_dir(subset_name)
        if not subset_dir.exists():
            return None
        try:
            return load_from_disk(str(subset_dir))
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            logger.warning(
                f"Failed to load cached subset '{subset_name}': {exc}. Rebuilding.",
            )
            shutil.rmtree(subset_dir, ignore_errors=True)
            return None

    @staticmethod
    def _ensure_pil_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy()
        if isinstance(image, dict):
            path = image.get("path")
            if path:
                with Image.open(path) as pil_image:
                    return pil_image.copy()
            image_bytes = image.get("bytes")
            if image_bytes is not None:
                with Image.open(io.BytesIO(image_bytes)) as pil_image:
                    return pil_image.copy()
        if isinstance(image, (bytes, bytearray)):
            with Image.open(io.BytesIO(image)) as pil_image:
                return pil_image.copy()
        raise TypeError(f"Unsupported image payload type: {type(image)!r}")

    def _load_one(self, link: str, category: str) -> Iterable[Data]:
        subset_name = self._subset_name(link)
        dataset = self._load_subset_from_cache(subset_name)
        if dataset is None:
            dataset = self._build_subset_dataset(link, category, subset_name)
        else:
            logger.info(
                f"Using cached subset '{subset_name}' for category '{category}'",
            )

        if dataset is None:
            return

        for record in dataset:
            try:
                images = [self._ensure_pil_image(image) for image in record["images"]]
                predict_image = self._ensure_pil_image(record["predict_image"])
                yield Data(
                    images=images,
                    predict_image=predict_image,
                    label=record["label"],
                    _id=record["id"],
                )
            except Exception as exc:
                logger.warning(
                    f"Skipping corrupted record '{record.get('id', '<unknown>')}' in subset '{subset_name}': {exc}",
                )

    def _build_subset_dataset(
        self, link: str, category: str, subset_name: str
    ) -> Dataset | None:
        worker_folder = self.cache_dir / subset_name
        worker_folder.mkdir(parents=True, exist_ok=True)
        subset_cache_dir = self._subset_cache_dir(subset_name)

        logger.info(
            f"Processing category '{category}' from {link} to build subset '{subset_name}'",
        )

        manager = get_progress_manager()
        records: list[dict[str, Any]] = []

        try:
            data_zip_path = worker_folder / "data.zip"
            download_file(
                link,
                data_zip_path,
                description=f"{category} ({subset_name})",
            )

            logger.info(f"Reading archive {subset_name}")
            frame_pattern = re.compile(r"^(.+)/([^/]+)/images/(frame\d+\.\w+)$")
            data_groups: dict[str, dict[str, list[tuple[str, str]]]] = {}

            logger.debug("Counting archive entries...")
            with ZipReader(data_zip_path) as reader:
                all_entries = reader.list_entries()
                total_entries = sum(1 for entry in all_entries if entry.is_file)

            logger.debug(f"Found {total_entries} files in archive")

            read_task = None
            if manager.is_active:
                read_task = manager.add_task(
                    f"[cyan]Reading {subset_name} data",
                    category,
                    link,
                    subset_name,
                    total=total_entries,
                )

            files_matched = 0
            with ZipReader(data_zip_path) as reader:
                for entry in all_entries:
                    if not entry.is_file:
                        continue

                    if read_task is not None and manager.is_active:
                        manager.update(read_task, advance=1)

                    match = frame_pattern.match(entry.pathname)
                    if not match:
                        continue

                    entry_category, entry_id, frame_name = match.groups()
                    if entry_category != category:
                        continue

                    files_matched += 1
                    entry_groups = data_groups.setdefault(entry_category, {})
                    entry_groups.setdefault(entry_id, []).append(
                        (frame_name, entry.pathname)
                    )

            if read_task is not None and manager.is_active:
                manager.remove_task(read_task)
            logger.info(f"Matched {files_matched} frame files for {link}")

            object_ids = sorted(data_groups.get(category, {}))
            if not object_ids:
                logger.warning(f"Category {category} not found in {link}")
            else:
                load_task = None
                if manager.is_active:
                    load_task = manager.add_task(
                        f"[green]Loading {category} objects",
                        category,
                        link,
                        subset_name,
                        total=len(object_ids),
                    )

                with ZipReader(data_zip_path) as reader:
                    for entry_id in object_ids:
                        frame_list = data_groups[category][entry_id]

                        if not frame_list:
                            logger.warning(f"No frames found for ID {entry_id}")
                            if load_task is not None and manager.is_active:
                                manager.update(load_task, advance=1)
                            continue

                        frame_list.sort(key=lambda item: item[0])
                        logger.debug(
                            f"Object {entry_id}: found {len(frame_list)} frames",
                            entry_id,
                            len(frame_list),
                        )

                        selected_frames, predict_frame = select_frames(frame_list)

                        try:
                            images = []
                            for _frame_name, pathname in selected_frames:
                                content = reader.read_file(pathname)
                                with Image.open(io.BytesIO(content)) as img:
                                    images.append(img.copy())

                            _frame_name, pathname = predict_frame
                            content = reader.read_file(pathname)
                            with Image.open(io.BytesIO(content)) as img:
                                predict_image = img.copy()

                            records.append(
                                {
                                    "id": entry_id,
                                    "label": category,
                                    "images": images,
                                    "predict_image": predict_image,
                                }
                            )
                        except Exception as exc:
                            logger.warning(
                                f"No valid images loaded for ID {entry_id}: {exc}",
                                entry_id,
                                exc,
                            )
                        finally:
                            if load_task is not None and manager.is_active:
                                manager.update(load_task, advance=1)

                if load_task is not None and manager.is_active:
                    manager.remove_task(load_task)
                logger.success(
                    f"Completed loading category '{category}': {len(records)} objects processed",
                    category,
                    len(records),
                )

            if subset_cache_dir.exists():
                shutil.rmtree(subset_cache_dir)

            if records:
                dataset = Dataset.from_list(records, features=SUBSET_FEATURES)
            else:
                logger.warning(
                    f"No valid records collected for subset '{subset_name}' (category '{category}').",
                    subset_name,
                    category,
                )
                empty_columns = {name: [] for name in SUBSET_FEATURES}  # type: ignore[assignment]
                dataset = Dataset.from_dict(empty_columns, features=SUBSET_FEATURES)

            dataset.save_to_disk(str(subset_cache_dir))

            metadata = {
                "category": category,
                "link": link,
                "repo_id": DATASET_REPO_ID,
                "num_records": len(records),
            }
            (subset_cache_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )
            logger.success(
                "Cached subset '%s' with %d records at %s",
                subset_name,
                len(records),
                subset_cache_dir,
            )
            return dataset
        except DownloadError as exc:
            logger.error("Failed to download %s: %s", link, exc)
            raise
        finally:
            if worker_folder.exists():
                shutil.rmtree(worker_folder)
                logger.debug("Cleaned up temporary folder: %s", worker_folder)


    def _collect_jobs(self) -> list[tuple[str, str]]:
        jobs: list[tuple[str, str]] = []
        for categories in self.links.values():
            if not isinstance(categories, dict):
                continue
            for category, links in categories.items():
                if category.upper() == "METADATA":
                    continue
                for link in links:
                    jobs.append((category, link))
        return jobs

    def iter_subset_datasets(self) -> Iterable[tuple[str, Dataset]]:
        for category, link in self._collect_jobs():
            subset_name = self._subset_name(link)
            dataset = self._load_subset_from_cache(subset_name)
            if dataset is None:
                dataset = self._build_subset_dataset(link, category, subset_name)
            if dataset is None:
                continue
            yield subset_name, dataset

    def push_subsets_to_hub(
        self,
        repo_id: str = DATASET_REPO_ID,
        *,
        private: bool | None = None,
        token: str | None = None,
    ) -> None:
        for subset_name, dataset in self.iter_subset_datasets():
            logger.info("Pushing subset '%s' to '%s'", subset_name, repo_id)
            dataset.push_to_hub(
                repo_id,
                config_name=subset_name,
                private=private,
                token=token,
            )
            logger.success(
                "Subset '%s' pushed to '%s' as config '%s'",
                subset_name,
                repo_id,
                subset_name,
            )

    def load(self) -> Iterable[Data]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        jobs = self._collect_jobs()

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
