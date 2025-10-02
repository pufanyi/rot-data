import io
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from random import randint

import numpy as np
from loguru import logger
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..utils.download import DownloadError, download_file
from ..utils.extract import ZipReader
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
        worker_folder = self.cache_dir / category / link.split("/")[-1].split(".")[0]
        worker_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing category '{category}' from {link}")

        try:
            data_zip_path = worker_folder / "data.zip"
            download_file(link, data_zip_path)

            # Read ZIP archive directly without extraction
            logger.info(f"Reading archive {data_zip_path.name} without extraction")
            frame_pattern = re.compile(r"^(.+)/([^/]+)/images/(frame\d+\.jpg)$")
            data_groups: dict[str, dict[str, list[tuple[str, bytes]]]] = {}

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                # First pass: count total entries
                logger.debug("Counting archive entries...")
                with ZipReader(data_zip_path) as reader:
                    all_entries = reader.list_entries()
                    total_entries = sum(1 for entry in all_entries if entry.is_file)

                logger.debug(f"Found {total_entries} files in archive")

                # Second pass: read and group data
                task = progress.add_task(
                    f"[cyan]Reading {link} data from archive",
                    total=total_entries,
                )

                files_matched = 0
                with ZipReader(data_zip_path) as reader:
                    for entry in all_entries:
                        if not entry.is_file:
                            continue

                        progress.update(task, advance=1)

                        # Match pattern: category/id/images/frame*.jpg
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

            logger.info(f"Matched {files_matched} frame files for {link}")

            # Process grouped data
            if category not in data_groups:
                logger.warning(f"Category {category} not found in {link}")
                return

            num_objects = len(data_groups[category])
            logger.info(f"Processing {num_objects} objects in category '{category}'")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[green]Loading {category} objects",
                    total=num_objects,
                )

                for entry_id in sorted(data_groups[category].keys()):
                    frame_list = data_groups[category][entry_id]

                    if not frame_list:
                        logger.warning(f"No frames found for ID {entry_id}")
                        progress.update(task, advance=1)
                        continue

                    # Sort frames by name
                    frame_list.sort(key=lambda x: x[0])
                    logger.debug(f"Object {entry_id}: found {len(frame_list)} frames")

                    # Select frames using the select_frames function
                    selected_frames, predict_frame = select_frames(frame_list)

                    # Now read only the selected frames
                    images = []
                    try:
                        with ZipReader(data_zip_path) as reader:
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
                        progress.update(task, advance=1)

            logger.success(
                f"Completed loading category '{category}': "
                f"{num_objects} objects processed"
            )

        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e

    def load(self) -> Iterable[Data]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        link = "https://dl.fbaipublicfiles.com/co3dv2_231130/apple_006.zip"
        category = "apple"
        yield from self._load_one(link, category)


if __name__ == "__main__":
    loader = CO3DDataLoader()
    for data in loader.load():
        print(data)
