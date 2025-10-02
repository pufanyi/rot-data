import io
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

import libarchive
from loguru import logger
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .data import Data, DataLoader
from ..utils.download import DownloadError, download_file


def select_frames(frame_paths: list[Path], num_frames: int) -> list[Path]:
    total_frames = len(frame_paths)
    if total_frames == 0:
        return []
    if total_frames <= num_frames:
        return frame_paths

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    return [frame_paths[i] for i in indices]


class CO3DDataLoader(DataLoader):
    def __init__(
        self,
        num_frames: int = 6,
        cache_dir: str | os.PathLike = "cache",
        num_threads: int = 4,
    ):
        self.links_file = Path(__file__).parents[1] / "links" / "co3d.json"
        with self.links_file.open() as f:
            self.links = json.load(f)

        self.num_frames = num_frames
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
            ) as progress:
                # First pass: count total entries
                logger.debug("Counting archive entries...")
                total_entries = 0
                with libarchive.file_reader(str(data_zip_path)) as archive:
                    for entry in archive:
                        if entry.isfile:
                            total_entries += 1
                
                logger.debug(f"Found {total_entries} files in archive")
                
                # Second pass: read and group data
                task = progress.add_task(
                    f"[cyan]Reading {category} data from archive",
                    total=total_entries,
                )
                
                files_matched = 0
                with libarchive.file_reader(str(data_zip_path)) as archive:
                    for entry in archive:
                        if not entry.isfile:
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
                        # Read file content into memory
                        content = b"".join(entry.get_blocks())

                        # Group by category and id
                        if entry_category not in data_groups:
                            data_groups[entry_category] = {}
                        if entry_id not in data_groups[entry_category]:
                            data_groups[entry_category][entry_id] = []

                        data_groups[entry_category][entry_id].append(
                            (frame_name, content)
                        )

            logger.info(
                f"Matched {files_matched} frame files for category '{category}'"
            )

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
                    logger.debug(
                        f"Object {entry_id}: found {len(frame_list)} frames, "
                        f"selecting {self.num_frames}"
                    )

                    # Select frames
                    selected_indices = [
                        int(i * len(frame_list) / self.num_frames)
                        for i in range(self.num_frames)
                    ]
                    selected_frames = [frame_list[i] for i in selected_indices]

                    images = []
                    for frame_name, content in selected_frames:
                        try:
                            img = Image.open(io.BytesIO(content))
                            images.append(img.copy())
                            img.close()
                        except Exception as e:
                            logger.error(
                                f"Failed to load image {frame_name} "
                                f"from {entry_id}: {e}"
                            )
                            continue

                    if images:
                        logger.debug(
                            f"Successfully loaded {len(images)} images for {entry_id}"
                        )
                        yield Data(
                            images=images,
                            label=category,
                            _id=entry_id,
                        )
                    else:
                        logger.warning(f"No valid images loaded for ID {entry_id}")

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
