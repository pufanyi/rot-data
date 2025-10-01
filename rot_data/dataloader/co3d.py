import json
import os
import uuid
from collections.abc import AsyncIterable
from pathlib import Path

from anyio import Path as AnyioPath
from loguru import logger

from .data import Data, DataLoader
from .utils import DownloadError, download_file, unzip_file


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
        self.cache_dir = AnyioPath(cache_dir)
        self.num_threads = num_threads

    async def _load_one(self, link: str, category: str) -> AsyncIterable[Data]:
        worker_folder = self.cache_dir / category / uuid.uuid4().hex
        await worker_folder.mkdir(parents=True, exist_ok=True)
        try:
            data_zip_path = worker_folder / "data.zip"
            await download_file(link, data_zip_path)
            await unzip_file(data_zip_path, worker_folder)
        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e
        finally:
            await worker_folder.rmdir()

    async def load(self) -> AsyncIterable[Data]:
        await self.cache_dir.mkdir(parents=True, exist_ok=True)
        pass
