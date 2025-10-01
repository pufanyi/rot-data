import json
import os
from collections.abc import AsyncIterable
import uuid
from loguru import logger
from anyio import Path

from .data import Data, DataLoader
from .utils import DownloadError, download_file


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
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads

    async def _load_one(self, link: str, category: str) -> AsyncIterable[Data]:
        worker_folder = self.cache_dir / category / uuid.uuid4().hex
        worker_folder.mkdir(parents=True, exist_ok=True)
        try:
            await download_file(link, worker_folder / "data.zip")
        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e
        finally:
            worker_folder.rmdir()


    async def load(self) -> AsyncIterable[Data]:
        pass
