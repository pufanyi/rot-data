import json
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
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
        self.cache_dir = Path(cache_dir).expanduser()
        self.num_threads = num_threads

    def _load_one(self, link: str, category: str) -> Iterable[Data]:
        worker_folder = self.cache_dir / category / uuid.uuid4().hex
        worker_folder.mkdir(parents=True, exist_ok=True)
        try:
            data_zip_path = worker_folder / "data.zip"
            download_file(link, data_zip_path)
            unzip_file(data_zip_path, worker_folder)
        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e
        finally:
            shutil.rmtree(worker_folder, ignore_errors=True)

    def load(self) -> Iterable[Data]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pass
