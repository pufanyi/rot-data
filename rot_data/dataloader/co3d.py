import json
import os
from collections.abc import AsyncIterable
from pathlib import Path

from .data import Data, DataLoader


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

    async def _load_one(self, link: str) -> AsyncIterable[Data]:
        pass

    async def load(self) -> AsyncIterable[Data]:
        pass
