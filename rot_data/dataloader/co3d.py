import json
import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from loguru import logger
from PIL import Image

from .data import Data, DataLoader
from .utils import DownloadError, download_file, unzip_file


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
        try:
            data_zip_path = worker_folder / "data.zip"
            if not data_zip_path.exists():
                download_file(link, data_zip_path)
            extract_dir = unzip_file(data_zip_path, worker_folder)
            
            category_dir = extract_dir / category
            if not category_dir.exists():
                logger.warning(f"Category directory {category_dir} not found in {link}")
                return
            
            for id_dir in sorted(category_dir.iterdir()):
                if not id_dir.is_dir():
                    continue
                
                images_dir = id_dir / "images"
                if not images_dir.exists():
                    logger.warning(f"Images directory not found in {id_dir}")
                    continue
                
                frame_files = sorted(
                    images_dir.glob("frame*.jpg"),
                    key=lambda p: p.stem
                )
                
                if not frame_files:
                    logger.warning(f"No frames found in {images_dir}")
                    continue
                
                selected_frames = select_frames(frame_files, self.num_frames)
                
                images = []
                for frame_path in selected_frames:
                    try:
                        img = Image.open(frame_path)
                        images.append(img.copy())
                        img.close()
                    except Exception as e:
                        logger.error(f"Failed to load image {frame_path}: {e}")
                        continue
                
                if images:
                    yield Data(
                        images=images,
                        label=category,
                        _id=id_dir.name,
                    )
        except DownloadError as e:
            logger.error(f"Failed to download {link}: {e}")
            raise e
        # finally:
        #     shutil.rmtree(worker_folder, ignore_errors=True)

    def load(self) -> Iterable[Data]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        link = "https://dl.fbaipublicfiles.com/co3dv2_231130/apple_006.zip"
        category = "apple"
        for data in self._load_one(link, category):
            yield data

if __name__ == "__main__":
    loader = CO3DDataLoader()
    for data in loader.load():
        print(data)
