from abc import ABC, abstractmethod
from collections.abc import Iterable

from PIL import Image


class Data:
    def __init__(self, _id: str, images: list[Image.Image], label: str):
        self.images = images
        self.label = label
        self._id = _id


class DataLoader(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> Iterable[Data]:
        raise NotImplementedError
