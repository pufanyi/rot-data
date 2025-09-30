from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image


@dataclass
class Data:
    images: list[Image.Image]
    label: str
    _id: str


class DataLoader(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> Iterable[Data]:
        raise NotImplementedError
