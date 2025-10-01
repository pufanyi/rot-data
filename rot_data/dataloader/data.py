from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass

from PIL import Image


@dataclass
class Data:
    images: list[Image.Image]
    label: str
    _id: str


class DataLoader(ABC):
    @abstractmethod
    async def load(self, *args, **kwargs) -> AsyncIterable[Data]:
        raise NotImplementedError
