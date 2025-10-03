from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rot_data.models import yolo


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeBoxes:
    def __init__(
        self, xyxy: list[list[float]], conf: list[float], cls: list[int]
    ) -> None:
        self.xyxy = _FakeTensor(np.array(xyxy, dtype=float))
        self.conf = _FakeTensor(np.array(conf, dtype=float))
        self.cls = _FakeTensor(np.array(cls, dtype=float))
        self.data = True
        self._length = len(conf)

    def __len__(self) -> int:
        return self._length


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeYOLO:
    names = {0: "chair", 1: "table"}
    next_boxes: _FakeBoxes | None = None

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, *_, **__) -> list[_FakeResult]:
        assert self.next_boxes is not None
        return [_FakeResult(self.next_boxes)]


@pytest.fixture(autouse=True)
def patch_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yolo, "_UltralyticsYOLO", _FakeYOLO, raising=False)


def test_detect_bbox_prefers_label_match() -> None:
    _FakeYOLO.next_boxes = _FakeBoxes(
        xyxy=[[1, 2, 11, 12], [3, 4, 13, 14]],
        conf=[0.4, 0.9],
        cls=[1, 0],
    )
    detector = yolo.YOLOBoundingBoxDetector("weights.pt")

    bbox = detector.detect_bbox(np.zeros((16, 16, 3), dtype=np.uint8), label="chair")

    assert bbox is not None
    assert bbox.label == "chair"
    assert bbox.score == pytest.approx(0.9)
    assert bbox.x_min == 3
    assert bbox.y_min == 4
    assert bbox.x_max == 13
    assert bbox.y_max == 14


def test_detect_bbox_returns_best_when_label_missing() -> None:
    _FakeYOLO.next_boxes = _FakeBoxes(
        xyxy=[[5, 6, 25, 26], [7, 8, 27, 28]],
        conf=[0.2, 0.8],
        cls=[1, 1],
    )
    detector = yolo.YOLOBoundingBoxDetector("weights.pt")

    bbox = detector.detect_bbox(np.zeros((16, 16, 3), dtype=np.uint8), label="chair")

    assert bbox is not None
    assert bbox.label == "table"
    assert bbox.score == pytest.approx(0.8)
    assert bbox.x_min == 7
    assert bbox.y_min == 8


def test_to_numpy_converts_grayscale() -> None:
    grayscale = np.arange(9, dtype=np.uint8).reshape(3, 3)

    rgb = yolo.YOLOBoundingBoxDetector._to_numpy(grayscale)

    assert rgb.shape == (3, 3, 3)
    # Each channel should equal the original grayscale values
    assert np.all(rgb[..., 0] == grayscale)
    assert np.all(rgb[..., 1] == grayscale)
    assert np.all(rgb[..., 2] == grayscale)
