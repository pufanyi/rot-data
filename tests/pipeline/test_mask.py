from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rot_data.models.yolo import BoundingBox
from rot_data.pipeline import mask


class _DummyDataset:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._records[index]

    @property
    def num_rows(self) -> int:
        return len(self._records)


class _DummyDetector:
    def __init__(self, *_, **__) -> None:
        self.calls: list[tuple[object, object]] = []

    def detect_bbox(self, image: object, label: object) -> BoundingBox:
        self.calls.append((image, label))
        return BoundingBox(
            x_min=1,
            y_min=2,
            x_max=3,
            y_max=4,
            score=0.99,
            label=str(label),
            class_id=0,
        )


def test_bbox_to_payload_handles_none() -> None:
    payload = mask.bbox_to_payload(None)

    assert payload == {
        "bbox_x_min": None,
        "bbox_y_min": None,
        "bbox_x_max": None,
        "bbox_y_max": None,
        "bbox_score": None,
        "bbox_label": None,
        "bbox_class_id": None,
    }


def test_annotate_dataset_builds_output(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "id": "item-1",
            "label": "chair",
            "images": ["image-a", "image-b"],
            "predict_image": "image-pred",
        }
    ]

    dataset = _DummyDataset(records)

    monkeypatch.setattr(
        mask.datasets,
        "load_dataset",
        lambda repo, split: dataset,
    )
    monkeypatch.setattr(mask, "YOLOBoundingBoxDetector", _DummyDetector)

    annotated_items: list[dict[str, object]] = []

    def fake_from_list(items):
        annotated_items.extend(items)
        assert len(items) == 1
        return _DummyDataset(items)

    monkeypatch.setattr(
        mask.datasets.Dataset,
        "from_list",
        classmethod(lambda cls, items: fake_from_list(items)),
    )

    args = SimpleNamespace(
        dataset_repo="source/repo",
        output_repo="target/repo",
        model_path="weights.pt",
        device=None,
        confidence=0.25,
        iou=0.7,
        skip_push=True,
    )

    result = mask.annotate_dataset(args)

    assert isinstance(result, _DummyDataset)
    assert annotated_items[0]["bbox_x_min"] == 1
    assert annotated_items[0]["bbox_score"] == 0.99
    assert annotated_items[0]["bbox_label"] == "chair"
