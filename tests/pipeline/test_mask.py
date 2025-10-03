from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from rot_data.models.yolo import BoundingBox
from rot_data.pipeline import mask_yolo


class DummyDataset:
    def __init__(self, records):
        self.records = records
        self.added_columns: dict[str, list[dict[str, object]]] = {}
        self.cast_columns: list[tuple[str, object]] = []
        self.push_calls: list[tuple[str, str]] = []

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:  # pragma: no cover - compatibility hook
        return len(self.records)

    def __getitem__(self, index: int):  # pragma: no cover - compatibility hook
        return self.records[index]

    @property
    def num_rows(self) -> int:
        return len(self.records)

    def add_column(self, name: str, values):
        self.added_columns[name] = list(values)
        return self

    def cast_column(self, name: str, feature):
        self.cast_columns.append((name, feature))
        return self

    def push_to_hub(
        self, repo: str, split: str
    ):  # pragma: no cover - hub is mocked out
        self.push_calls.append((repo, split))


class DummyDetector:
    instances: list[DummyDetector] = []

    def __init__(self, *_, **__):
        self.calls: list[tuple[list[Image.Image], list[str | None]]] = []
        DummyDetector.instances.append(self)

    def detect_batch(self, images, labels):
        batch_images = list(images)
        batch_labels = [label if label is None else str(label) for label in labels]
        self.calls.append((batch_images, batch_labels))
        return [
            BoundingBox(
                x_min=0,
                y_min=0,
                x_max=image.width,
                y_max=image.height,
                score=0.99,
                label=label or "object",
                class_id=0,
            )
            for image, label in zip(batch_images, batch_labels, strict=True)
        ]


class DummyImageFeature:
    def __init__(self) -> None:
        self.encoded: list[Image.Image] = []

    def encode_example(self, image: Image.Image) -> dict[str, Image.Image]:
        self.encoded.append(image)
        return {"image": image}


def test_mask_image_with_bbox_blanks_region() -> None:
    image = Image.new("RGB", (4, 4), color=(255, 255, 255))
    bbox = BoundingBox(1, 1, 3, 3, 0.9, "chair", 0)

    masked = mask_yolo.mask_image_with_bbox(image, bbox)

    assert masked.getpixel((0, 0)) == (255, 255, 255)
    for x in range(bbox.x_min, bbox.x_max):
        for y in range(bbox.y_min, bbox.y_max):
            assert masked.getpixel((x, y)) == (0, 0, 0)


def test_mask_dataset_adds_mask_column(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [
        Image.new("RGB", (2, 2), color=(255, 0, 0)),
        Image.new("RGB", (2, 2), color=(0, 255, 0)),
        Image.new("RGB", (2, 2), color=(0, 0, 255)),
    ]
    dataset = DummyDataset(
        [
            {"predict_image": images[0], "label": "chair"},
            {"predict_image": images[1], "label": "table"},
            {"predict_image": images[2], "label": "lamp"},
        ]
    )

    DummyDetector.instances = []
    image_feature = DummyImageFeature()

    monkeypatch.setattr(mask_yolo.datasets, "load_dataset", lambda repo, split: dataset)
    monkeypatch.setattr(mask_yolo, "YOLOBoundingBoxDetector", DummyDetector)
    monkeypatch.setattr(mask_yolo, "DatasetImage", lambda: image_feature)

    args = SimpleNamespace(
        dataset_repo="source/repo",
        output_repo="target/repo",
        model_path="weights.pt",
        device="cpu",
        devices=None,
        confidence=0.25,
        iou=0.7,
        batch_size=2,
        num_workers=1,
        skip_push=True,
    )

    result = mask_yolo.mask_dataset(args)

    assert result is dataset
    assert "masked_image" in dataset.added_columns
    masked_column = dataset.added_columns["masked_image"]
    assert len(masked_column) == dataset.num_rows
    assert all(entry["image"].getpixel((0, 0)) == (0, 0, 0) for entry in masked_column)
    assert image_feature.encoded == [entry["image"] for entry in masked_column]
    assert dataset.push_calls == []

    assert len(DummyDetector.instances) == 1
    calls = DummyDetector.instances[0].calls
    assert len(calls) == 2  # one batch of two images, one batch of one image
    assert [len(batch) for batch, _ in calls] == [2, 1]
    assert calls[0][1] == ["chair", "table"]
    assert calls[1][1] == ["lamp"]


def test_resolve_devices_prefers_autodetected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mask_yolo, "_autodetect_devices", lambda: ["cuda:0", "cuda:1"])
    assert mask_yolo._resolve_devices(None, None) == ["cuda:0", "cuda:1"]


def test_resolve_devices_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mask_yolo, "_autodetect_devices", lambda: [])
    assert mask_yolo._resolve_devices(None, None) == [None]
