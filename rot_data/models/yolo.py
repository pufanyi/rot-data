"""Utilities for running YOLO object detection models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency error handling
    from ultralytics import YOLO as _UltralyticsYOLO
except ModuleNotFoundError:  # pragma: no cover - import guard
    _UltralyticsYOLO = None


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """Structured representation of a predicted bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    label: str
    class_id: int | None = None

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serialisable representation of the bounding box."""
        return asdict(self)


class YOLOBoundingBoxDetector:
    """Thin wrapper around an Ultralytics YOLO model for bounding boxes."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        if _UltralyticsYOLO is None:  # pragma: no cover - import guard
            raise ImportError(
                "YOLOBoundingBoxDetector requires the 'ultralytics' package. "
                "Install it with `uv add ultralytics` or add it to your dependencies."
            )
        self.model_path = str(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self._model = _UltralyticsYOLO(self.model_path)

        names = getattr(self._model, "names", None) or getattr(
            getattr(self._model, "model", None), "names", {}
        )
        if isinstance(names, Sequence):
            self.class_names = dict(enumerate(names))
        elif isinstance(names, dict):
            self.class_names = {int(idx): str(name) for idx, name in names.items()}
        else:  # pragma: no cover - defensive fallback for unexpected formats
            self.class_names = {}

    def detect_bbox(
        self,
        image: Image.Image | np.ndarray,
        label: str | None = None,
    ) -> BoundingBox | None:
        """Return the highest-confidence bounding box predicted for *image*.

        If *label* is provided, the detector will prioritise boxes that match
        the label (case-insensitive). If no matching boxes are found, the
        highest-confidence prediction overall is returned. ``None`` is
        returned when the model does not produce any boxes.
        """

        np_image = self._to_numpy(image)

        results = self._model.predict(
            np_image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return None

        prediction = results[0]
        return self._select_bbox(prediction, label)

    def detect_batch(
        self,
        images: Sequence[Image.Image | np.ndarray],
        labels: Sequence[str | None] | None = None,
    ) -> list[BoundingBox | None]:
        """Return the best bounding boxes for a batch of *images*.

        The sequence of *labels* must be the same length as *images* if
        provided. Each entry influences the selection logic in the same way as
        :meth:`detect_bbox`.
        """

        if labels is None:
            labels = [None] * len(images)
        elif len(labels) != len(images):
            raise ValueError("labels must be the same length as images")

        np_images = [self._to_numpy(image) for image in images]

        results = self._model.predict(
            np_images,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if not isinstance(results, list):
            results = list(results)

        if len(results) != len(np_images):
            raise RuntimeError(
                "YOLO model returned a mismatched number of predictions: "
                f"expected {len(np_images)}, got {len(results)}"
            )

        return [
            self._select_bbox(prediction, label)
            for prediction, label in zip(results, labels, strict=True)
        ]

    def _select_bbox(
        self,
        prediction: object,
        label: str | None,
    ) -> BoundingBox | None:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or boxes.data is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        candidates: list[BoundingBox] = []
        for coords, score, class_id in zip(xyxy, scores, class_ids, strict=True):
            x_min, y_min, x_max, y_max = coords
            name = self.class_names.get(class_id, str(class_id))
            bbox = BoundingBox(
                x_min=int(round(float(x_min))),
                y_min=int(round(float(y_min))),
                x_max=int(round(float(x_max))),
                y_max=int(round(float(y_max))),
                score=float(score),
                label=name,
                class_id=class_id,
            )
            candidates.append(bbox)

        if not candidates:
            return None

        if label is not None:
            label_lower = str(label).lower()
            matches = [box for box in candidates if box.label.lower() == label_lower]
            if matches:
                return max(matches, key=lambda box: box.score)

        return max(candidates, key=lambda box: box.score)

    @staticmethod
    def _to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
        """Normalise the supported image inputs into an RGB ndarray."""
        if isinstance(image, np.ndarray):
            if image.ndim == 2:  # grayscale
                return np.stack((image,) * 3, axis=-1)
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            if image.ndim == 3 and image.shape[2] == 4:
                return image[..., :3]
            raise ValueError(
                f"Unsupported ndarray shape for YOLO detection: {image.shape!r}"
            )

        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                return np.array(image.convert("RGB"))
            return np.array(image)

        raise TypeError(
            "`image` must be a PIL.Image.Image or numpy.ndarray instance; "
            f"got {type(image)!r}"
        )
