"""Pipeline that applies YOLO-guided masks to the CO3D dataset."""

from __future__ import annotations

import argparse
from typing import Any

import datasets
from datasets import Image as DatasetImage
from loguru import logger
from PIL import Image, ImageDraw
from tqdm import tqdm

from rot_data.models.yolo import BoundingBox, YOLOBoundingBoxDetector

DATASET_REPO_ID = "pufanyi/co3d-filtered"
OUTPUT_REPO_ID = "pufanyi/co3d-masked-yolo"
DEFAULT_MODEL_PATH = "rot_data/models/yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply YOLO-driven masks to a dataset")
    parser.add_argument(
        "--dataset-repo",
        default=DATASET_REPO_ID,
        help="Source dataset repository on the Hugging Face Hub",
    )
    parser.add_argument(
        "--output-repo",
        default=OUTPUT_REPO_ID,
        help="Destination repository for the masked dataset",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path or identifier for the YOLO weights",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string forwarded to Ultralytics (e.g. 'cpu', 'cuda:0')",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold passed to YOLO",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for YOLO NMS",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing the masked dataset to the Hub",
    )
    return parser.parse_args()


def _mask_fill_for_mode(mode: str) -> tuple[int, ...] | int:
    if mode == "RGBA":
        return (0, 0, 0, 0)
    if mode == "RGB":
        return (0, 0, 0)
    if mode == "LA":
        return (0, 0)
    return 0


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def mask_image_with_bbox(image: Image.Image, bbox: BoundingBox | None) -> Image.Image:
    """Return *image* with the YOLO bounding box region filled using a solid mask."""

    masked = image.copy()

    if bbox is None:
        return masked

    width, height = masked.size
    left = _clamp(bbox.x_min, 0, width)
    top = _clamp(bbox.y_min, 0, height)
    right = _clamp(bbox.x_max, 0, width)
    bottom = _clamp(bbox.y_max, 0, height)

    if left >= right or top >= bottom:
        logger.debug(
            "Discarding degenerate bbox (%s) for image sized %sx%s",
            bbox,
            width,
            height,
        )
        return masked

    draw = ImageDraw.Draw(masked)
    draw.rectangle(
        [left, top, right - 1, bottom - 1],
        fill=_mask_fill_for_mode(masked.mode),
    )
    return masked


def mask_dataset(args: argparse.Namespace) -> datasets.Dataset:
    logger.info(f"Loading dataset '{args.dataset_repo}' from the Hugging Face Hub")
    dataset = datasets.load_dataset(args.dataset_repo, split="train")

    detector = YOLOBoundingBoxDetector(
        model_path=args.model_path,
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
    )

    image_feature = DatasetImage()
    masked_images: list[dict[str, Any]] = []

    missing_detections = 0
    for record in tqdm(dataset, desc="Applying YOLO masks"):
        image = record["predict_image"]
        label = record.get("label")
        bbox = detector.detect_bbox(image, label)
        if bbox is None:
            missing_detections += 1
        masked_image = mask_image_with_bbox(image, bbox)
        masked_images.append(image_feature.encode_example(masked_image))

    logger.info(
        "Processed %d items; %d lacked detections",
        dataset.num_rows,
        missing_detections,
    )

    dataset = dataset.add_column("masked_image", masked_images)
    dataset = dataset.cast_column("masked_image", image_feature)

    if not args.skip_push:
        logger.info(
            "Pushing masked dataset to '%s' on the Hugging Face Hub",
            args.output_repo,
        )
        dataset.push_to_hub(args.output_repo, split="train")
        logger.success("Masked dataset pushed to '%s'", args.output_repo)

    return dataset


def main() -> None:
    args = parse_args()
    mask_dataset(args)


if __name__ == "__main__":
    main()
