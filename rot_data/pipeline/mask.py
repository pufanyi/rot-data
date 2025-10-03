"""Pipeline that adds YOLO bounding boxes to the CO3D dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import datasets
import tqdm
from loguru import logger

from rot_data.models.yolo import BoundingBox, YOLOBoundingBoxDetector

DATASET_REPO_ID = "pufanyi/co3d"
OUTPUT_REPO_ID = "pufanyi/co3d-with-bboxes"
DEFAULT_MODEL_PATH = Path("rot_data/models/yolov8n.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate dataset with YOLO boxes")
    parser.add_argument(
        "--dataset-repo",
        default=DATASET_REPO_ID,
        help="Source dataset repository on the Hugging Face Hub",
    )
    parser.add_argument(
        "--output-repo",
        default=OUTPUT_REPO_ID,
        help="Destination repository for the annotated dataset",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
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
        help="Skip pushing the annotated dataset to the Hub",
    )
    return parser.parse_args()


def bbox_to_payload(bbox: BoundingBox | None) -> dict[str, Any]:
    if bbox is None:
        return {
            "bbox_x_min": None,
            "bbox_y_min": None,
            "bbox_x_max": None,
            "bbox_y_max": None,
            "bbox_score": None,
            "bbox_label": None,
            "bbox_class_id": None,
        }

    return {
        "bbox_x_min": bbox.x_min,
        "bbox_y_min": bbox.y_min,
        "bbox_x_max": bbox.x_max,
        "bbox_y_max": bbox.y_max,
        "bbox_score": bbox.score,
        "bbox_label": bbox.label,
        "bbox_class_id": bbox.class_id,
    }


def annotate_dataset(args: argparse.Namespace) -> datasets.Dataset:
    logger.info("Loading dataset '%s' from the Hugging Face Hub", args.dataset_repo)
    dataset = datasets.load_dataset(args.dataset_repo, split="train")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.warning(
            f"Model file '{model_path}' not found locally. Ultralytics will attempt to download it.",
        )

    detector = YOLOBoundingBoxDetector(
        model_path=args.model_path,
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
    )

    logger.info("Running YOLO detection on %d samples", dataset.num_rows)
    annotated_records: list[dict[str, Any]] = []
    for record in tqdm.tqdm(dataset, desc="Detecting bounding boxes"):
        bbox = detector.detect_bbox(record["predict_image"], record.get("label"))
        payload = bbox_to_payload(bbox)
        annotated_records.append(
            {
                "id": record["id"],
                "label": record["label"],
                "images": record["images"],
                "predict_image": record["predict_image"],
                **payload,
            }
        )

    annotated_dataset = datasets.Dataset.from_list(annotated_records)
    logger.success(
        "Annotated dataset assembled with %d entries", annotated_dataset.num_rows
    )

    if not args.skip_push:
        logger.info(
            "Pushing annotated dataset to '%s' on the Hugging Face Hub",
            args.output_repo,
        )
        annotated_dataset.push_to_hub(args.output_repo, split="train")
        logger.success("Annotated dataset pushed to '%s'", args.output_repo)

    return annotated_dataset


def main() -> None:
    args = parse_args()
    annotate_dataset(args)


if __name__ == "__main__":
    main()
