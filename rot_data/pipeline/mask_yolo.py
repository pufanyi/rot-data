"""Pipeline that applies YOLO-guided masks to the CO3D dataset."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterable, Sequence
from queue import Queue
from typing import Any

import datasets
from datasets import Image as DatasetImage
from loguru import logger
from PIL import Image, ImageDraw
from tqdm import tqdm

from rot_data.models.yolo import BoundingBox, YOLOBoundingBoxDetector
from rot_data.pipeline.mask import mask_record

DATASET_REPO_ID = "pufanyi/co3d-filtered"
OUTPUT_REPO_ID = "pufanyi/co3d-masked-yolo"
DEFAULT_MODEL_PATH = "yolo11x.pt"


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
        "--devices",
        default=None,
        help=(
            "Comma-separated list of devices assigned to workers. Overrides --device "
            "when provided."
        ),
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
        "--batch-size",
        type=int,
        default=16,
        help="Number of images per YOLO inference batch",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers to use for inference",
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
            f"Discarding degenerate bbox ({bbox}) for image sized {width}x{height}"
        )
        return masked

    draw = ImageDraw.Draw(masked)
    draw.rectangle(
        [left, top, right - 1, bottom - 1],
        fill=_mask_fill_for_mode(masked.mode),
    )
    return masked


def _autodetect_devices() -> list[str | None]:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return []

    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]

    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return ["mps"]
    except AttributeError:  # pragma: no cover - defensive guard for older torch
        pass

    return []


def _resolve_devices(devices_arg: str | None, device: str | None) -> list[str | None]:
    if devices_arg:
        parsed = [entry.strip() for entry in devices_arg.split(",")]
        devices = [entry or None for entry in parsed if entry or entry == ""]
        resolved = [dev if dev is not None else device for dev in devices]
        return resolved or [device]
    if device is not None:
        return [device]

    auto_devices = _autodetect_devices()
    if auto_devices:
        return auto_devices

    return [None]


def _chunk_records(dataset: Iterable, batch_size: int):
    batch: list[dict[str, Any]] = []
    for record in dataset:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _run_batch(
    detector: YOLOBoundingBoxDetector,
    batch: Sequence[dict[str, Any]],
) -> tuple[list[Image.Image], int]:
    images = [record["predict_image"] for record in batch]
    labels = [record.get("label") for record in batch]
    bboxes = detector.detect_batch(images, labels)

    masked_images: list[Image.Image] = []
    missing = 0
    for record, bbox in zip(batch, bboxes, strict=True):
        image = record["predict_image"]
        if bbox is None:
            missing += 1
            masked_images.append(mask_record(image))
            continue
        masked_images.append(mask_image_with_bbox(image, bbox))

    return masked_images, missing


def mask_dataset(args: argparse.Namespace) -> datasets.Dataset:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    logger.info(f"Loading dataset {args.dataset_repo!r} from the Hugging Face Hub")
    dataset = datasets.load_dataset(args.dataset_repo, split="train")
    if not isinstance(dataset, datasets.Dataset):
        raise RuntimeError("Expected a single dataset split, got a DatasetDict")

    devices = _resolve_devices(args.devices, args.device)
    num_workers = args.num_workers or len(devices)
    if num_workers <= 0:
        raise ValueError("--num-workers must resolve to at least one worker")

    worker_devices = [devices[index % len(devices)] for index in range(num_workers)]
    worker_labels = [dev if dev is not None else "auto" for dev in worker_devices]
    devices_summary = ", ".join(worker_labels)
    logger.info(
        "Running YOLO detection with "
        f"{num_workers} workers, batch size {args.batch_size} on {devices_summary}"
    )

    task_queue: Queue[tuple[int, list[dict[str, Any]]] | None] = Queue(
        max(1, num_workers * 2)
    )
    batched_results: dict[int, list[Image.Image]] = {}
    batched_missing: dict[int, int] = {}
    results_lock = threading.Lock()

    def worker(worker_index: int, device: str | None) -> None:
        detector = YOLOBoundingBoxDetector(
            model_path=args.model_path,
            device=device,
            confidence=args.confidence,
            iou=args.iou,
        )
        while True:
            item = task_queue.get()
            if item is None:
                task_queue.task_done()
                break
            batch_idx, batch = item
            try:
                masked_batch, missing = _run_batch(detector, batch)
            except Exception as exc:  # pragma: no cover - surface worker errors
                logger.exception(f"Worker {worker_index} failed while masking a batch")
                task_queue.task_done()
                raise exc
            else:
                with results_lock:
                    batched_results[batch_idx] = masked_batch
                    batched_missing[batch_idx] = missing
                task_queue.task_done()

    workers = [
        threading.Thread(target=worker, args=(idx, device), daemon=True)
        for idx, device in enumerate(worker_devices)
    ]
    for thread in workers:
        thread.start()

    total_batches = 0
    for batch in _chunk_records(dataset, args.batch_size):
        task_queue.put((total_batches, batch))
        total_batches += 1

    for _ in workers:
        task_queue.put(None)

    task_queue.join()

    for thread in workers:
        thread.join()

    image_feature = DatasetImage()
    masked_images: list[dict[str, Any]] = []
    missing_detections = 0

    progress = tqdm(
        total=dataset.num_rows,
        desc="Collecting masks",
        leave=False,
    )
    try:
        for batch_idx in range(total_batches):
            batch_images = batched_results.get(batch_idx, [])
            missing_detections += batched_missing.get(batch_idx, 0)
            for masked_image in batch_images:
                masked_images.append(image_feature.encode_example(masked_image))
            progress.update(len(batch_images))
    finally:
        progress.close()

    if len(masked_images) != dataset.num_rows:
        raise RuntimeError(
            "Masked image count does not match dataset rows: "
            f"{len(masked_images)} vs {dataset.num_rows}"
        )

    logger.info(
        f"Processed {dataset.num_rows} items; {missing_detections} lacked detections"
    )

    dataset = dataset.add_column("masked_image", masked_images)
    dataset = dataset.cast_column("masked_image", image_feature)

    if not args.skip_push:
        logger.info(
            f"Pushing masked dataset to {args.output_repo!r} on the Hugging Face Hub"
        )
        dataset.push_to_hub(args.output_repo, split="train")
        logger.success(f"Masked dataset pushed to {args.output_repo!r}")

    return dataset


def main() -> None:
    args = parse_args()
    mask_dataset(args)


if __name__ == "__main__":
    main()
