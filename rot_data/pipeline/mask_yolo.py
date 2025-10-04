"""Pipeline that applies YOLO-guided masks to the CO3D dataset."""

from __future__ import annotations

import argparse
import math
import threading
from collections.abc import Iterable, Iterator, Sequence
from queue import Queue
from random import Random
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
MASK_MIN_RATIO = 0.10
MASK_MAX_RATIO = 0.25
MASK_DEFAULT_RATIO = 0.18
MASK_RATIO_STDDEV = (MASK_MAX_RATIO - MASK_MIN_RATIO) / 6
MASK_RATIO_SAMPLE_COUNT = 10

_ratio_rng = Random()


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


def _compute_mask_dimensions(
    width: int,
    height: int,
    target_ratio: float,
) -> tuple[int, int, float]:
    """Return mask dimensions that respect the target area ratio constraints."""

    ratio = max(MASK_MIN_RATIO, min(MASK_MAX_RATIO, target_ratio))
    if width <= 0 or height <= 0:
        return 0, 0, 0.0

    image_area = width * height
    scale = math.sqrt(ratio)
    mask_width = max(1, min(width, int(round(width * scale))))
    mask_height = max(1, min(height, int(round(height * scale))))

    def mask_ratio() -> float:
        return (mask_width * mask_height) / image_area

    current_ratio = mask_ratio()

    # Expand if rounding shrank the area below the lower bound.
    while current_ratio < MASK_MIN_RATIO and (
        mask_width < width or mask_height < height
    ):
        if mask_width < width:
            mask_width += 1
        if mask_height < height and mask_ratio() < MASK_MIN_RATIO:
            mask_height += 1
        current_ratio = mask_ratio()

    # Shrink if rounding inflated the area beyond the upper bound.
    while current_ratio > MASK_MAX_RATIO and (mask_width > 1 or mask_height > 1):
        if mask_width > 1:
            mask_width -= 1
            current_ratio = mask_ratio()
            if current_ratio <= MASK_MAX_RATIO:
                break
        if mask_height > 1:
            mask_height -= 1
        current_ratio = mask_ratio()

    return mask_width, mask_height, current_ratio


def _expand_mask_for_strict_overlap(
    mask_width: int,
    mask_height: int,
    image_width: int,
    image_height: int,
    bbox_width: int,
    bbox_height: int,
) -> tuple[int, int]:
    """Ensure the mask can overlap at least half of the bounding box area."""

    if bbox_width <= 0 or bbox_height <= 0:
        return mask_width, mask_height

    required_overlap = math.ceil(bbox_width * bbox_height * 0.5)
    if required_overlap <= 0:
        return mask_width, mask_height

    adjusted_width = mask_width
    adjusted_height = mask_height

    for _ in range(5):
        overlap_width = min(adjusted_width, bbox_width)
        overlap_height = min(adjusted_height, bbox_height)

        if overlap_width <= 0 or overlap_height <= 0:
            break

        if overlap_width * overlap_height >= required_overlap:
            break

        updated = False

        if overlap_width < bbox_width and adjusted_width < image_width:
            needed_width = math.ceil(
                required_overlap / max(1, overlap_height)
            )
            new_width = min(image_width, max(adjusted_width, needed_width))
            if new_width != adjusted_width:
                adjusted_width = new_width
                updated = True

        overlap_width = min(adjusted_width, bbox_width)
        overlap_height = min(adjusted_height, bbox_height)
        if overlap_width * overlap_height >= required_overlap:
            break

        if overlap_height < bbox_height and adjusted_height < image_height:
            needed_height = math.ceil(
                required_overlap / max(1, overlap_width)
            )
            new_height = min(image_height, max(adjusted_height, needed_height))
            if new_height != adjusted_height:
                adjusted_height = new_height
                updated = True

        if not updated:
            break

    return adjusted_width, adjusted_height


def _iter_mask_ratios(rng: Random | None = None) -> Iterator[float]:
    """Yield candidate mask area ratios sampled from a truncated normal distribution."""

    generator = rng or _ratio_rng

    for _ in range(MASK_RATIO_SAMPLE_COUNT):
        sample = generator.normalvariate(MASK_DEFAULT_RATIO, MASK_RATIO_STDDEV)
        yield max(MASK_MIN_RATIO, min(MASK_MAX_RATIO, sample))


def _candidate_positions(
    start: int,
    end: int,
    mask_extent: int,
    limit: int,
) -> set[int]:
    """Generate plausible mask offsets along a single axis."""

    if limit <= 0:
        return {0}

    span = max(0, end - start)
    candidates: set[int] = set()
    centres = (
        (start + end - mask_extent) // 2,
        start,
        end - mask_extent,
        start + max(1, span // 4),
        end - mask_extent - max(1, span // 4),
    )
    for base in centres:
        candidates.add(_clamp(base, 0, limit))

    step_extent = max(1, mask_extent // 6)
    for offset in range(-mask_extent, mask_extent + 1, step_extent):
        candidates.add(_clamp((start + end - mask_extent) // 2 + offset, 0, limit))

    step_span = max(1, max(span // 3, 1))
    for offset in range(-span, span + 1, step_span):
        candidates.add(_clamp(start + offset, 0, limit))
        candidates.add(_clamp(end - mask_extent + offset, 0, limit))

    # Ensure boundary positions are always considered.
    candidates.add(0)
    candidates.add(limit)
    return candidates


def _find_mask_bounds(
    image_width: int,
    image_height: int,
    mask_width: int,
    mask_height: int,
    bbox_left: int,
    bbox_top: int,
    bbox_right: int,
    bbox_bottom: int,
) -> tuple[tuple[int, int, int, int], bool] | None:
    bbox_width = bbox_right - bbox_left
    bbox_height = bbox_bottom - bbox_top
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    bbox_area = bbox_width * bbox_height
    if bbox_area <= 1:
        return None

    mask_area = mask_width * mask_height
    if mask_area <= 0:
        return None

    strict_overlap = math.ceil(bbox_area * 0.5)
    max_overlap = min(mask_area, bbox_area - 1)
    if max_overlap <= 0:
        return None

    required_overlap = min(strict_overlap, max_overlap)

    limit_x = max(0, image_width - mask_width)
    limit_y = max(0, image_height - mask_height)

    left_candidates = _candidate_positions(bbox_left, bbox_right, mask_width, limit_x)
    top_candidates = _candidate_positions(bbox_top, bbox_bottom, mask_height, limit_y)

    target_overlap = min(
        max_overlap,
        max(required_overlap, int(round(bbox_area * 0.75))),
    )

    best_bounds: tuple[int, int, int, int] | None = None
    best_diff: float | None = None
    best_overlap = -1

    for left in left_candidates:
        right = min(image_width, left + mask_width)
        for top in top_candidates:
            bottom = min(image_height, top + mask_height)

            overlap_width = min(bbox_right, right) - max(bbox_left, left)
            if overlap_width <= 0:
                continue
            overlap_height = min(bbox_bottom, bottom) - max(bbox_top, top)
            if overlap_height <= 0:
                continue

            overlap_area = overlap_width * overlap_height
            if overlap_area < required_overlap or overlap_area >= bbox_area:
                continue

            diff = abs(overlap_area - target_overlap)
            if (
                best_bounds is None
                or diff < (best_diff or float("inf"))
                or (diff == best_diff and overlap_area > best_overlap)
            ):
                best_bounds = (left, top, right, bottom)
                best_diff = diff
                best_overlap = overlap_area

    if best_bounds is None:
        return None

    meets_strict_requirement = best_overlap >= strict_overlap

    return best_bounds, meets_strict_requirement


def mask_image_with_bbox(image: Image.Image, bbox: BoundingBox | None) -> Image.Image:
    """Return *image* with the YOLO bounding box region filled using a solid mask."""

    masked = image.copy()

    if bbox is None:
        return masked

    width, height = masked.size
    bbox_left = _clamp(bbox.x_min, 0, width)
    bbox_top = _clamp(bbox.y_min, 0, height)
    bbox_right = _clamp(bbox.x_max, 0, width)
    bbox_bottom = _clamp(bbox.y_max, 0, height)

    if bbox_left >= bbox_right or bbox_top >= bbox_bottom:
        # logger.debug(
        #     f"Discarding degenerate bbox ({bbox}) for image sized {width}x{height}"
        # )
        return masked

    bbox_width = bbox_right - bbox_left
    bbox_height = bbox_bottom - bbox_top

    if bbox_width <= 0 or bbox_height <= 0:
        return masked

    for ratio in _iter_mask_ratios():
        mask_width, mask_height, _ = _compute_mask_dimensions(width, height, ratio)
        if mask_width == 0 or mask_height == 0:
            continue

        mask_width, mask_height = _expand_mask_for_strict_overlap(
            mask_width,
            mask_height,
            width,
            height,
            bbox_width,
            bbox_height,
        )

        result = _find_mask_bounds(
            width,
            height,
            mask_width,
            mask_height,
            bbox_left,
            bbox_top,
            bbox_right,
            bbox_bottom,
        )
        if result is None:
            continue

        bounds, meets_strict = result
        left, top, right, bottom = bounds
        draw = ImageDraw.Draw(masked)
        draw.rectangle(
            [left, top, right - 1, bottom - 1],
            fill=_mask_fill_for_mode(masked.mode),
        )
        # if not meets_strict:
        #     logger.debug(
        #         "Applied best-effort mask that covers less than half of bounding "
        #         f"box area for bbox {bbox}"
        #     )
        return masked

    logger.warning(
        "Unable to satisfy mask constraints for bbox "
        f"{bbox} on image sized {width}x{height}; skipping targeted mask"
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
