"""Utilities related to masking CO3D images."""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor

import datasets
from datasets import Image as DatasetImage
from PIL import Image, ImageDraw
from tqdm import tqdm


def _clamped_normal(
    mean: float, std_dev: float, min_value: float, max_value: float
) -> float:
    """Sample from a normal distribution and clamp the result to the desired range."""

    sample = random.normalvariate(mean, std_dev)
    return max(min(sample, max_value), min_value)


def _mask_fill_for_mode(mode: str):
    if mode == "RGBA":
        return (0, 0, 0, 0)
    if mode == "RGB":
        return (0, 0, 0)
    if mode == "LA":
        return (0, 0)
    return 0


def mask_record(image: Image.Image) -> Image.Image:
    """
    Apply a solid mask near the image centre with slight
    randomness in size and position.
    """

    masked = image.copy()
    width, height = masked.size

    mask_width_ratio = _clamped_normal(0.3, 0.05, 0.2, 0.4)
    mask_height_ratio = _clamped_normal(0.3, 0.05, 0.2, 0.4)

    mask_width = max(1, int(width * mask_width_ratio))
    mask_height = max(1, int(height * mask_height_ratio))

    centre_x_ratio = _clamped_normal(0.5, 0.1, 0.35, 0.65)
    centre_y_ratio = _clamped_normal(0.5, 0.1, 0.35, 0.65)

    centre_x = int(width * centre_x_ratio)
    centre_y = int(height * centre_y_ratio)

    left = max(0, min(centre_x - mask_width // 2, width - mask_width))
    top = max(0, min(centre_y - mask_height // 2, height - mask_height))
    right = left + mask_width
    bottom = top + mask_height

    draw = ImageDraw.Draw(masked)
    draw.rectangle(
        [left, top, right - 1, bottom - 1], fill=_mask_fill_for_mode(masked.mode)
    )

    return masked


def main():
    dataset = datasets.load_dataset("pufanyi/co3d-filtered", split="train")
    image_feature = DatasetImage()
    predict_images = dataset["predict_image"]

    def to_mask(image: Image.Image) -> dict:
        masked = mask_record(image)
        return image_feature.encode_example(masked)

    max_workers = min(32, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        masked_dataset = list(
            tqdm(
                executor.map(to_mask, predict_images),
                total=len(predict_images),
                desc="Applying mask",
            )
        )

    dataset = dataset.add_column("masked_image", masked_dataset)
    dataset = dataset.cast_column("masked_image", image_feature)
    dataset.push_to_hub("pufanyi/co3d-masked")


if __name__ == "__main__":
    main()
