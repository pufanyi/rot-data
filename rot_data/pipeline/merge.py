import json
from pathlib import Path

import datasets
from PIL import Image


def dataset_generator():
    folder = Path("output")
    count = 0
    with open("output/data.jsonl") as f:
        for line in f:
            data = json.loads(line)
            data["images"] = [Image.open(folder / image) for image in data["images"]]
            data["predict_image"] = Image.open(folder / data["predict_image"])
            yield data
            count += 1


if __name__ == "__main__":
    dataset = datasets.Dataset.from_generator(dataset_generator, keep_in_memory=True)
    dataset.push_to_hub("pufanyi/co3d")
