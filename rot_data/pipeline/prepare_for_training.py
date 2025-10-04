import json
from pathlib import Path

import datasets
import tqdm
from PIL import Image

DATASET = "pufanyi/co3d"

TEMPLATE = """\
<image><image><image> You are given {num_views} camera views of the same scene. \
The last image has a black masked part. Please generate the \
last image and complete the black part.
"""


def save_image(image: Image.Image, folder: Path, path: str):
    image.save(folder / path)
    return path


if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET, split="train")
    folder = Path("data/co3d")
    folder.mkdir(exist_ok=True, parents=True)
    jsonl_data_path = Path(folder, "data.jsonl")
    image_folder = Path(folder, "images")
    image_folder.mkdir(exist_ok=True, parents=True)

    with open(jsonl_data_path, "w") as f:
        for i, item in enumerate(tqdm.tqdm(dataset, desc="Processing dataset")):
            raw_data: dict = item
            item_id = raw_data["id"]
            image_names = [
                save_image(img, image_folder, f"{item_id}_{j}.jpg")
                for j, img in enumerate(raw_data["images"])
            ]
            masked_image_name = save_image(
                raw_data["masked_image"], image_folder, f"{item_id}_masked.jpg"
            )
            predict_image_name = save_image(
                raw_data["predict_image"], image_folder, f"{item_id}_predict.jpg"
            )
            final_data = {
                "id": i,
                "conversations": [
                    {
                        "from": "human",
                        "value": TEMPLATE.format(num_views=len(image_names)),
                    },
                    {"from": "gpt", "value": "<image>"},
                ],
                "image": image_names + [masked_image_name, predict_image_name],
            }
            output_line = json.dumps(final_data)
            f.write(output_line + "\n")
