from tkinter import image_names
import datasets
from pathlib import Path

DATASET = "pufanyi/co3d"


if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET, split="train")
    folder = Path("data/co3d")
    folder.mkdir(exist_ok=True, parents=True)
    jsonl_data_path = Path(folder, "data.jsonl")
    image_folder = Path(folder, "images")
    with open(jsonl_data_path, "w") as f:
        for item in dataset:
            raw_data: dict = item
            # {'id': '444_63769_125899', 'label': 'apple', 'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=704x1253 at 0x7F9EFACE4B90>, <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=704x1253 at 0x7F9EFACE4F50>], 'predict_image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=704x1253 at 0x7F9EFA3675C0>, 'masked_image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=704x1253 at 0x7F9EFAD20050>}
            item_id = raw_data["id"]
            image_names = [ f"{item_id}_{j}.jpg" for j in range(len(raw_data["images"])) ]
            print(item_id, image_names)
            
