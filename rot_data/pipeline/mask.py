import cv2
import datasets
import numpy as np
import tqdm
from PIL import Image

DATASET_REPO_ID = "pufanyi/co3d"
OUTPUT_REPO_ID = "pufanyi/co3d-filtered"


if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET_REPO_ID, split="train")
    output_dataset = []
    model = some_class_that_detects_bbox("yolo")
    for record in tqdm.tqdm(dataset):
        label = record["label"]
        predict_image = record["predict_image"]
        bbox = model.detect_bbox(predict_image, label)
        