import cv2
import datasets
import numpy as np
import tqdm
from PIL import Image

DATASET_REPO_ID = "pufanyi/co3d"
OUTPUT_REPO_ID = "pufanyi/co3d-filtered"


def compute_laplacian_variance(image: Image.Image) -> float:
    """Compute Laplacian variance for image sharpness assessment."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad(image: Image.Image) -> float:
    """Compute Tenengrad (Sobel gradient energy) for image sharpness assessment."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.mean(gradient_magnitude**2))


if __name__ == "__main__":
    dataset = datasets.load_dataset(DATASET_REPO_ID, split="train")
    output_dataset = []
    for record in tqdm.tqdm(dataset):
        # Collect all three images
        all_images = list(record["images"]) + [record["predict_image"]]

        # Compute sizes
        sizes = [image.size for image in all_images]

        # Compute Laplacian variance for each image
        laplacian_vars = [compute_laplacian_variance(img) for img in all_images]

        # Compute Tenengrad for each image
        tenengrads = [compute_tenengrad(img) for img in all_images]

        output_dataset.append(
            {
                "id": record["id"],
                "label": record["label"],
                "images": record["images"],
                "predict_image": record["predict_image"],
                "size_0": sizes[0],
                "size_1": sizes[1],
                "size_2": sizes[2],
                "laplacian_var_0": laplacian_vars[0],
                "laplacian_var_1": laplacian_vars[1],
                "laplacian_var_2": laplacian_vars[2],
                "tenengrad_0": tenengrads[0],
                "tenengrad_1": tenengrads[1],
                "tenengrad_2": tenengrads[2],
            }
        )

    output_dataset = datasets.Dataset.from_list(output_dataset)
    output_dataset.push_to_hub(OUTPUT_REPO_ID, split="train")
