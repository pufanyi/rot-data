import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

DATASET_REPO_ID = "pufanyi/co3d-raw"


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


def compute_batch_statistics_gpu(
    images: list[Image.Image], device: torch.device
) -> tuple[list[float], list[float]]:
    """
    Compute Laplacian variance and Tenengrad for a batch of images using GPU.
    Handles images of different sizes by processing them individually on GPU.

    Returns:
        tuple: (laplacian_variances, tenengrads)
    """
    # Prepare kernels on GPU (reused for all images)
    laplacian_kernel = (
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )  # [1, 1, 3, 3]

    sobel_x_kernel = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    sobel_y_kernel = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    laplacian_vars = []
    tenengrads = []

    # Process each image individually on GPU
    for img in images:
        # Convert PIL to grayscale numpy array
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Convert to torch tensor [1, 1, H, W]
        img_tensor = (
            torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Apply Laplacian
        laplacian = F.conv2d(img_tensor, laplacian_kernel, padding=1)
        lap_var = torch.var(laplacian).item()
        laplacian_vars.append(lap_var)

        # Apply Sobel
        sobel_x = F.conv2d(img_tensor, sobel_x_kernel, padding=1)
        sobel_y = F.conv2d(img_tensor, sobel_y_kernel, padding=1)
        gradient_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        tenengrad = torch.mean(gradient_magnitude**2).item()
        tenengrads.append(tenengrad)

    return laplacian_vars, tenengrads


def process_record(
    record: dict[str, Any],
) -> tuple[float, float, float, list[dict[str, float]]]:
    """Process a single record and return min statistics and per-image details (CPU version)."""
    # Collect all images
    all_images = list(record["images"]) + [record["predict_image"]]

    # Compute sizes
    sizes = [image.size for image in all_images]
    areas = [s[0] * s[1] for s in sizes]

    # Compute Laplacian variance for each image
    laplacian_vars = [compute_laplacian_variance(img) for img in all_images]

    # Compute Tenengrad for each image
    tenengrads = [compute_tenengrad(img) for img in all_images]

    # Build per-image details
    per_image_stats = []
    for i, (area, lap_var, tenengrad) in enumerate(
        zip(areas, laplacian_vars, tenengrads)
    ):
        is_predict = i == len(all_images) - 1
        per_image_stats.append(
            {
                "image_index": i,
                "is_predict_image": is_predict,
                "area": area,
                "laplacian_variance": lap_var,
                "tenengrad": tenengrad,
            }
        )

    return min(areas), min(laplacian_vars), min(tenengrads), per_image_stats


def process_batch_gpu(
    records: list[dict[str, Any]], device: torch.device
) -> list[tuple[float, float, float, list[dict[str, float]]]]:
    """Process a batch of records using GPU acceleration."""
    # Collect all images from all records
    all_images_flat = []
    record_image_counts = []

    for record in records:
        images = list(record["images"]) + [record["predict_image"]]
        all_images_flat.extend(images)
        record_image_counts.append(len(images))

    # Compute statistics on GPU in one batch
    laplacian_vars, tenengrads = compute_batch_statistics_gpu(
        all_images_flat, device
    )

    # Organize results back by record
    results = []
    idx = 0
    for count in record_image_counts:
        record_lap_vars = laplacian_vars[idx : idx + count]
        record_tenengrads = tenengrads[idx : idx + count]

        # Get images for this record to compute areas
        record_images = all_images_flat[idx : idx + count]
        areas = [img.size[0] * img.size[1] for img in record_images]

        # Build per-image details
        per_image_stats = []
        for i, (area, lap_var, tenengrad) in enumerate(
            zip(areas, record_lap_vars, record_tenengrads)
        ):
            is_predict = i == count - 1
            per_image_stats.append(
                {
                    "image_index": i,
                    "is_predict_image": is_predict,
                    "area": area,
                    "laplacian_variance": lap_var,
                    "tenengrad": tenengrad,
                }
            )

        results.append(
            (min(areas), min(record_lap_vars), min(record_tenengrads), per_image_stats)
        )
        idx += count

    return results


def process_gpu_worker(
    gpu_id: int,
    dataset_repo: str,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    verbose: bool,
    return_dict: dict,
) -> None:
    """Worker function for multi-GPU processing."""
    device = torch.device(f"cuda:{gpu_id}")

    # Load dataset
    dataset = datasets.load_dataset(dataset_repo, split="train")

    min_areas = []
    min_laplacian_vars = []
    min_tenengrads = []
    all_image_stats = []

    # Calculate number of records to process
    num_records = end_idx - start_idx
    num_batches = (num_records + batch_size - 1) // batch_size

    if verbose:
        print(
            f"GPU {gpu_id}: Processing records {start_idx} to {end_idx} "
            f"({num_records} records, {num_batches} batches)"
        )

    pbar = tqdm.tqdm(
        total=num_records,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        disable=verbose,
    )

    for i in range(num_batches):
        batch_start_idx = start_idx + i * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, end_idx)

        # Load batch
        batch_indices = list(range(batch_start_idx, batch_end_idx))
        batch_dataset = dataset.select(batch_indices)
        batch_records = [batch_dataset[j] for j in range(len(batch_dataset))]

        # Process on GPU
        batch_results = process_batch_gpu(batch_records, device)

        # Collect results
        for min_area, min_lap_var, min_tenengrad, per_image_stats in batch_results:
            min_areas.append(min_area)
            min_laplacian_vars.append(min_lap_var)
            min_tenengrads.append(min_tenengrad)
            all_image_stats.append(per_image_stats)

        pbar.update(len(batch_records))

    pbar.close()

    # Store results
    return_dict[gpu_id] = {
        "min_areas": min_areas,
        "min_laplacian_vars": min_laplacian_vars,
        "min_tenengrads": min_tenengrads,
        "all_image_stats": all_image_stats,
    }


def save_statistics_jsonl(
    min_areas: list[float],
    min_laplacian_vars: list[float],
    min_tenengrads: list[float],
    all_image_stats: list[list[dict[str, float]]],
    output_dir: Path,
) -> None:
    """Save statistics for each record and each image as a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statistics.jsonl"

    with open(output_path, "w") as f:
        for idx, (area, lap_var, tenengrad, image_stats) in enumerate(
            zip(min_areas, min_laplacian_vars, min_tenengrads, all_image_stats)
        ):
            record = {
                "record_index": idx,
                "min_area": area,
                "min_laplacian_variance": lap_var,
                "min_tenengrad": tenengrad,
                "images": image_stats,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved statistics to {output_path}")


def plot_distributions(
    min_areas: list[float],
    min_laplacian_vars: list[float],
    min_tenengrads: list[float],
    output_dir: Path,
) -> None:
    """Generate and save distribution plots for the three statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot minimum areas
    axes[0].hist(min_areas, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Minimum Area (pixels²)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Minimum Image Areas")
    axes[0].grid(True, alpha=0.3)

    # Plot minimum Laplacian variances
    axes[1].hist(min_laplacian_vars, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Minimum Laplacian Variance")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Minimum Laplacian Variances")
    axes[1].grid(True, alpha=0.3)

    # Plot minimum Tenengrads
    axes[2].hist(min_tenengrads, bins=50, edgecolor="black", alpha=0.7)
    axes[2].set_xlabel("Minimum Tenengrad")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Distribution of Minimum Tenengrads")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "statistics_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved distribution plot to {plot_path}")
    plt.close()

    # Also save individual plots for better detail
    for data, name, xlabel in [
        (min_areas, "min_areas", "Minimum Area (pixels²)"),
        (min_laplacian_vars, "min_laplacian_vars", "Minimum Laplacian Variance"),
        (min_tenengrads, "min_tenengrads", "Minimum Tenengrad"),
    ]:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {name.replace('_', ' ').title()}")
        plt.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(data)
        median_val = np.median(data)
        plt.axvline(
            mean_val,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_val:.2f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="--",
            label=f"Median: {median_val:.2f}",
        )
        plt.legend()

        individual_plot_path = output_dir / f"{name}_distribution.png"
        plt.savefig(individual_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved {name} distribution plot to {individual_plot_path}")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset by computing image quality statistics"
    )
    parser.add_argument(
        "--dataset-repo",
        default=DATASET_REPO_ID,
        help="Source dataset repository on the Hugging Face Hub",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing (CPU mode only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/statistics"),
        help="Directory to save distribution plots",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration (requires CUDA-capable device)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU processing (GPU mode only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g., 'cuda:0', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed timing information for each batch",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available GPUs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset_repo}...")
    dataset = datasets.load_dataset(args.dataset_repo, split="train")

    min_areas: list[float] = []
    min_laplacian_vars: list[float] = []
    min_tenengrads: list[float] = []
    all_image_stats: list[list[dict[str, float]]] = []

    if args.use_gpu:
        # Determine number of GPUs to use
        num_available_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            num_gpus = num_available_gpus
        else:
            num_gpus = min(args.num_gpus, num_available_gpus)

        if num_gpus == 0:
            print("No CUDA GPUs available, falling back to CPU mode")
            args.use_gpu = False

    if args.use_gpu and num_gpus > 1:
        # Multi-GPU mode
        print(f"Using {num_gpus} GPUs for parallel processing")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Split dataset across GPUs
        total_records = len(dataset)
        records_per_gpu = (total_records + num_gpus - 1) // num_gpus

        # Use multiprocessing with shared dict
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        start_time = time.time()

        for gpu_id in range(num_gpus):
            start_idx = gpu_id * records_per_gpu
            end_idx = min((gpu_id + 1) * records_per_gpu, total_records)

            if start_idx >= total_records:
                break

            p = mp.Process(
                target=process_gpu_worker,
                args=(
                    gpu_id,
                    args.dataset_repo,
                    start_idx,
                    end_idx,
                    args.batch_size,
                    args.verbose,
                    return_dict,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        elapsed_time = time.time() - start_time

        # Collect results from all GPUs
        for gpu_id in range(len(processes)):
            gpu_results = return_dict[gpu_id]
            min_areas.extend(gpu_results["min_areas"])
            min_laplacian_vars.extend(gpu_results["min_laplacian_vars"])
            min_tenengrads.extend(gpu_results["min_tenengrads"])
            all_image_stats.extend(gpu_results["all_image_stats"])

        print(f"\nMulti-GPU processing completed in {elapsed_time:.2f}s")
        print(f"Average time per GPU: {elapsed_time:.2f}s")
        print(f"Throughput: {len(min_areas) / elapsed_time:.2f} records/second")

    elif args.use_gpu and num_gpus == 1:
        # GPU mode
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(
                f"GPU Memory: "
                f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
            )

        # Process in batches
        batch_size = args.batch_size
        num_batches = (len(dataset) + batch_size - 1) // batch_size

        print(f"Processing {len(dataset)} records in {num_batches} batches...")
        pbar = tqdm.tqdm(
            total=len(dataset),
            desc="Processing records",
            position=0,
            leave=True,
            disable=args.verbose,
        )

        total_load_time = 0
        total_gpu_time = 0

        for i in range(num_batches):
            batch_start = time.time()

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))

            # Load batch data
            load_start = time.time()
            batch_indices = list(range(start_idx, end_idx))
            batch_dataset = dataset.select(batch_indices)
            batch_records = [batch_dataset[j] for j in range(len(batch_dataset))]
            load_time = time.time() - load_start
            total_load_time += load_time

            # Process batch on GPU
            gpu_start = time.time()
            batch_results = process_batch_gpu(batch_records, device)
            gpu_time = time.time() - gpu_start
            total_gpu_time += gpu_time

            # Collect results
            for min_area, min_lap_var, min_tenengrad, per_image_stats in batch_results:
                min_areas.append(min_area)
                min_laplacian_vars.append(min_lap_var)
                min_tenengrads.append(min_tenengrad)
                all_image_stats.append(per_image_stats)

            batch_total = time.time() - batch_start

            if args.verbose:
                print(
                    f"Batch {i+1}/{num_batches}: "
                    f"Load={load_time:.2f}s, GPU={gpu_time:.2f}s, "
                    f"Total={batch_total:.2f}s"
                )
            else:
                # Update progress bar
                pbar.update(len(batch_records))

        if not args.verbose:
            pbar.close()

        print("\nTiming summary:")
        print(f"  Total data loading time: {total_load_time:.2f}s")
        print(f"  Total GPU processing time: {total_gpu_time:.2f}s")
        print(
            f"  Average per batch: "
            f"{(total_load_time + total_gpu_time) / num_batches:.2f}s"
        )

    else:
        # CPU mode with multithreading
        print(f"Using CPU with {args.num_workers} worker threads")
        lock = Lock()

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_record, record): idx
                for idx, record in enumerate(dataset)
            }

            # Process completed tasks with progress bar
            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing records",
                position=0,
                leave=True,
            ):
                (
                    min_area,
                    min_laplacian_var,
                    min_tenengrad,
                    per_image_stats,
                ) = future.result()
                with lock:
                    min_areas.append(min_area)
                    min_laplacian_vars.append(min_laplacian_var)
                    min_tenengrads.append(min_tenengrad)
                    all_image_stats.append(per_image_stats)

    print(f"\nProcessed {len(min_areas)} records")
    print(
        f"Minimum areas - Mean: {np.mean(min_areas):.2f}, "
        f"Median: {np.median(min_areas):.2f}"
    )
    print(
        f"Minimum Laplacian variances - Mean: {np.mean(min_laplacian_vars):.2f}, "
        f"Median: {np.median(min_laplacian_vars):.2f}"
    )
    print(
        f"Minimum Tenengrads - Mean: {np.mean(min_tenengrads):.2f}, "
        f"Median: {np.median(min_tenengrads):.2f}"
    )

    # Save statistics to JSONL
    print("\nSaving statistics to JSONL...")
    save_statistics_jsonl(
        min_areas, min_laplacian_vars, min_tenengrads, all_image_stats, args.output_dir
    )

    # Generate and save distribution plots
    print("\nGenerating distribution plots...")
    plot_distributions(min_areas, min_laplacian_vars, min_tenengrads, args.output_dir)


if __name__ == "__main__":
    main()
