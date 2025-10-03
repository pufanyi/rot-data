import json
from pathlib import Path
from typing import Any

import datasets


def get_data_info(path: Path) -> dict[str, Any]:
    data_info = {}
    with path.open("r") as f:
        for line in f:
            data = json.loads(line)
            _id = data["record_index"]
            data_info[_id] = data
    return data_info


def filter_function(data, data_info: dict[str, Any]) -> bool:
    return (
        data["label"] not in ["car", "stopsign"]
        and data_info["min_area"] >= 6e5
        and data_info["min_laplacian_variance"] >= 200
        and data_info["min_tenengrad"] >= 4000
    )


if __name__ == "__main__":
    data_info = get_data_info(Path("output/statistics/statistics.jsonl"))
    dataset = datasets.load_dataset("pufanyi/co3d-raw", split="train")
    dataset = dataset.filter(
        lambda x: filter_function(x, data_info[x["id"]]), num_proc=32
    )
    dataset.push_to_hub("pufanyi/co3d-filtered")
