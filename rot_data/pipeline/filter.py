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


def filter_function(data_info: dict[str, Any]) -> bool:
    return (
        data_info["min_area"] >= 1.5e6
        and data_info["min_laplacian_variance"] >= 300
        and data_info["min_tenengrad"] >= 10000
    )


if __name__ == "__main__":
    data_info = get_data_info(Path("output/statistics/statistics.jsonl"))
    print(len(data_info))
    for k, v in data_info.items():
        print(k, v)
        break
    dataset = datasets.load_dataset("pufanyi/co3d-raw", split="train")
    dataset = dataset.filter(lambda x: filter_function(data_info[x["id"]]), num_proc=32)
    dataset.push_to_hub("pufanyi/co3d-filtered")
