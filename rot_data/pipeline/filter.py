import datasets
import json
from pathlib import Path
from typing import Any


def get_data_info(path: Path) -> dict[str, Any]:
    data_info = {}
    with path.open("r") as f:
        for line in f:
            data = json.loads(line)
            _id = data["record_index"]
            data_info[_id] = data
    return data_info

def filter_function(record: dict[str, Any]) -> bool:
    return True


if __name__ == "__main__":
    dataset = datasets.load_dataset("pufanyi/co3d-raw", split="train")
    data_info = get_data_info(Path("output/statistics/statistics.jsonl"))
    dataset = dataset.filter(filter_function, num_proc=32)