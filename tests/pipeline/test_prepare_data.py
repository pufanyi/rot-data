import json
from pathlib import Path

import pytest
from PIL import Image

from rot_data.dataloader.data import Data, DataLoader
from rot_data.pipeline import prepare_data
from rot_data.pipeline.prepare_data import (
    DatasetPreparationError,
    prepare_and_push,
    prepare_dataset,
    push_dataset_to_hub,
)


class DummyLoader(DataLoader):
    def __init__(self, samples: list[Data]):
        self._samples = samples

    def load(self):
        yield from self._samples


def _make_sample(sample_id: str, label: str = "dummy") -> Data:
    def _image(color: tuple[int, int, int]) -> Image.Image:
        img = Image.new("RGB", (8, 8), color=color)
        return img

    images = [_image((255, 0, 0)), _image((0, 255, 0))]
    predict = _image((0, 0, 255))
    return Data(images=images, predict_image=predict, label=label, _id=sample_id)


def test_prepare_dataset_writes_metadata_and_images(tmp_path: Path) -> None:
    loader = DummyLoader([_make_sample("sample-id")])

    prepared = prepare_dataset(loader, tmp_path, overwrite=True)

    assert len(prepared) == 1
    sample = prepared[0]
    assert sample.id == "sample-id"

    metadata_path = tmp_path / "metadata.jsonl"
    assert metadata_path.exists()
    metadata_entries = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert metadata_entries[0]["id"] == "sample-id"

    for relative_path in sample.context_images + [sample.predict_image]:
        absolute_path = tmp_path / relative_path
        assert absolute_path.exists()
        with Image.open(absolute_path) as img:
            assert img.size == (8, 8)


def test_prepare_dataset_handles_duplicate_ids(tmp_path: Path) -> None:
    loader = DummyLoader([_make_sample("duplicate"), _make_sample("duplicate")])
    prepared = prepare_dataset(loader, tmp_path, overwrite=True)

    assert len(prepared) == 2
    assert prepared[0].context_images[0] != prepared[1].context_images[0]


def test_push_dataset_to_hub_invokes_api(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    class StubApi:
        def __init__(self) -> None:
            self.created: list[dict[str, str]] = []
            self.uploaded: list[dict[str, str]] = []

        def create_repo(self, **kwargs):
            self.created.append(kwargs)

        def upload_folder(self, **kwargs):
            self.uploaded.append(kwargs)

    api = StubApi()

    push_dataset_to_hub(
        tmp_path,
        "user/repo",
        token="token",
        commit_message="message",
        api=api,
    )

    assert api.created[0]["repo_id"] == "user/repo"
    assert api.uploaded[0]["folder_path"] == str(tmp_path.resolve())
    assert api.uploaded[0]["commit_message"] == "message"


def test_prepare_and_push_pipeline(monkeypatch, tmp_path: Path) -> None:
    samples = [_make_sample("sample-123")]
    loader = DummyLoader(samples)

    def fake_resolve(name: str, **_kwargs) -> DataLoader:
        return loader

    uploaded: list[Path] = []

    def fake_push(folder: Path, repo_id: str, **_kwargs) -> None:
        uploaded.append(folder)

    monkeypatch.setattr(prepare_data, "resolve_loader", fake_resolve)
    monkeypatch.setattr(prepare_data, "push_dataset_to_hub", fake_push)

    output_dir = tmp_path / "prepared"
    result = prepare_and_push(
        dataset="co3d",
        repo_id="user/repo",
        output_dir=output_dir,
        push_to_hub=True,
        overwrite_output=True,
        max_samples=1,
    )

    assert result == output_dir
    assert (output_dir / "metadata.jsonl").exists()
    assert uploaded[0] == output_dir


def test_prepare_and_push_requires_samples(monkeypatch, tmp_path: Path) -> None:
    loader = DummyLoader([])

    def fake_resolve(name: str, **_kwargs) -> DataLoader:
        return loader

    monkeypatch.setattr(prepare_data, "resolve_loader", fake_resolve)

    with pytest.raises(DatasetPreparationError):
        prepare_and_push(
            dataset="co3d",
            repo_id="user/repo",
            output_dir=tmp_path / "prepared",
            push_to_hub=False,
            overwrite_output=True,
        )
