# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the CLI entry point; keep argument parsing and user prompts here while delegating heavy logic to modules under `rot_data/`.
- `rot_data/dataloader/` defines core abstractions in `data.py` and concrete sources like `co3d.py`; add new loaders alongside existing ones and register them through the shared interfaces.
- Lightweight dataset metadata such as URL manifests live in `rot_data/links/`; avoid committing raw assets and reference remote storage instead.
- Mirror runtime modules with tests under `tests/`; for example, changes in `rot_data/dataloader/co3d.py` should be covered in `tests/dataloader/test_co3d.py`.

## Build, Test, and Development Commands
- `uv sync` installs locked dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py --help` surfaces available CLI actions before running them.
- `uv run ruff check rot_data main.py` enforces import ordering, formatting, and modernization rules.
- YOLO pipelines accept larger checkpoints (e.g. YOLO11x); pass the weight via `--model-path` and scale `--batch-size`/`--devices` to match available GPUs.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case for functions and modules, and PascalCase for classes such as `CO3DDataLoader`.
- Keep modules cohesive and favour pure helpers in `rot_data/utils.py`; annotate public APIs with typing hints and lightweight docstrings when behaviour is non-obvious.
- Run Ruff before committing to auto-fix import order (`ruff --fix --select I,UP` if needed).
- Loguru prefers f-string formatting; use `logger.info(f"Masked dataset pushed to {repo}")` instead of `%s`.

## Testing Guidelines
- Use `pytest` for unit coverage; name files `test_<module>.py` and mark async flows with `@pytest.mark.asyncio`.
- Exercise edge cases for download retries, cache hydration, and loader selection; prefer in-memory or `tmp_path` fixtures when sample assets are required.
- Add regression tests whenever introducing new data sources or caching branches.

## Commit & Pull Request Guidelines
- Match the repo history with single-line imperative commits (`Add CO3D cache eviction guard`).
- Keep PRs focused, link related issues, and document data source implications or new configuration knobs.
- Confirm lint, tests, and critical CLI flows in the PR description; attach command snippets or output when behaviour changes.

## Security & Configuration Tips
- Store secrets outside the repo and expose them via environment variables read inside loaders.
- Default caches to `cache/`; extend `.gitignore` if you introduce new directories for downloaded artifacts.
