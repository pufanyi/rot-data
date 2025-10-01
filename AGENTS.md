# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the CLI entry point; keep user-facing logic here and delegate heavy lifting to modules under `rot_data/`.
- `rot_data/dataloader/` hosts loader abstractions (`data.py` defines `Data` + `DataLoader`) and concrete implementations such as `CO3DDataLoader` in `co3d.py`; extend these when adding new sources.
- `rot_data/links/` stores lightweight metadata (e.g., `co3d.json` URLs). Place large datasets in external storage; only ship references here.
- Create companion tests under a top-level `tests/` package that mirrors the module layout.

## Build, Test, and Development Commands
- `uv sync` installs locked dependencies defined in `pyproject.toml`/`uv.lock`.
- `uv run python main.py` exercises the CLI with the currently synced environment.
- `uv run ruff check rot_data main.py` enforces linting before opening a PR.
- `uv run pytest` executes the full test suite once you add tests.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and descriptive snake_case for modules, functions, and variables; reserve PascalCase for classes like `CO3DDataLoader`.
- Keep modules cohesive; prefer pure functions inside `utils.py` and use dataclasses for structured payloads.
- Sort imports and auto-format via Ruff's `I` and `UP` rules; use type hints throughout to maintain clarity.

## Testing Guidelines
- Write focused unit tests with pytest; name files `test_<module>.py` and async cases with `pytest.mark.asyncio`.
- Cover edge cases for download retries, caching behavior, and new loader implementations; target at least the changed lines.
- When fixtures need sample assets, generate them dynamically or cache them under `tmp_path` to avoid committing binaries.

## Commit & Pull Request Guidelines
- Match the existing history: single-sentence, imperative commits (`Refactor DataLoader cache handling`) that describe the change.
- Scope each PR narrowly, referencing issues when available and documenting data sources or caching implications in the description.
- Include command output or screenshots when behavior changes, and confirm lint/tests pass in a checklist.

## Data & Caching Notes
- Use `cache/` (created automatically) for downloaded artifacts; add it to `.gitignore` if you introduce new cache paths.
- Never commit raw CO3D assets or credentials; expose configuration via env vars and document defaults in the PR.
