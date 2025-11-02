# Repository Guidelines

## Project Structure & Module Organization
Place all runtime code directly under `toy_translator/` at the repository root—skip an intermediate `src/` layer. Keep core translation logic in focused modules (for example `toy_translator/phrasebook.py` or `toy_translator/pipelines.py`), park CLI helpers in `toy_translator/cli.py`, and share constants via `toy_translator/settings.py`. Put throwaway explorations in `notes/` and small sample assets in `assets/`. Prefer lightweight modules over deep packages so navigation stays flat.

## Build, Test, and Development Commands
**Always use `uv` for running Python commands.** Use `uv sync` to recreate the virtual environment and `uv run python -m <module>` to execute any Python script or module. For example:
- Run the pipeline: `uv run python toy-translator.py`
- Run a specific module: `uv run python -m toy_translator.convert_dataset`
- Manual testing: `uv run python -c "from toy_translator.phrasebook import translate; print(translate('안녕'))"`

If a dependency becomes unavoidable, record it with `uv add <package>` so the minimal `pyproject.toml` stays authoritative; otherwise leave the file alone.

## Coding Style & Naming Conventions
Target Python 3.11+, with 4-space indentation, type hints on outward-facing functions, snake_case for functions, PascalCase for classes, and UPPER_CASE for constants. Keep modules lean (<300 lines) and split helpers when they grow large. Use whatever editor formatting you prefer, but keep imports explicit and avoid wildcard imports.

## Testing Guidelines
This toy project relies on manual verification instead of a dedicated test suite. When adding features, include a short reproduction snippet in the pull request description (for example, a `uv run python -c "from toy_translator.phrasebook import translate; print(translate('안녕'))"` command). Keep modules self-checking with docstring examples or lightweight assertions guarded by `if __name__ == "__main__":` blocks when helpful.

## Commit & Pull Request Guidelines
Start commit subjects with a short imperative prefix (`feat`, `fix`, `docs`, etc.), then detail every meaningful change as bullet points in the commit body. Each bullet should mention the file or module touched and what changed (e.g., `- toy_translator/phrasebook.py: add basic Korean-to-English map`). Pull requests should include the same bullet list plus manual QA notes or sample translations so reviewers can confirm behavior quickly. Rebase on `main` before requesting review.

## Environment & Configuration Notes
Document secrets and API keys in `.env.example`, and load them at runtime via `dotenv` if needed. Never commit personal `.env` files—only reference keys from code and ensure `.env` stays git-ignored. When adding external services, provide a short section in `README.md` with the required environment variables and any `uv run` commands needed for setup.

## Collaboration Workflow
Before writing code, investigate the problem, sketch the design, and review both with the maintainer; do not start implementation until explicit approval is given. After receiving an implementation directive, restate this guideline to confirm alignment, then proceed with the requested task.
Only create commits when explicitly instructed; never commit changes on your own initiative.
