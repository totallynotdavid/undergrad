# Contributing

Use the setup and dependency instructions in
[`docs/environment.md`](docs/environment.md), which owns the repository tool
installation, workspace synchronization, site dependencies, and native
requirements.

For notebook changes, run the checks in
[`docs/notebooks.md`](docs/notebooks.md). For site changes, run the checks and
build in [`docs/site.md`](docs/site.md).

Ruff checks the Python code. The site uses Oxlint, Oxfmt, Svelte check, and
Vitest. Run the complete repository checks with:

```sh
uv run ruff check .
mise run check-notebooks
bun run --cwd site check
bun run --cwd site lint
bun run --cwd site test
```

Format site code with:

```sh
bun run --cwd site fmt
```

Keep generated catalog data consistent with the manifest by running
`uv run python scripts/export_notebooks.py data` from the repository root after
changing [`notebooks.toml`](notebooks.toml).
