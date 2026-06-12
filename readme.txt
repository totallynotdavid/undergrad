Quick start:
uv sync --all-packages

Repository layout:
- packages/: uv workspace members; each course owns its notebooks, local data, helper code, and dependencies
- site/: SvelteKit static site

Open a notebook:
uv run --package <package> marimo edit <notebook.py>

Check a notebook:
uv run --package <package> marimo check <notebook.py>

Build the GitHub Pages site locally:
uv run python scripts/export_notebooks.py export --clean
bun install --cwd site
bun run --cwd site build

Start the local site development server:
uv run python scripts/export_notebooks.py data
bun install --cwd site
bun run --cwd site dev

Some notebooks need external data or credentials.
Examples:
- packages/tecnicas-de-teledeteccion/correccion-atmosferica-gaofen1/main.py
- packages/fundamentos-de-riesgos-y-desastres/sigrid/analisis_huaicos_ancash.py
