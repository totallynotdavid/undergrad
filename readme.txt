Quick start:
uv sync --all-packages

Repository layout:
- packages/: uv workspace members; each course owns its notebooks, local data, helper code, and dependencies
- site/: SvelteKit static site

Open a notebook:
uv run --package <package> marimo edit <notebook.py>

Check a notebook:
uv run --package <package> marimo check <notebook.py>

Check every notebook listed in the site manifest:
mise run check-notebooks

Run the GDAL notebook locally:
uv run --package tecnicas-de-teledeteccion --extra gdal marimo edit packages/tecnicas-de-teledeteccion/raster-y-gdal/clase_2_gdal.py

The GDAL binding is an optional extra because it requires native GDAL headers.
On Ubuntu, install them with:
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev

On macOS, install GDAL with:
brew install gdal

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
