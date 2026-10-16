# Notebooks

Course notebooks are Python files managed by marimo. A package owns the
dependencies for its notebooks. Open a notebook with the package that contains
it:

```sh
uv run --package fisica-computacional marimo edit packages/fisica-computacional/metodos-numericos/pendulos-y-osciladores/pendulo_simple.py
```

Check the same notebook without opening the editor:

```sh
uv run --package fisica-computacional marimo check packages/fisica-computacional/metodos-numericos/pendulos-y-osciladores/pendulo_simple.py
```

The repository-wide check reads [`notebooks.toml`](../notebooks.toml), validates
every catalog path, and checks every listed notebook:

```sh
mise run check-notebooks
```

## Catalog entries

Add a notebook to `notebooks.toml` under an existing course and section. For a
new course, add the course package and its catalog entries. The `file` path is
relative to the package directory. Set `export = true` to publish a notebook on
the site as a read-only page of its results. Set `export = false` to link its
source instead.

When a published page moves or merges into another, remove its entry and add a
`[[redirects]]` table so the old address keeps working. `from` is the old slug
and `to` is the slug of an exported notebook:

```toml
[[redirects]]
from = "curso/seccion/pagina-vieja"
to = "curso/seccion/pagina-nueva"
```

After changing the catalog, validate it and regenerate the site data:

```sh
uv run python scripts/export_notebooks.py validate
uv run python scripts/export_notebooks.py data
```

The export command checks every entry with `export = true`, runs it, and writes
its results page. It stops with an error when a cell fails or the page is too
large (see [architecture](architecture.md#export-pipeline)), and it writes no
page for that notebook:

```sh
uv run python scripts/export_notebooks.py export --clean
```

marimo renders each figure at twice its matplotlib `figure.dpi` and shows it at
the size it would have at 100 dpi. Notebooks that publish many figures lower
`plt.rcParams["figure.dpi"]` (to 72, or 70 for the tsunami notebook) so the page
stays under the size limit.

Test the export script itself with:

```sh
mise run test-scripts
```

The statistical mechanics notebooks compute their results with numba at export
time. Those and the numerical methods notebooks (differentiation, root finding,
interpolation and floating point) have tests that run each notebook and check
its numbers:

```sh
mise run test-notebooks
```

The tsunami notebook imports the `tsunamis` module of its package. Test that
module with:

```sh
mise run test-tsunamis
```

Native extensions and external services are local requirements. See
[`environment.md`](environment.md) for GDAL setup. A notebook that uses Earth
Engine or local data must explain its required inputs in its visible marimo
cells.
