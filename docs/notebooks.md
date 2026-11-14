# Notebooks

Notebooks are Python files managed by marimo. The package of a topic owns the
dependencies of its notebooks. Open a notebook with its package:

```sh
uv run --package dynamics marimo edit packages/dynamics/pendulo_simple.py
```

Check the same notebook without opening the editor:

```sh
uv run --package dynamics marimo check packages/dynamics/pendulo_simple.py
```

The repository-wide check reads [`notebooks.toml`](../notebooks.toml), validates
every catalog path, and checks every listed notebook:

```sh
mise run lint:notebooks
```

## Catalog entries

Add a notebook to `notebooks.toml` under the `[[topics]]` table of its package.
For a new topic, create the package under `packages/` with its `pyproject.toml`,
then add a `[[topics]]` table whose `id` is the package directory. The `file`
path is relative to the package directory. Every notebook in the catalog is
published on the site as a read-only page of its results.

After changing the catalog, validate it and regenerate the site data, then
commit the regenerated `site/src/lib/catalog.generated.ts`.
`mise run lint:catalog` fails when that file is out of date:

```sh
uv run python scripts/export_notebooks.py validate
mise run data
```

The export command checks every notebook, runs it, and writes its results page.
It stops with an error when a cell fails or the page is too large (see
[architecture](architecture.md#export-pipeline)), and it writes no page for that
notebook:

```sh
uv run python scripts/export_notebooks.py export --clean
```

marimo renders each figure at twice its matplotlib `figure.dpi` and shows it at
the size it would have at 100 dpi. Notebooks that publish many figures lower
`plt.rcParams["figure.dpi"]` (to 72, or 70 for the tsunami notebook) so the page
stays under the size limit.

## Tests

Each package with tests has a task named after it. `mise run test` runs them
all, with the export script and site tests:

| Task                                | What it checks                                                                                              |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `mise run test:numerical-methods`   | Runs the differentiation, root finding, interpolation and floating-point notebooks and checks their numbers |
| `mise run test:statistical-physics` | The molecular dynamics module, and the hypersphere notebook, which computes with numba                      |
| `mise run test:remote-sensing`      | The offline GaoFen-1 atmospheric correction                                                                 |
| `mise run test:geophysics`          | The `tsunamis` module and the INGEMMET hazard inventory download                                            |
| `mise run test:scripts`             | The catalog and export scripts                                                                              |

Package dependencies and external services are local requirements. A notebook
that uses Earth Engine or local data must explain its required inputs in its
visible marimo cells.
