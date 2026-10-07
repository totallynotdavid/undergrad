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
relative to the package directory. Set `export = true` to publish a notebook
on the site. Set `export = false` to link its source instead. The default mode
is `interactive` for published entries and `source` for local entries. A
`results` mode publishes static HTML results.

After changing the catalog, validate it and regenerate the site data:

```sh
uv run python scripts/export_notebooks.py validate
uv run python scripts/export_notebooks.py data
```

The export command checks and publishes every entry with `export = true`:

```sh
uv run python scripts/export_notebooks.py export --clean
```

Native extensions and external services are local requirements. See
[`environment.md`](environment.md) for GDAL and Fortran setup. A notebook that
uses Earth Engine or local data must explain its required inputs in its visible
marimo cells.
