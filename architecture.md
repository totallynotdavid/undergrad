# Architecture

Undergrad is a `uv` workspace. Each directory under `packages/` is a course
package with its own dependencies and teaching material. The root
`pyproject.toml` defines the workspace and shared development tools. The
workspace lockfile is `uv.lock`.

## Notebook sources

Course notebooks and supporting modules live in their course package. The
catalog in [`notebooks.toml`](notebooks.toml) is the source of truth for the
lessons shown by the site. It assigns each entry a course, section, source path,
summary, package, and publication mode.

[`scripts/notebooks_manifest.py`](scripts/notebooks_manifest.py) reads the
catalog, checks paths, slugs, and modes, and writes the typed frontend data to
[`site/src/lib/catalog.generated.ts`](site/src/lib/catalog.generated.ts). Every
entry receives a source URL to the repository. Published entries also receive a
site asset path.

## Export pipeline

[`scripts/export_notebooks.py`](scripts/export_notebooks.py) owns validation,
catalog generation, and notebook export. The `validate` command checks only the
manifest. The `data` command regenerates the frontend catalog. The `export`
command checks each published notebook and exports interactive notebooks as HTML
WASM or result notebooks as static HTML.

The pipeline runs `uv run --locked` with the package named by each catalog
entry. This keeps notebook imports scoped to the course that owns them.

```text
notebooks.toml
      |
      v
notebooks_manifest.py ----> catalog.generated.ts ----> Svelte components
      |
      v
export_notebooks.py ------> site/static/notebooks ----> static site build
```

## Site

[`site/src/routes/+page.svelte`](site/src/routes/+page.svelte) renders the
catalog. It loads the generated site data, applies the search and course filters
from [`site/src/lib/filters.ts`](site/src/lib/filters.ts), and groups entries by
course and section.

[`site/src/lib/components/NotebookCard.svelte`](site/src/lib/components/NotebookCard.svelte)
uses the `export` field to choose the published asset for exported entries and
the source URL for other entries.

[`site/src/routes/+layout.ts`](site/src/routes/+layout.ts) enables prerendering.
The SvelteKit adapter produces a static site in `site/build`. The deployment
workflow in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)
generates the catalog, checks and exports published notebooks, then builds that
directory for GitHub Pages.

## Course dependencies

Each package `pyproject.toml` owns its course dependencies. The remote-sensing
package declares the GDAL Python binding as its optional `gdal` extra. Native
requirements and version compatibility are documented in
[`docs/environment.md`](docs/environment.md). The repository installer covers
the Fortran compiler only.
