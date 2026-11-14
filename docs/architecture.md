# Architecture

Notebooks is a `uv` workspace. Each directory under `packages/` is a topic
package with its own dependencies, notebooks, supporting modules and data. Each
directory under `lib/` is a library that topic packages share; a package lists
it as a dependency with `{ workspace = true }` in `[tool.uv.sources]`. The root
`pyproject.toml` defines the workspace and shared development tools. The
workspace lockfile is `uv.lock`.

`lib/animation` builds the Plotly figures whose frames play in the browser, so
an exported page animates without a Python kernel. `figura_animada` adds the
play and pause buttons and the slider; see [notebooks](notebooks.md#animations).

## Notebook sources

A notebook lives in the package of its topic, next to the modules and data it
reads. The catalog in [`notebooks.toml`](../notebooks.toml) is the source of
truth for the site. Each `[[topics]]` table names a package by its directory
under `packages/`, and each `[[topics.notebooks]]` entry gives a notebook's id,
title, file and summary. A notebook's slug is `topic-id/notebook-id`. The site
publishes every notebook in the catalog.

[`scripts/notebooks_manifest.py`](../scripts/notebooks_manifest.py) reads the
catalog, checks ids, paths and packages, and writes the typed frontend data to
[`site/src/lib/catalog.generated.ts`](../site/src/lib/catalog.generated.ts).
Each notebook receives a source URL to the repository and the path of its
published page.

## Export pipeline

[`scripts/export_notebooks.py`](../scripts/export_notebooks.py) owns validation,
catalog generation, and notebook export. The `validate` command checks only the
manifest. The `data` command regenerates the frontend catalog, and
`data --check` fails when the committed catalog is out of date without writing
it. The `export` command does not touch the catalog. It checks each notebook,
runs it with `marimo export html`, and writes its results as one static page. A
notebook whose check or export fails fails the command, and its page is not
written. So does a page over 1,000,000 bytes (`MAX_PAGE_BYTES`), because marimo
embeds every figure in the page: a Matplotlib figure as a base64 PNG and a
Plotly figure as JSON escaped twice, about 2.7 times its raw size. Lower the
figure resolution, plot fewer points or use fewer frames to fit.

The export rewrites the page shell that marimo generates: the catalog title and
summary replace the title and description, the language is `es` because the
notebook text is Spanish, and marimo's favicon, manifest, touch icon and two
decorative image preloads are removed. The export fails if marimo's template no
longer has one of these parts. The page still loads marimo's frontend scripts
from the jsDelivr CDN, pinned to the locked marimo version.

The pipeline runs `uv run --locked` with the package of each notebook's topic.
This keeps notebook imports scoped to the package that owns them.

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

[`site/src/routes/+page.svelte`](../site/src/routes/+page.svelte) renders the
catalog. It loads the generated site data, applies the search and topic filters
from [`site/src/lib/filters.ts`](../site/src/lib/filters.ts), and groups the
notebooks by topic. Each card links to the notebook's published page.

[`site/src/routes/+layout.ts`](../site/src/routes/+layout.ts) enables
prerendering. The SvelteKit adapter produces a static site in `site/build`. The
deployment workflow in
[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) generates the
catalog, checks and exports the notebooks, then builds that directory for GitHub
Pages.

[`layout.md`](layout.md) defines the folder map and site build boundary. `site/`
contains only the frontend and generated site inputs. Notebook source, data,
documentation, and build tooling remain outside it.

## Package dependencies

Each package `pyproject.toml` owns the dependencies of its notebooks. The
remote-sensing package declares rasterio as a normal dependency for its raster
notebook. Package requirements are documented in
[`environment.md`](environment.md).
