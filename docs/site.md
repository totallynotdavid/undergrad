# Site

The site is a prerendered SvelteKit application in [`site/`](../site/). The
catalog generator writes the course and notebook data before the site build.
Before a site build, follow the catalog validation, data generation, and export
workflow in [`notebooks.md`](notebooks.md).

Install the site dependencies and build it:

```sh
bun install --cwd site --frozen-lockfile
bun run --cwd site build
```

The build writes the static site to `site/build`. After generating the catalog
data and exported notebook assets as described in
[`notebooks.md`](notebooks.md), start the Vite development server:

```sh
bun run --cwd site dev
```

Run the export command before opening an exported notebook in the development
server. `NotebookCard` uses the generated asset path for exported entries. The
generated files in `site/static/notebooks` are gitignored rather than checked
in.
