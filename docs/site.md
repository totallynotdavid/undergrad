# Site

The site is a prerendered SvelteKit application in [`site/`](../site/). The
catalog generator writes the course and notebook data to a committed file that
the site build reads. Before a site build, follow the catalog validation, data
generation, and export workflow in [`notebooks.md`](notebooks.md).

Install the dependencies (see [`environment.md`](environment.md)), then export
the notebooks and build the site:

```sh
mise run build
```

The build writes the static site to `site/build`. Check the site's code with
`mise run lint:web`, `mise run typecheck` and `mise run test:web`. After
generating the catalog data and exported notebook assets as described in
[`notebooks.md`](notebooks.md), start the Vite development server:

```sh
bun run --cwd site dev
```

Run the export command before opening an exported notebook in the development
server. `NotebookCard` uses the generated asset path for exported entries. The
generated files in `site/static/notebooks` are gitignored rather than checked
in.
