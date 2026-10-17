# Contributing

Use the setup and dependency instructions in
[`docs/environment.md`](../docs/environment.md), which owns the repository tool
installation, workspace synchronization, site dependencies, and native
requirements.

For notebook changes, run the checks in
[`docs/notebooks.md`](../docs/notebooks.md). For site changes, run the checks
and build in [`docs/site.md`](../docs/site.md).

Ruff checks and formats the Python code. Oxlint checks the site, Oxfmt formats
the site and the Markdown, JSON and YAML files, Svelte check types the site, and
unittest and Vitest run the tests. Run everything CI runs with:

```sh
mise run check
```

`mise run lint`, `mise run typecheck`, `mise run test` and `mise run build` run
one part of it. Apply safe lint fixes and format the code with:

```sh
mise run fix
```

Keep generated catalog data consistent with the manifest by running
`mise run data` after changing [`notebooks.toml`](../notebooks.toml).
