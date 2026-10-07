# Undergrad

Undergrad is a collection of Python course notebooks for undergraduate physics
and geoscience classes. Students and instructors can read, run, and inspect the
computations. Set up the workspace and open a numerical methods notebook with
`uv`:

```sh
mise install -y
uv sync --all-packages
uv run --package fisica-computacional marimo edit packages/fisica-computacional/metodos-numericos/pendulos-y-osciladores/pendulo_simple.py
```

The repository contains course packages for numerical physics, dynamical
systems, statistical mechanics, artificial intelligence, remote sensing, and
risk and disaster studies. The [notebook catalog](notebooks.toml) selects the
lessons that the static site publishes or links to as source.

The [architecture](architecture.md) describes the package, catalog, export, and
site boundaries. The [manual](docs/readme.md) covers notebook checks, optional
native dependencies, and local site builds. Contributors should read
[contributing.md](contributing.md) before changing a notebook or the site.
