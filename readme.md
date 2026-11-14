# Notebooks

Notebooks is a collection of computational physics notebooks, written in
Spanish. Each one works through a physics or geoscience problem in Python, and
readers can run and inspect every computation. Set up the workspace and open the
simple pendulum notebook:

```sh
mise install -y
mise run install
uv run --package dynamics marimo edit packages/dynamics/pendulo_simple.py
```

The notebooks are grouped by topic. Each topic is a package under `packages/`:
numerical methods, dynamics and differential equations, statistical and
molecular physics, radiation and remote sensing, and solid Earth and hazards.
The [notebook catalog](notebooks.toml) lists the notebooks that the static site
publishes. A published notebook is a read-only page of its results. The reader's
browser never runs Python.

The [architecture](docs/architecture.md) describes the package, catalog, export,
and site boundaries. The [manual](docs/readme.md) covers notebook checks,
optional native dependencies, and local site builds. Contributors should read
[contributing.md](.github/contributing.md) before changing a notebook or the
site.

The [repository layout](docs/layout.md) maps each top-level folder to its site,
notebook-source, tooling, or documentation boundary.

The code written for this repository is under the [MIT license](LICENSE).
