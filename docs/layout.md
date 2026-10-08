# Repository layout

The repository has one site boundary. `site/` contains the frontend and its
generated static inputs. Course material stays in `packages/` and enters the
site only through the explicit notebook export command. The SvelteKit build
does not scan the repository root or course packages directly.

## Top-level folders

| Folder           | Kind            | What belongs there                                                                                  |
| ---------------- | --------------- | --------------------------------------------------------------------------------------------------- |
| `site/`          | site            | SvelteKit source, frontend tests, static site assets, and ignored generated notebook exports        |
| `packages/`      | notebook source | Course packages, notebooks, supporting Python/Fortran/MATLAB source, data, and package dependencies |
| `scripts/`       | tooling         | Catalog validation, frontend data generation, notebook checks, and publication orchestration        |
| `.github/`       | tooling         | CI and GitHub Pages deployment workflows                                                            |
| `.devcontainer/` | tooling         | Reproducible development-container configuration                                                    |
| `docs/`          | other           | Contributor and operator documentation; it is not site content                                      |
| `.ruff_cache/`   | other           | Ignored Ruff analysis cache; never part of a build input                                            |

The table covers repository content and the generated cache that can appear at
the repository root. Git metadata, virtual environments, and agent credentials
are checkout-local infrastructure rather than repository folders and are not
part of this layout.

## Top-level files

| File                                              | Kind    | Purpose                                                                  |
| ------------------------------------------------- | ------- | ------------------------------------------------------------------------ |
| `notebooks.toml`                                  | tooling | Catalog source: selects the notebook lessons and their publication modes |
| `pyproject.toml`, `uv.lock`                       | tooling | Python workspace and locked dependency graph                             |
| `mise.toml`                                       | tooling | Pinned developer tools and repository tasks                              |
| `install.sh`                                      | tooling | Optional native dependency setup                                         |
| `architecture.md`, `contributing.md`, `readme.md` | other   | Project, contributor, and user documentation                             |

## Build boundary

The site build path is narrow:

```text
packages/ + notebooks.toml
        │
        ├─ scripts/notebooks_manifest.py ──> site/src/lib/catalog.generated.ts
        └─ scripts/export_notebooks.py ────> site/static/notebooks/ (generated)
                                                        │
site/src/ + site/static/ ──────────────────────────────┘
                              └─> site/build/ (generated deploy artifact)
```

`site/static/notebooks/`, `site/build/`, `site/.svelte-kit/`, and
`site/node_modules/` are ignored generated or installed content. Do not place
course source, research notes, native build output, or documentation under
`site/`. Add those files to `packages/` or the appropriate
tooling/documentation folder instead.
