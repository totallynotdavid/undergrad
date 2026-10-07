# Environment

The repository uses the tool versions in [`mise.toml`](../mise.toml): Bun,
Python, Ruff, and uv. The Python workspace requires Python 3.14. Install the
tools and synchronize all course packages with:

```sh
mise install -y
uv sync --all-packages
```

Install the site dependencies separately:

```sh
bun install --cwd site --frozen-lockfile
```

The Fortran notebooks use a native compiler, and the GDAL raster notebook needs
the native GDAL library and development headers. On Debian or Ubuntu,
[`install.sh`](../install.sh) installs `gfortran-13`, `gdal-bin`, and
`libgdal-dev`:

```sh
./install.sh
```

The GDAL raster notebook has an optional package dependency named `gdal`. The
project pins the Python binding and lockfile to 3.12.2 because Ubuntu 26.04
ships libgdal 3.12.2 through apt. The native library, development headers, and
`gdal-config` must report the same 3.12.2 version. On Debian or Ubuntu, install
the native packages with:

```sh
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
```

Homebrew currently has no `gdal@3.12` or `gdal@3.12.2` formula.
`brew info gdal@3.12` should report that no formula is available, and the
unversioned formula currently provides 3.13.3, which is newer than this
project's pin. Build the pinned release locally instead:

```sh
brew install cmake
curl -LO https://github.com/OSGeo/gdal/releases/download/v3.12.2/gdal-3.12.2.tar.gz
tar xf gdal-3.12.2.tar.gz
cmake -S gdal-3.12.2 -B gdal-3.12.2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/gdal-3.12.2"
cmake --build gdal-3.12.2/build --parallel
cmake --install gdal-3.12.2/build
export GDAL_CONFIG="$HOME/.local/gdal-3.12.2/bin/gdal-config"
export PATH="$(dirname "$GDAL_CONFIG"):$PATH"
```

The Homebrew formula name and the installed native version can be checked with:

```sh
brew info gdal@3.12
gdal-config --version
# Must print: 3.12.2
```

After the native prerequisites are available, start the lesson with the package
extra:

```sh
uv run --locked --package tecnicas-de-teledeteccion --extra gdal marimo edit packages/tecnicas-de-teledeteccion/raster-y-gdal/clase_2_gdal.py
```

Some lessons need external data or credentials. Their source files state the
required inputs. The current examples are
[`main.py`](../packages/tecnicas-de-teledeteccion/correccion-atmosferica-gaofen1/main.py),
[`analisis_huaicos_ancash.py`](../packages/fundamentos-de-riesgos-y-desastres/sigrid/analisis_huaicos_ancash.py),
and the Earth Engine lesson in
[`clase_1_earth_engine.py`](../packages/tecnicas-de-teledeteccion/earth-engine-y-6s/clase_1_earth_engine.py).
