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

The Fortran notebooks use a native compiler. On Debian or Ubuntu,
[`install.sh`](../install.sh) installs `gfortran-13`:

```sh
./install.sh
```

The GDAL raster notebook has an optional package dependency named `gdal`. The
lockfile selects GDAL Python binding 3.13.3. The native GDAL library must be
version 3.13.3 or newer, with matching development headers and `gdal-config`.
Ubuntu 26.04 currently ships libgdal 3.12.2 through apt, which is too old for
the locked binding and fails during the build. On Debian or Ubuntu, install the
native packages with:

```sh
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
```

On macOS, install GDAL with Homebrew:

```sh
brew install gdal
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
