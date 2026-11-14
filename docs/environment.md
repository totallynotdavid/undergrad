# Environment

The repository uses the tool versions in [`mise.toml`](../mise.toml): Bun, Node,
Python, and uv. The Python workspace requires Python 3.14. Install the tools,
then the locked Python workspace and site dependencies:

```sh
mise install -y
mise run install
```

The remote-sensing raster notebook uses rasterio. Its package includes the
runtime libraries needed to open the committed GeoTIFF, so no system raster
installation is required. Open the notebook with:

```sh
uv run --locked --package remote-sensing marimo edit packages/remote-sensing/landsat_rasterio.py
```

Some notebooks need external data or credentials. Their source files state the
required inputs. The current examples are the GaoFen-1 correction in
[`gaofen1/prepare.py`](../packages/remote-sensing/gaofen1/prepare.py) and the
Earth Engine notebook
[`earth_engine_peru.py`](../packages/remote-sensing/earth_engine_peru.py), which
reads `EE_PROJECT` from `packages/remote-sensing/.env`.
