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
installation is required. Start the lesson with:

```sh
uv run --locked --package tecnicas-de-teledeteccion marimo edit packages/tecnicas-de-teledeteccion/raster-y-gdal/clase_2_gdal.py
```

Some lessons need external data or credentials. Their source files state the
required inputs. The current examples are
[`main.py`](../packages/tecnicas-de-teledeteccion/correccion-atmosferica-gaofen1/main.py)
and the Earth Engine lesson in
[`clase_1_earth_engine.py`](../packages/tecnicas-de-teledeteccion/earth-engine-y-6s/clase_1_earth_engine.py).
