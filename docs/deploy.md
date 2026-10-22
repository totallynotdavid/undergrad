# Deployment

GitHub Pages builds run on `ubuntu-latest` (Ubuntu 24.04). The workflow installs
the repository's pinned tools and Python dependencies, checks and exports the
published notebooks, and then builds `site/build`.

To reproduce the build on an Ubuntu 24.04 maintainer host, run:

```sh
mise install -y
mise run install
mise run check
```

The raster notebook reads a committed GeoTIFF through rasterio. The build does
not require a system raster package or an Earth Engine login.
