import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    return Path, mo, np, plt, rasterio


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent
    filepath = notebook_dir / "landsat_scene_007068_20240427_reflectance.tif"
    return filepath


@app.cell
def _(mo):
    mo.md(r"""
    # Lectura y visualización de un raster con rasterio

    Rasterio representa el raster como un dataset con una o más bandas. En
    esta práctica abrimos una imagen Landsat, consultamos su metadata, leemos
    la primera banda como un arreglo de NumPy y enmascaramos sus valores
    NoData.
    """)
    return


@app.cell
def _(filepath, np, rasterio):
    metadata = None
    masked_array = None

    if filepath.exists():
        with rasterio.open(filepath) as dataset:
            band_array = dataset.read(1)
            nodata = dataset.nodata
            if nodata is None:
                masked_array = np.ma.masked_invalid(band_array)
            else:
                masked_array = np.ma.masked_equal(band_array, nodata)
            metadata = {
                "count": dataset.count,
                "crs": dataset.crs,
                "descriptions": dataset.descriptions,
                "dtype": dataset.dtypes[0],
                "height": dataset.height,
                "nodata": nodata,
                "raster_tags": dataset.tags(),
                "band_tags": dataset.tags(1),
                "transform": dataset.transform,
                "width": dataset.width,
            }

    return masked_array, metadata


@app.cell
def _(filepath, metadata, mo):
    metadata_output = None
    if metadata is None:
        metadata_output = mo.md(
            f"No se encontró un raster de entrada en `{filepath}`. "
            "Coloca allí el archivo Landsat para ejecutar el análisis."
        )
    else:
        metadata_output = mo.md(
            f"""
            **Dataset:** `{filepath.name}`

            - Dimensiones: {metadata["width"]} columnas por
              {metadata["height"]} filas.
            - Bandas: {metadata["count"]}.
            - Tipo de dato de la banda 1: `{metadata["dtype"]}`.
            - Valor NoData: `{metadata["nodata"]}`.
            - CRS: `{metadata["crs"]}`.
            - Geotransform: `{metadata["transform"]}`.
            - Nombres de bandas: `{", ".join(metadata["descriptions"])}`.
            - Entradas de metadata del dataset: {len(metadata["raster_tags"])}.
            - Entradas de metadata de la banda: {len(metadata["band_tags"])}.
            """
        )
    metadata_output
    return metadata_output


@app.cell
def _(masked_array, mo, plt):
    figure_output = None
    if masked_array is None:
        figure_output = mo.md(
            "La figura se generará cuando el raster de entrada esté disponible."
        )
    else:
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(masked_array, cmap="gray")
        axis.set_title("Landsat 8: banda B1 (DN de reflectancia)")
        axis.set_xlabel("Columna")
        axis.set_ylabel("Fila")
        figure.colorbar(image, ax=axis, label="Reflectancia")
        figure_output = figure
    figure_output
    return figure_output


if __name__ == "__main__":
    app.run()
