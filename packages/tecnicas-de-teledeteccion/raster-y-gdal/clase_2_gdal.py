import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from osgeo import gdal, gdal_array

    return Path, gdal, gdal_array, mo, np, plt


@app.cell
def _(Path):
    notebook_dir = Path(__file__).resolve().parent
    filepath = notebook_dir / "LC08_L1TP_007068_20240427_20240427_02_RT_refl.tif"
    output_path = notebook_dir / "public" / "img_4.png"
    return filepath, notebook_dir, output_path


@app.cell
def _(mo):
    mo.md(r"""
    # Lectura y visualización de un raster con GDAL

    GDAL representa el raster como un dataset con una o más bandas. En esta
    práctica abrimos una imagen Landsat, consultamos su metadata, leemos la
    primera banda como un arreglo de NumPy y enmascaramos sus valores NoData.
    """)
    return


@app.cell
def _(filepath, gdal, gdal_array, np):
    dataset = None
    band = None
    band_array = None
    masked_array = None

    if filepath.exists():
        dataset = gdal.Open(str(filepath))
        if dataset is not None:
            band = dataset.GetRasterBand(1)
            raster_array = gdal_array.LoadFile(str(filepath))
            band_array = raster_array[0] if raster_array.ndim == 3 else raster_array
            nodata = band.GetNoDataValue()
            if nodata is None:
                masked_array = np.ma.masked_invalid(band_array)
            else:
                masked_array = np.ma.masked_equal(band_array, nodata)

    return band, band_array, dataset, masked_array


@app.cell
def _(band, dataset, filepath, gdal, mo):
    metadata_output = None
    if dataset is None or band is None:
        metadata_output = mo.md(
            f"No se encontró un raster de entrada en `{filepath}`. "
            "Coloca allí el archivo Landsat para ejecutar el análisis."
        )
    else:
        if band.GetMinimum() is None or band.GetMaximum() is None:
            band.ComputeStatistics(False)

        metadata = dataset.GetMetadata()
        band_metadata = band.GetMetadata()
        data_type = gdal.GetDataTypeName(band.DataType)
        metadata_output = mo.md(
            f"""
            **Dataset:** `{filepath.name}`

            - Dimensiones: {dataset.RasterXSize} columnas por
              {dataset.RasterYSize} filas.
            - Bandas: {dataset.RasterCount}.
            - Tipo de dato de la banda 1: `{data_type}`.
            - Valor NoData: `{band.GetNoDataValue()}`.
            - Mínimo y máximo: `{band.GetMinimum()}`, `{band.GetMaximum()}`.
            - Entradas de metadata del dataset: {len(metadata)}.
            - Entradas de metadata de la banda: {len(band_metadata)}.
            """
        )
    metadata_output
    return metadata_output


@app.cell
def _(masked_array, mo, output_path, plt):
    figure_output = None
    if masked_array is None:
        figure_output = mo.md(
            "La figura se generará cuando el raster de entrada esté disponible."
        )
    else:
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(masked_array, cmap="gray")
        axis.set_title("Banda 1")
        axis.set_xlabel("Columna")
        axis.set_ylabel("Fila")
        figure.colorbar(image, ax=axis, label="Reflectancia")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        figure_output = mo.vstack(
            [mo.md(f"La figura se guardó en `{output_path}`."), figure]
        )
    figure_output
    return figure_output


if __name__ == "__main__":
    app.run()
