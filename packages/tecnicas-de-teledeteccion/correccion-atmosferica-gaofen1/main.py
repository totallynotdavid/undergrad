import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo

    return dedent, mo


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            Este notebook calcula reflectancia de superficie con 6S para las bandas
            azul, verde, roja y NIR del producto GaoFen-1 PMS.

            La ruta predeterminada usa un clip GeoTIFF y valores de 6S locales.
            La escena es el producto GaoFen-1 del propietario, conservado en su
            carpeta compartida de Google Drive.
            Activa la opción live para descargar la escena completa y consultar
            Earth Engine.
            """
        )
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            La hora de adquisición está en hora estándar de China (UTC+8) y se convierte a UTC para 6S y Earth Engine. El resultado difiere del cuaderno del propietario de 2024: aquel dividía el DN entre la ganancia, usaba coeficientes de GF-1 en lugar de GF-1D y tomaba la hora de adquisición como UTC.
            """
        )
    )
    return


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from correction import (
        CLIP_PATH,
        INPUTS_PATH,
        live_data,
        load_clip,
        load_inputs,
        load_metadata,
        run_correction,
    )

    live_option = mo.ui.checkbox(
        label="Usar la escena completa de Drive y valores live de Earth Engine",
        value=False,
    )
    return (
        CLIP_PATH,
        INPUTS_PATH,
        live_data,
        load_clip,
        load_inputs,
        load_metadata,
        live_option,
        np,
        pd,
        plt,
        run_correction,
    )


@app.cell
def _(CLIP_PATH, INPUTS_PATH, live_option, mo):
    mo.vstack(
        [
            live_option,
            mo.md(
                f"Fuente offline: `{CLIP_PATH.name}` y `{INPUTS_PATH.name}`. "
                "El clip conserva la resolución completa de la escena."
            ),
        ]
    )
    return


@app.cell
def _(live_data, live_option, load_clip, load_inputs, load_metadata):
    if live_option.value:
        sensor, profile, metadata, inputs = live_data()
        source = "Drive + Earth Engine (live)"
    else:
        sensor, profile = load_clip()
        metadata = load_metadata()
        inputs = load_inputs()
        source = "clip y JSON comprometidos (offline)"
    return inputs, metadata, profile, sensor, source


@app.cell
def _(metadata, mo, profile, sensor, source):
    mo.md(
        f"**Fuente activa:** {source}. **Fecha:** {metadata['date'].date()}. "
        f"**Tamaño:** {sensor.shape[1]} × {sensor.shape[2]} píxeles. "
        f"**CRS:** `{profile['crs']}`."
    )
    return


@app.cell
def _(inputs, metadata, run_correction, sensor):
    Reflectancia = run_correction(sensor, metadata, inputs).astype("float32")
    return (Reflectancia,)


@app.cell
def _(Reflectancia, mo, np, pd):
    band_names = ["Azul", "Verde", "Roja", "NIR"]
    summary = pd.DataFrame(
        {
            "Banda": band_names,
            "Media": [float(np.nanmean(Reflectancia[i])) for i in range(4)],
            "Máximo": [float(np.nanmax(Reflectancia[i])) for i in range(4)],
            "Píxeles válidos": [
                f"{np.isfinite(Reflectancia[i]).mean():.1%}" for i in range(4)
            ],
        }
    )
    mo.ui.table(summary)
    return band_names, summary


@app.cell
def _(Reflectancia, mo, np, plt, sensor):
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.25), dpi=72)
    raw = np.moveaxis(sensor[[2, 1, 0]].astype(float), 0, -1)
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    corrected = np.clip(np.moveaxis(Reflectancia[[2, 1, 0]], 0, -1), 0, 0.5)
    axes[0].imshow(raw)
    axes[0].set_title("GaoFen-1 RGB")
    axes[1].imshow(corrected, vmin=0, vmax=0.5)
    axes[1].set_title("Reflectancia 6S")
    ndvi = (Reflectancia[3] - Reflectancia[2]) / (Reflectancia[3] + Reflectancia[2])
    axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title("NDVI")
    for axis in axes:
        axis.set_axis_off()
    fig.tight_layout()
    mo.as_html(fig)
    return (fig,)


if __name__ == "__main__":
    app.run()
