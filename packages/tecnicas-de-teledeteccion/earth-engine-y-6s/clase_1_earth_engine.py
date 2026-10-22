import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Earth Engine en Peru: costa, Andes y Amazonia

    Esta clase compara Lima, Cusco e Iquitos con SRTM, MODIS MCD12Q1 y MODIS
    MOD11A1. La tabla y la serie vienen por defecto de
    `peru_earth_engine.json`, un resultado versionado para que la pagina se
    pueda exportar sin credenciales ni red.

    Para repetir las consultas, configura `EE_PROJECT` en el `.env` del
    paquete, marca la casilla de consulta en vivo y ejecuta el notebook con la
    extra de Earth Engine. El script de preparación aplica timeout y reintentos
    a las llamadas de Earth Engine.
    """)
    return


@app.cell
def _():
    import json
    import os
    import runpy
    from pathlib import Path

    import matplotlib.pyplot as plt

    return Path, json, os, plt, runpy


@app.cell
def _(mo):
    live_query = mo.ui.checkbox(
        label="Ejecutar consultas Earth Engine en vivo (requiere credenciales)",
        value=False,
    )
    return (live_query,)


@app.cell
def _(Path, json, live_query, os, runpy):
    notebook_dir = Path(__file__).resolve().parent
    fixture_path = notebook_dir / "peru_earth_engine.json"
    run_live = live_query.value or os.environ.get("TELEDETECCION_RUN_LIVE") == "1"
    if run_live:
        prepare = runpy.run_path(str(notebook_dir / "prepare_earth_engine.py"))
        lesson_data = prepare["query_live_data"]()
        data_source = "consulta Earth Engine en vivo"
    else:
        lesson_data = json.loads(fixture_path.read_text(encoding="utf-8"))
        data_source = f"fixture versionado: `{fixture_path.name}`"
    return data_source, lesson_data, run_live


@app.cell
def _(data_source, lesson_data, mo, run_live):
    import pandas as pd

    places = pd.DataFrame(lesson_data["places"])[
        ["name", "description", "longitude", "latitude"]
    ].rename(
        columns={
            "name": "Lugar",
            "description": "Contraste",
            "longitude": "Longitud",
            "latitude": "Latitud",
        }
    )
    elevation = pd.DataFrame(lesson_data["elevation"])[
        ["place", "elevation_m", "date", "dataset_id"]
    ].rename(
        columns={
            "place": "Lugar",
            "elevation_m": "Elevacion (m)",
            "date": "Fecha",
            "dataset_id": "Dataset",
        }
    )
    land_cover = pd.DataFrame(lesson_data["land_cover"])[
        ["place", "class_code", "class_name", "date", "dataset_id"]
    ].rename(
        columns={
            "place": "Lugar",
            "class_code": "Codigo IGBP",
            "class_name": "Clase de cobertura",
            "date": "Fecha",
            "dataset_id": "Dataset",
        }
    )
    lst = pd.DataFrame(lesson_data["land_surface_temperature"])
    lst["fecha"] = pd.to_datetime(lst["date"])
    source_label = "en vivo" if run_live else "offline"
    return (
        elevation,
        land_cover,
        lst,
        mo.md(
            f"Datos {source_label}; fuente: {data_source}. "
            "LST es la media en 5 km del compuesto de mediana mensual de los "
            "pixeles que pasan el filtro QC_Day. Lima no tiene pixeles validos "
            "en junio-agosto por la niebla costera."
        ),
        places,
    )


@app.cell
def _(elevation, land_cover, mo, places):
    mo.vstack(
        [
            mo.md("### Coordenadas de los contrastes"),
            places,
            mo.md("### Elevacion SRTM"),
            elevation,
            mo.md("### Cobertura MODIS MCD12Q1 (LC_Type1)"),
            land_cover,
        ]
    )
    return


@app.cell
def _(lst, mo, plt):
    figure, axis = plt.subplots(figsize=(7, 3.5), dpi=90)
    for place, values in lst.groupby("place", sort=False):
        axis.plot(values["fecha"], values["lst_celsius"], marker="o", label=place)
    axis.set_title("MODIS MOD11A1: temperatura superficial diurna")
    axis.set_ylabel("Temperatura (°C)")
    axis.set_xlabel("Mes de 2025")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    mo.vstack(
        [
            mo.md(
                "### Serie mensual con QC aplicado\n\n"
                "Se conservan solo `QC_Day` con bits 0-1 = 00 y bit 2 = 0; "
                "la escala es 0.02 K y se convierte a °C."
            ),
            figure,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
