import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    TITANIC_PATH = Path(__file__).resolve().parent / "titanic3.csv"
    return TITANIC_PATH, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Clase 1: limpieza de datos tabulares

    Objetivo: construir un flujo básico de limpieza con un dataset real (`titanic`),
    identificando columnas redundantes y manejando valores faltantes.

    Los datos son la versión `titanic3` del Departamento de Bioestadística de la
    Universidad de Vanderbilt: los pasajeros del Titanic, sin la tripulación,
    reunidos por Thomas Cason a partir de la Encyclopedia Titanica. Datos
    obtenidos de [hbiostat.org/data](https://hbiostat.org/data/), cortesía del
    Departamento de Bioestadística de la Universidad de Vanderbilt. Se leen del
    archivo `titanic3.csv`, junto a este cuaderno, sin descargar nada.
    """)
    return


@app.cell
def _(TITANIC_PATH, pd):
    df_raw = pd.read_csv(TITANIC_PATH)
    return (df_raw,)


@app.cell
def _(df_raw, mo):
    mo.md(f"""
    Registros: **{len(df_raw)}** | Columnas: **{df_raw.shape[1]}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1) Vista inicial del dataset
    """)
    return


@app.cell
def _(df_raw):
    df_raw.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2) Valores faltantes por columna
    """)
    return


@app.cell
def _(df_raw):
    df_raw.isna().sum().sort_values(ascending=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3) Selección de columnas

    En esta sesión retiramos las columnas que no sirven para el ejercicio:

    - `name`, `ticket` y `home.dest` identifican al pasajero y no describen su
      viaje de forma comparable.
    - `cabin` falta en la mayoría de los pasajeros.
    - `boat` y `body` registran lo que ocurrió después del naufragio: el bote
      salvavidas en el que subió un sobreviviente y el número del cuerpo
      recuperado. Revelan el resultado que se quiere predecir, así que un modelo
      no debe verlas.
    """)
    return


@app.cell
def _(df_raw):
    columns_to_drop = ["name", "ticket", "home.dest", "cabin", "boat", "body"]
    df_reduced = df_raw.drop(columns=columns_to_drop)
    df_reduced.head(10)
    return (df_reduced,)


@app.cell
def _(mo):
    mo.md("""
    ## 4) Dataset sin nulos
    """)
    return


@app.cell
def _(df_reduced):
    df_clean = df_reduced.dropna().reset_index(drop=True)
    df_clean.head(10)
    return (df_clean,)


@app.cell
def _(df_clean, df_raw, mo):
    kept = len(df_clean)
    total = len(df_raw)
    pct = 100 * kept / total if total else 0
    mo.md(f"Se conservan **{kept} / {total}** filas ({pct:.1f}%).")
    return


@app.cell
def _(df_clean):
    df_clean.isna().sum()
    return


if __name__ == "__main__":
    app.run()
