import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns

    return mo, pd, sns


@app.cell
def _(mo):
    mo.md("""
    # Clase 1: limpieza de datos tabulares

    Objetivo: construir un flujo básico de limpieza con un dataset real (`titanic`),
    identificando columnas redundantes y manejando valores faltantes.
    """)
    return


@app.cell
def _(sns):
    titanic = sns.load_dataset("titanic")
    return (titanic,)


@app.cell
def _(mo, titanic):
    mo.md(f"""
    Registros: **{len(titanic)}** | Columnas: **{titanic.shape[1]}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1) Vista inicial del dataset
    """)
    return


@app.cell
def _(pd, titanic):
    df_raw = pd.DataFrame(titanic)
    df_raw.head(10)
    return (df_raw,)


@app.cell
def _(df_raw, mo):
    missing_counts = df_raw.isna().sum().sort_values(ascending=False)
    mo.md("## 2) Valores faltantes por columna")
    missing_counts
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3) Selección de columnas

    En esta sesión retiramos columnas redundantes o poco útiles para el ejercicio:
    `class`, `who`, `adult_male`, `deck`, `alive`, `embarked`.
    """)
    return


@app.cell
def _(df_raw):
    columns_to_drop = ["class", "who", "adult_male", "deck", "alive", "embarked"]
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
