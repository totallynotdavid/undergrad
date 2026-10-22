import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import json
    import textwrap
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from ingemmet_peligros.inventario import download

    plt.rcParams["figure.dpi"] = 72
    DATOS = Path(__file__).resolve().parent / "datos"
    return DATOS, download, gpd, json, mo, pd, plt, textwrap


@app.cell
def _(mo):
    mo.md("""
    # Movimientos en masa en Áncash

    Este cuaderno usa el inventario nacional de peligros geológicos de INGEMMET
    para responder tres preguntas sobre Áncash:

    - qué movimientos en masa se han registrado y con qué frecuencia,
    - qué provincias concentran más registros,
    - qué factores condicionan los huaicos, es decir, los flujos de detritos y
      de lodo.

    Los datos vienen de una copia guardada junto al cuaderno, así que no
    necesita red.
    """)
    return


@app.cell
def _(mo):
    usar_servicio = mo.ui.checkbox(
        value=False,
        label="Consultar el servicio de INGEMMET en vez de la copia guardada",
    )
    mo.md(f"""
    ## Fuente de datos

    Capa `SERV_PELIGROS_GEOLOGICOS/MapServer/0` de GeoCatMin (INGEMMET),
    filtrada con `NM_DPTO='Ancash'`. La copia guardada está descrita en
    `datos/NOTA.md`. Marque la casilla para descargar los datos actuales; hace
    falta red y la descarga tarda unos segundos.

    {usar_servicio}
    """)
    return (usar_servicio,)


@app.cell
def _(DATOS, download, gpd, json, pd, usar_servicio):
    if usar_servicio.value:
        _geojson, dominios = download()
        puntos = gpd.GeoDataFrame.from_features(_geojson["features"], crs="EPSG:4326")
        origen = "servicio de INGEMMET"
    else:
        puntos = gpd.read_file(DATOS / "peligros_ancash.geojson")
        dominios = json.loads((DATOS / "dominios.json").read_text(encoding="utf-8"))
        origen = "copia guardada"
    puntos["FECHA"] = pd.to_datetime(puntos["FECHA"], utc=True)
    return dominios, origen, puntos


@app.cell
def _(dominios, origen, mo, puntos):
    mo.md(f"""
    Origen: **{origen}**. Registros: **{len(puntos):,}**, todos puntos en WGS84
    con {len(puntos.columns) - 1} atributos.

    Los atributos codificados (`TIP_PELIGRO`, `NOM_PELIGRO`, `FACT_SITIO`,
    `FACT_GEOGR`, `FACT_ANTRO`) se traducen con los dominios de la capa, que
    están en `datos/dominios.json`. Un código que la capa usa pero que su
    dominio no define se muestra tal cual, marcado como *sin descripción en el
    dominio*, porque no hay base para inventarle un nombre.
    """)
    return


@app.cell
def _(dominios, pd, puntos):
    def decodificar(codigo, tabla):
        return dominios[tabla].get(codigo, f"{codigo} (sin descripción en el dominio)")

    def factores(serie, tabla):
        """Devuelve una fila por código separado por coma."""
        codigos = serie.dropna().str.split(",").explode().str.strip()
        return codigos.map(lambda c: decodificar(c, tabla))

    # Solo los códigos 1504 y 1505 representan huaicos. El aluvión y las
    # avalanchas tienen otros códigos y no se cuentan.
    HUAICO = ["1000|1500|1504", "1000|1500|1505"]

    datos = pd.DataFrame(
        {
            "tipo": puntos["TIP_PELIGRO"].map(lambda c: decodificar(c, "TIP_PELIGRO")),
            "subtipo": puntos["NOM_PELIGRO"].map(
                lambda c: "Sin subtipo" if pd.isna(c) else decodificar(c, "NOM_PELIGRO")
            ),
            "provincia": puntos["NM_PROV"],
            "anio": puntos["FECHA"].dt.year,
            "huaico": puntos["NOM_PELIGRO"].isin(HUAICO),
        }
    )
    return datos, decodificar, factores


@app.cell
def _(mo):
    mo.md("""
    ## Tipos de movimiento en masa

    `COD_PELIGRO` vale `1000` (movimientos en masa) en todos los registros de
    Áncash de esta capa. `TIP_PELIGRO` separa los tipos y `NOM_PELIGRO` los
    subtipos.
    """)
    return


@app.cell
def _(datos, plt):
    por_tipo = (
        datos["tipo"].value_counts().rename_axis("tipo").reset_index(name="registros")
    )
    por_tipo["porcentaje"] = (por_tipo["registros"] / len(datos) * 100).round(1)

    fig_tipos, ax_tipos = plt.subplots(figsize=(8, 3.5))
    ax_tipos.barh(por_tipo["tipo"][::-1], por_tipo["registros"][::-1])
    ax_tipos.set_xlabel("Registros")
    ax_tipos.set_title("Movimientos en masa en Áncash por tipo")
    ax_tipos.grid(axis="x", alpha=0.25)
    fig_tipos.tight_layout()
    por_tipo
    return (fig_tipos,)


@app.cell
def _(fig_tipos):
    fig_tipos
    return


@app.cell
def _(datos, plt, textwrap):
    por_subtipo = (
        datos["subtipo"]
        .value_counts()
        .rename_axis("subtipo")
        .reset_index(name="registros")
    )
    _top = por_subtipo.head(12)

    fig_subtipos, ax_subtipos = plt.subplots(figsize=(8, 5))
    ax_subtipos.barh(
        [textwrap.fill(s, 38) for s in _top["subtipo"]][::-1],
        _top["registros"][::-1],
    )
    ax_subtipos.set_xlabel("Registros")
    ax_subtipos.set_title("Los 12 subtipos más registrados")
    ax_subtipos.grid(axis="x", alpha=0.25)
    fig_subtipos.tight_layout()
    fig_subtipos
    return (por_subtipo,)


@app.cell
def _(mo, por_subtipo):
    _sin_dominio = por_subtipo[por_subtipo["subtipo"].str.contains("sin descripción")][
        "registros"
    ].sum()
    _sin_subtipo = por_subtipo.loc[
        por_subtipo["subtipo"] == "Sin subtipo", "registros"
    ].sum()
    _nombrados = por_subtipo[~por_subtipo["subtipo"].str.contains("sin descripción")]
    _mayor = _nombrados[_nombrados["subtipo"] != "Sin subtipo"]["subtipo"].iloc[0]
    mo.md(f"""
    El subtipo con nombre más registrado es *{_mayor.lower()}*.
    {_sin_dominio} registros llevan un código de subtipo que el dominio de la
    capa no define, y {_sin_subtipo} no tienen subtipo.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Provincias más afectadas

    El conteo usa la provincia que figura en cada registro (`NM_PROV`). Los
    polígonos provinciales simplificados solo se usan para dibujar el mapa.
    """)
    return


@app.cell
def _(datos, pd):
    por_provincia = (
        datos.groupby("provincia")
        .agg(registros=("tipo", "size"), huaicos=("huaico", "sum"))
        .sort_values("registros", ascending=False)
        .reset_index()
    )
    por_provincia["porcentaje_huaicos"] = (
        por_provincia["huaicos"] / por_provincia["registros"] * 100
    ).round(1)
    _principal = (
        datos.groupby(["provincia", "tipo"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
        .drop_duplicates("provincia")
        .set_index("provincia")["tipo"]
    )
    por_provincia["tipo_principal"] = por_provincia["provincia"].map(_principal)
    pd.set_option("display.max_rows", 30)
    por_provincia
    return (por_provincia,)


@app.cell
def _(plt, por_provincia):
    fig_prov, ax_prov = plt.subplots(figsize=(8, 5.5))
    _p = por_provincia[::-1]
    ax_prov.barh(_p["provincia"], _p["registros"], label="Otros movimientos")
    ax_prov.barh(_p["provincia"], _p["huaicos"], label="Huaicos", color="tab:red")
    ax_prov.set_xlabel("Registros")
    ax_prov.set_title("Registros por provincia")
    ax_prov.legend(loc="lower right")
    ax_prov.grid(axis="x", alpha=0.25)
    fig_prov.tight_layout()
    fig_prov
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mapa sobre las provincias
    """)
    return


@app.cell
def _(DATOS, gpd, por_provincia, puntos):
    _provincias = gpd.read_file(DATOS / "peru_provincial_simple.geojson")
    _provincias["NOMBPROV"] = _provincias["NOMBPROV"].str.upper()
    _conteo = por_provincia.assign(NOMBPROV=por_provincia["provincia"].str.upper())
    mapa = _provincias.merge(_conteo, on="NOMBPROV", how="inner")
    # Los polígonos simplificados requieren un margen de 0.1° para incluir
    # puntos que caen justo fuera de sus límites.
    dentro = puntos.within(mapa.union_all().buffer(0.1))
    return dentro, mapa


@app.cell
def _(datos, dentro, mapa, plt, puntos):
    _en_mapa = puntos[dentro]
    _huaicos = _en_mapa[datos.loc[dentro, "huaico"]]
    fig_mapa, ax_mapa = plt.subplots(figsize=(8, 8))
    mapa.plot(
        column="registros",
        ax=ax_mapa,
        cmap="Greys",
        edgecolor="gray",
        legend=True,
        legend_kwds={"label": "Registros por provincia", "shrink": 0.6},
    )
    _en_mapa.drop(_huaicos.index).plot(
        ax=ax_mapa, color="tab:blue", markersize=4, alpha=0.7, label="Otros movimientos"
    )
    _huaicos.plot(ax=ax_mapa, color="tab:red", markersize=6, alpha=0.8, label="Huaicos")
    for _, _fila in mapa.iterrows():
        _c = _fila["geometry"].representative_point()
        ax_mapa.text(
            _c.x,
            _c.y,
            f"{_fila['provincia']}\n{int(_fila['registros'])}",
            fontsize=6,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 0.8, "lw": 0},
        )
    ax_mapa.legend(loc="lower left", markerscale=2)
    ax_mapa.set_title("Registros por provincia y ubicación de cada punto")
    ax_mapa.set_axis_off()
    fig_mapa.tight_layout()
    fig_mapa
    return


@app.cell
def _(dentro, mo, puntos):
    _fuera = puntos[~dentro]
    mo.md(f"""
    {len(_fuera)} registro(s) tienen coordenadas fuera de Áncash y no se
    dibujan: {", ".join(_fuera["COD_INV"] + " (" + _fuera["NM_PROV"] + ")")}. Su
    provincia sí cuenta en las tablas.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Qué condiciona los huaicos

    Cada registro puede citar varios factores. Los gráficos muestran qué
    porcentaje de los huaicos cita cada factor, junto al porcentaje de los
    demás movimientos en masa, para ver cuáles distinguen a los huaicos.

    - `FACT_SITIO`: condiciones del terreno (roca, suelo, pendiente).
    - `FACT_GEOGR`: entorno geográfico, donde están las lluvias y otros
      desencadenantes.
    - `FACT_ANTRO`: intervención humana.
    """)
    return


@app.cell
def _(datos, factores, pd, puntos):
    def tabla_factores(campo, dominio):
        grupos = {"Huaicos": datos["huaico"], "Otros movimientos": ~datos["huaico"]}
        columnas = {}
        for _nombre, _mascara in grupos.items():
            _citados = factores(puntos.loc[_mascara, campo], dominio)
            _citados = _citados[~_citados.reset_index().duplicated().to_numpy()]
            columnas[_nombre] = _citados.value_counts() / _mascara.sum() * 100
        _tabla = pd.DataFrame(columnas).fillna(0).round(1)
        return _tabla.sort_values("Huaicos", ascending=False)

    factores_sitio = tabla_factores("FACT_SITIO", "FACT_SITIO")
    factores_geogr = tabla_factores("FACT_GEOGR", "FACT_GEOGR")
    factores_antro = tabla_factores("FACT_ANTRO", "FACT_ANTRO")
    return factores_antro, factores_geogr, factores_sitio


@app.cell
def _(plt, textwrap):
    def grafico_factores(tabla, titulo):
        _t = tabla[::-1]
        _y = range(len(_t))
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(_t) + 1.5))
        ax.barh(
            [i + 0.2 for i in _y],
            _t["Huaicos"],
            height=0.4,
            label="Huaicos",
            color="tab:red",
        )
        ax.barh(
            [i - 0.2 for i in _y],
            _t["Otros movimientos"],
            height=0.4,
            label="Otros movimientos",
            color="tab:gray",
        )
        ax.set_yticks(list(_y), [textwrap.fill(n, 42) for n in _t.index], fontsize=8)
        ax.set_xlabel("% de los registros del grupo que citan el factor")
        ax.set_title(titulo)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        return fig

    return (grafico_factores,)


@app.cell
def _(factores_sitio, grafico_factores):
    grafico_factores(factores_sitio, "Factores del sitio (FACT_SITIO)")
    return


@app.cell
def _(factores_geogr, grafico_factores):
    grafico_factores(
        factores_geogr, "Entorno geográfico y desencadenantes (FACT_GEOGR)"
    )
    return


@app.cell
def _(factores_antro):
    # Pocos registros citan factores antrópicos: una tabla basta.
    factores_antro.rename_axis("Factor antrópico (FACT_ANTRO), % de cada grupo")


@app.cell
def _(datos, factores_geogr, factores_sitio, mo, puntos):
    _h = puntos[datos["huaico"]]
    _n = len(_h)
    _sin_antro = int(_h["FACT_ANTRO"].isna().sum())

    def _mayor_brecha(tabla):
        _brecha = (tabla["Huaicos"] - tabla["Otros movimientos"]).drop(
            [i for i in tabla.index if "sin descripción" in i]
        )
        return _brecha.idxmax(), _brecha.max()

    _sitio, _b_sitio = _mayor_brecha(factores_sitio)
    _geogr, _b_geogr = _mayor_brecha(factores_geogr)
    _lluvias = ["Lluvias intensas", "Lluvias prolongadas"]
    _lluvia = factores_geogr.reindex(_lluvias)["Huaicos"].fillna(0)
    _top = ", ".join(f"*{n.lower()}*" for n in factores_sitio.index[:3])
    mo.md(f"""
    Lectura de los gráficos, con {_n} huaicos:

    - Los factores del sitio que más citan los huaicos son {_top}.
    - El factor del sitio con más ventaja para los huaicos sobre los demás
      movimientos es *{_sitio.lower()}*, con {_b_sitio:.0f} puntos porcentuales.
      En el entorno geográfico es *{_geogr.lower()}*, con {_b_geogr:.0f} puntos.
    - Las lluvias intensas figuran en el {_lluvia[_lluvias[0]]:.0f} % de los
      huaicos y las prolongadas en el {_lluvia[_lluvias[1]]:.0f} %. Es poco
      para un flujo que se asocia a lluvias. El código `11` de `FACT_GEOGR` es
      el más frecuente y el dominio de la capa no lo define, así que no se
      puede leer como lluvia ni como ningún otro factor.
    - {_sin_antro} de los {_n} huaicos no registran ningún factor antrópico, así
      que ese campo describe pocos casos y no permite comparar grupos.
    """)
    return


@app.cell
def _(datos, mo, puntos):
    _por_anio = datos["anio"].value_counts()
    _anio, _n = _por_anio.index[0], _por_anio.iloc[0]
    _estudio = puntos.loc[datos["anio"] == _anio, "TIESTU_ANO"].mode().iloc[0]
    mo.md(f"""
    ## Cómo leer estos conteos

    Es un inventario de campo, no un catálogo de eventos. **{_n:,}** de los
    {len(datos):,} registros ({_n / len(datos):.0%}) tienen fecha de {int(_anio)}
    y el estudio más frecuente entre ellos es «{_estudio}». Las
    diferencias entre provincias reflejan en parte dónde se hizo el trabajo de
    campo, además de dónde ocurren los movimientos. La capa no da el volumen ni
    los daños de cada evento, así que un conteo mayor no significa mayor riesgo.
    """)
    return


if __name__ == "__main__":
    app.run()
