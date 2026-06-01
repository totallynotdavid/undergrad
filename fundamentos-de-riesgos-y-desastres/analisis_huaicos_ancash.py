import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    GEOJSON_PATH = NOTEBOOK_DIR / "peru_provincial_simple.geojson"
    return GEOJSON_PATH, gpd, mo, pd, plt, requests


@app.cell
def _(mo):
    mo.md("""
    # Análisis de Huaicos en Ancash, Perú

    "
        "Este notebook consulta eventos de peligros desde SIGRID (CENEPRED), "
        "filtra la región de Áncash y resume:
    "
        "- conteos por tipo de evento,
    "
        "- provincias más afectadas,
    "
        "- distribución espacial sobre provincias.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Fuente de datos

    "
        "Consulta ArcGIS REST de SIGRID para peligros geodinámicos. "
        "El filtro espacial viene incluido en la URL.
    """)
    return


@app.cell
def _():
    sigrid_url = (
        "https://sigrid.cenepred.gob.pe/arcgis/rest/services/Cartografia_Peligros/"
        "MapServer/5020100/query?f=json&where=1%3D1&returnGeometry=true"
        "&spatialRel=esriSpatialRelIntersects"
        "&geometry=%7B%22rings%22%3A%5B%5B%5B-8989295.539643947%2C-1375988.8655829763%5D"
        "%2C%5B-8989295.539643947%2C-808520.3675939788%5D%2C%5B-8363123.403931949%2C"
        "-808520.3675939788%5D%2C%5B-8363123.403931949%2C-1375988.8655829763%5D%2C"
        "%5B-8989295.539643947%2C-1375988.8655829763%5D%5D%5D%2C%22spatialReference%22%3A"
        "%7B%22wkid%22%3A102100%2C%22latestWkid%22%3A3857%7D%7D"
        "&geometryType=esriGeometryPolygon&inSR=102100&outFields=*&outSR=102100"
    )
    return (sigrid_url,)


@app.cell
def _(pd, requests, sigrid_url):
    response = requests.get(sigrid_url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features", [])
    attributes = [feature.get("attributes", {}) for feature in features]
    raw_df = pd.DataFrame(attributes)
    return (raw_df,)


@app.cell
def _(mo, raw_df):
    mo.md(f"""
    Registros descargados: **{len(raw_df):,}**
    """)
    return


@app.cell
def _(raw_df):
    ancash_df = raw_df[raw_df["dpto"].astype(str).str.upper() == "ANCASH"].copy()
    ancash_df["prov"] = ancash_df["prov"].astype(str).str.upper()
    ancash_df["peligro_es"] = ancash_df["peligro_es"].astype(str)
    ancash_df = ancash_df.dropna(subset=["latitud", "longitud", "prov", "peligro_es"])
    ancash_df
    return (ancash_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Resumen por tipo de evento
    """)
    return


@app.cell
def _(ancash_df):
    event_distribution = (
        ancash_df["peligro_es"].value_counts().rename_axis("evento").reset_index(name="cantidad")
    )
    event_distribution
    return (event_distribution,)


@app.cell
def _(ancash_df):
    province_event_counts = (
        ancash_df.groupby(["peligro_es", "prov"]).size().reset_index(name="cantidad")
    )
    idx = province_event_counts.groupby("peligro_es")["cantidad"].idxmax()
    most_affected = province_event_counts.loc[idx].sort_values("peligro_es")
    most_affected.columns = ["evento", "provincia_mas_afectada", "cantidad"]
    most_affected
    return


@app.cell
def _(event_distribution, plt):
    fig_events, ax_events = plt.subplots(figsize=(10, 5))
    top = event_distribution.head(12)
    ax_events.bar(top["evento"], top["cantidad"])
    ax_events.set_title("Top 12 tipos de peligro en Áncash")
    ax_events.set_xlabel("Tipo de peligro")
    ax_events.set_ylabel("Número de eventos")
    ax_events.tick_params(axis="x", rotation=45)
    ax_events.grid(alpha=0.25)
    fig_events.tight_layout()
    fig_events
    return


@app.cell
def _(GEOJSON_PATH, gpd):
    peru_provinces = gpd.read_file(GEOJSON_PATH)
    return (peru_provinces,)


@app.cell
def _(ancash_df, peru_provinces):
    events_per_province = ancash_df["prov"].value_counts().rename_axis("prov").reset_index(name="num_eventos")
    merged = peru_provinces.merge(events_per_province, left_on="NOMBPROV", right_on="prov", how="left")
    merged["num_eventos"] = merged["num_eventos"].fillna(0)
    ancash_map = merged[merged["NOMBPROV"].isin(ancash_df["prov"].unique())].copy()
    return (ancash_map,)


@app.cell
def _(ancash_df, ancash_map, gpd, plt):
    fig_map, ax_map = plt.subplots(figsize=(11, 11))
    ancash_map.plot(
        column="num_eventos",
        ax=ax_map,
        cmap="OrRd",
        edgecolor="gray",
        legend=True,
        legend_kwds={"label": "Número de peligros por provincia", "orientation": "horizontal"},
    )

    points = gpd.GeoDataFrame(
        ancash_df,
        geometry=gpd.points_from_xy(ancash_df["longitud"], ancash_df["latitud"]),
        crs="EPSG:4326",
    )
    points.plot(ax=ax_map, marker="o", color="navy", markersize=6, alpha=0.6)

    for _, row in ancash_map.iterrows():
        if row["num_eventos"] > 0:
            centroid = row["geometry"].centroid
            ax_map.text(centroid.x, centroid.y, int(row["num_eventos"]), fontsize=9, ha="center", va="center")

    ax_map.set_title("Peligros registrados en provincias de Áncash")
    ax_map.set_axis_off()
    fig_map
    return


if __name__ == "__main__":
    app.run()
