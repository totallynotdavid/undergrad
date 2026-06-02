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
    Documentación de earthengine-api: https://developers.google.com/earth-engine/guides/python_install

    1. Instalar [gcloud](https://cloud.google.com/sdk/docs/install?hl=es-419#deb):
        ```
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        sudo apt-get update && sudo apt-get install google-cloud-cli
        ```
    2. Configura gcloud:
        ```
        gcloud init
        ```
    3. Obtén tu `EE_PROJECT`:
        - Lista proyectos: `gcloud projects list`
        - o crea uno: `gcloud projects create <project-id>`
        - registra acceso a Earth Engine: https://code.earthengine.google.com/register
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import ee
    from dotenv import load_dotenv
    return Path, ee, load_dotenv, os


@app.cell
def _(Path, load_dotenv, os):
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    project = os.environ.get("EE_PROJECT", "").strip()
    return env_path, project


@app.cell
def _(mo):
    project_input = mo.ui.text(
        label="EE project",
        value="",
    )
    project_form = mo.ui.form(
        project_input,
        submit_button_label="Guardar en .env",
        clear_on_submit=False,
    )
    return project_form, project_input


@app.cell
def _(env_path, load_dotenv, os, project, project_form):
    project_resolved = project
    if project_form.value is not None:
        candidate = str(project_form.value).strip()
        if candidate:
            current = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
            lines = [line for line in current.splitlines() if not line.startswith("EE_PROJECT=")]
            lines.append(f"EE_PROJECT={candidate}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            load_dotenv(env_path, override=True)
            project_resolved = os.environ.get("EE_PROJECT", "").strip()
    return (project_resolved,)


@app.cell
def _(ee, mo, project_form, project_resolved):
    if not project_resolved:
        mo.stop(
            True,
            mo.vstack(
                [
                    mo.md(
                        "Configura Earth Engine.\n\n"
                        "1. Obtén el project id con `gcloud projects list`.\n"
                        "2. Registra acceso en https://code.earthengine.google.com/register.\n"
                        "3. Escribe el `project-id` y pulsa `Guardar en .env`.\n"
                        "4. Ejecuta una vez: `uv run --package tecnicas-de-teledeteccion earthengine authenticate`."
                    ),
                    project_form,
                ]
            ),
        )
    ee.Initialize(project=project_resolved)

    # Print metadata for a DEM dataset.
    print(ee.Image('USGS/SRTMGL1_003').getInfo())
    return (ee,)


@app.cell
def _(ee):
    # Import the MODIS land cover collection.
    ee.ImageCollection('MODIS_061_MCD12Q1')

    # Import the MODIS land surface temperature collection.
    lst = ee.ImageCollection('MODIS_061_MOD11A1')

    # Import the USGS ground elevation image.
    ee.Image('USGS/SRTMGL1_003')
    return (lst,)


@app.cell
def _(lst):
    # Initial date of interest (inclusive).
    i_date = '2017-01-01'
    f_date = '2020-01-01'
    # Final date of interest (exclusive).
    # Selection of appropriate bands and dates for LST.
    lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)
    return


@app.cell
def _(ee):
    # Define the urban location of interest as a point near Lyon, France.
    u_lon = 4.8148
    u_lat = 45.7758
    ee.Geometry.Point(u_lon, u_lat)

    # Define the rural location of interest as a point away from the city.
    r_lon = 5.175964
    r_lat = 45.574064
    ee.Geometry.Point(r_lon, r_lat)
    return


if __name__ == "__main__":
    app.run()
