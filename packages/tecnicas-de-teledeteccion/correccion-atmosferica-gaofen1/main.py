import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Atmospheric Correction of Gaofen-1 images, using 6S Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este notebook implementa la corrección atmosférica para una imagen multiespectral GaoFen-1
    y calcula reflectancia de superficie para análisis de coberturas (suelo, vegetación y agua).

    Caso de estudio:
    - Sensor: GaoFen-1
    - Zona: Loreto
    - Fecha: 14-09-2020

    Estructura técnica del flujo:
    - carga de metadatos XML e imagen ORTO (.tif)
    - estimación de parámetros atmosféricos (6S + Earth Engine)
    - corrección por banda y ensamblado de reflectancia
    - índices de vegetación y firmas espectrales por píxel
    """)
    return


@app.cell
def _():
    import fnmatch
    import math
    import os
    from datetime import datetime
    from pathlib import Path
    from xml.dom import minidom

    import earthpy.plot as ep
    import ee
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rasterio
    import sixs_bin
    from atmosfera import Atmospheric
    from dotenv import load_dotenv
    from Py6S import AeroProfile, AtmosProfile, Geometry, SixS
    from wavelengths import PredefinedWavelengths, Wavelength

    return (
        Atmospheric,
        AeroProfile,
        AtmosProfile,
        Geometry,
        Path,
        PredefinedWavelengths,
        SixS,
        Wavelength,
        datetime,
        ee,
        ep,
        fnmatch,
        load_dotenv,
        math,
        minidom,
        np,
        os,
        pd,
        plt,
        rasterio,
        sixs_bin,
    )


@app.cell
def _(Path, load_dotenv, os):
    notebook_course_dir = Path(__file__).resolve().parents[1]
    env_path = notebook_course_dir / ".env"
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
            lines = [
                line
                for line in current.splitlines()
                if not line.startswith("EE_PROJECT=")
            ]
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
    return


@app.cell
def _(Path):
    # Paths definition
    base_dir = Path(__file__).resolve().parent
    dir_dataset = base_dir
    dir_atcor = base_dir / "public"
    dir_atcor.mkdir(parents=True, exist_ok=True)
    return dir_atcor, dir_dataset


@app.cell
def _(dir_dataset, fnmatch, minidom, mo, os):
    # load input image: XML and TIF
    xml_candidates = fnmatch.filter(os.listdir(dir_dataset), "*-MUX.xml")
    mo.stop(len(xml_candidates) == 0, mo.md(f"No se encontró XML en `{dir_dataset}`."))
    xml = minidom.parse(str(dir_dataset / xml_candidates[0]))

    files = os.listdir(dir_dataset)
    lista = [f for f in files if "ORTO" in f.upper() and f.lower().endswith(".tif")]
    mo.stop(
        len(lista) == 0,
        mo.md(
            f"No se encontró GeoTIFF ORTO en `{dir_dataset}`. "
            "Se requiere un archivo como `ORTO_...tif` para ejecutar la corrección."
        ),
    )
    lista.sort()
    return lista, xml


@app.cell
def _(dir_dataset, lista, rasterio):
    # Read image as matrix
    file = lista[0]
    sensor = []
    for k in range(1, 5):
        filename = dir_dataset / file
        data1 = rasterio.open(filename)
        data2 = data1.read(k)
        sensor.append(data2)  # image stacking
    return data1, sensor


@app.cell
def _(data1):
    # Close image to clear space in memory
    metada1 = data1.profile
    data1.close()
    return (metada1,)


@app.cell
def _(sensor):
    # Create a dictionary of names for every band of the image
    PeruSen = sensor
    BAND = {
        "B1": PeruSen[0],
        "B2": PeruSen[1],
        "B3": PeruSen[2],
        "B4": PeruSen[3],
    }
    return (BAND,)


@app.cell
def _(BAND, np):
    # Join the bands to visualize in RGB composition
    b1 = BAND["B1"]
    b2 = BAND["B2"]
    b3 = BAND["B3"]
    b4 = BAND["B4"]
    ORTHO = np.dstack((b1, b2, b3, b4))
    ortho = ORTHO.transpose([2, 0, 1])
    return b1, b2, b3, b4, ortho


@app.cell
def _(datetime, xml):
    # Read metadata parameters required for the 6S model
    # Observation time
    tiempo_inicial = xml.getElementsByTagName("StartTime")[0]  # image start time
    fecha = datetime.strptime(tiempo_inicial.firstChild.data[0:9], "%Y-%m-%d")
    mes = fecha.timetuple().tm_mon  # month
    dia = fecha.timetuple().tm_mday  # day
    fecha.timetuple().tm_year  # year
    # Solar geometric
    sun_z = float(
        xml.getElementsByTagName("SolarZenith")[0].firstChild.data[0:4]
    )  # Solar Zenith Angle
    sun_a = float(
        xml.getElementsByTagName("SolarAzimuth")[0].firstChild.data[0:4]
    )  # Solar Azimuth Angle
    # View Geometric
    view_z = float(
        xml.getElementsByTagName("SatelliteZenith")[0].firstChild.data[0:4]
    )  # Sat Zenith Angle
    view_a = float(
        xml.getElementsByTagName("SatelliteAzimuth")[0].firstChild.data[0:5]
    )  # Sat Azimuth Angle
    # image coordinates
    LAT = float(
        xml.getElementsByTagName("CenterLatitude")[0].firstChild.data[0:10]
    )  # Center Latitude
    LON = float(
        xml.getElementsByTagName("CenterLongitude")[0].firstChild.data[0:10]
    )  # Center Longitude
    return LAT, LON, dia, fecha, mes, sun_a, sun_z, view_a, view_z


@app.cell
def _():
    # Read calibration coefficientes
    # Published yearly at the Dunhuang test site by the China Centre for Resource Satellite Data and
    # Application (CCRSDA)
    # https://doi.org/10.3390/rs8020132
    # Gain=[0.0738, 0.0656, 0.059, 0.0585]
    Gain = [0.1490, 0.1328, 0.1311, 0.1217]
    Bias = [0.0, 0.0, 0.0, 0.0]
    return Bias, Gain


@app.cell
def _(Atmospheric, LAT, LON, ee, fecha):
    # Reading auxiliary data
    dato = ee.Date(fecha.isoformat())
    geom = ee.Geometry.Point(LON, LAT)

    # Digital Elevation Model - DEM SRTMN
    SRTM = ee.Image("USGS/SRTMGL1_003")
    alt = (
        SRTM.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom.centroid())
        .get("elevation")
        .getInfo()
    )
    km = alt / 1000  # Py6S uses units of kilometers

    # Atmospheric data
    h2o = Atmospheric.water(geom, dato).getInfo()
    o3 = Atmospheric.ozone(geom, dato).getInfo()
    aot = Atmospheric.aerosol(geom, dato).getInfo()
    return aot, h2o, km, o3


@app.cell
def _(
    AeroProfile,
    AtmosProfile,
    Geometry,
    SixS,
    aot,
    dia,
    h2o,
    km,
    mes,
    o3,
    sixs_bin,
    sun_a,
    sun_z,
    view_a,
    view_z,
):
    # Setting parameters for the 6S model
    sixs_path = sixs_bin.get_path("1.1")
    s = SixS(path=str(sixs_path))

    # Atmosphere
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(
        h2o, o3
    )  # Set the atmosphere profile
    s.aero_profile = AeroProfile.Continental  # Set the aerosol profile
    s.aot550 = aot

    # Geometry
    s.geometry = Geometry.User()
    s.geometry.solar_z = sun_z
    s.geometry.solar_a = sun_a
    s.geometry.view_z = view_z
    s.geometry.view_a = view_a
    s.geometry.month = mes
    s.geometry.day = dia
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(km)  # Altitude GaoFen-1 -> 704.22 km
    return (s,)


@app.cell
def _(PredefinedWavelengths, Wavelength):
    # Read Spectral Response Function. RSR for given band name
    def spectralResponseFunction(bandname):
        bandSelect = {
            "B1": PredefinedWavelengths.GF1PMS_B1,
            "B2": PredefinedWavelengths.GF1PMS_B2,
            "B3": PredefinedWavelengths.GF1PMS_B3,
            "B4": PredefinedWavelengths.GF1PMS_B4,
        }
        return Wavelength(bandSelect[bandname])

    return (spectralResponseFunction,)


@app.cell
def _(BAND, Bias, Gain):
    # Convert Raw radiometric counts (DN) to TOA Radiance (L)
    # Formulae L=DN/GAIN+BIAS [watt/m2/steradians/micrometers]
    def nd_to_rad(bandname):
        BIAS_PERU = Bias
        BIAS_BAND = {
            "B1": BIAS_PERU[0],
            "B2": BIAS_PERU[1],
            "B3": BIAS_PERU[2],
            "B4": BIAS_PERU[3],
        }
        GAIN_PERU = Gain
        GAIN_BAND = {
            "B1": GAIN_PERU[0],
            "B2": GAIN_PERU[1],
            "B3": GAIN_PERU[2],
            "B4": GAIN_PERU[3],
        }
        rad = (BAND[bandname] / GAIN_BAND[bandname]) + BIAS_BAND[bandname]
        return rad

    return (nd_to_rad,)


@app.cell
def _(math, nd_to_rad, np, s, spectralResponseFunction):
    # Calculate surface reflectance from at-sensor radiance given waveband name
    def surface_reflectance(bandname):
        # run 6S for this waveband
        s.wavelength = spectralResponseFunction(bandname)
        s.run()
        # extract 6S outputs
        Edir = s.outputs.direct_solar_irradiance  # direct solar irradiance
        Edif = s.outputs.diffuse_solar_irradiance  # diffuse solar irradiance
        Lp = s.outputs.atmospheric_intrinsic_radiance  # path radiance
        absorb = s.outputs.trans["global_gas"].upward  # absorption transmissivity
        scatter = s.outputs.trans[
            "total_scattering"
        ].upward  # scattering transmissivity
        tau2 = absorb * scatter  # total transmissivity
        # radiance to surface reflectance
        rad = nd_to_rad(bandname)
        ref = ((rad - Lp) * math.pi) / (tau2 * (Edir + Edif))
        result = np.where(ref <= 0, np.nan, ref)  # set values less than zero to NaN
        return result

    return (surface_reflectance,)


@app.cell
def _(surface_reflectance):
    # Applied Atmospheric Correction
    blue = surface_reflectance("B1")
    green = surface_reflectance("B2")
    red = surface_reflectance("B3")
    nir = surface_reflectance("B4")
    return blue, green, nir, red


@app.cell
def _(blue, green, nir, np, red):
    # staking de las imagenes
    reflectancia = np.dstack((blue, green, red, nir))
    Reflectancia = reflectancia.transpose([2, 0, 1])
    return (Reflectancia,)


@app.cell
def _(Reflectancia):
    print("Proceso culminado:")
    print("Bandas:", Reflectancia.shape[0])
    print("Filas:", Reflectancia.shape[1])
    print("Columnas:", Reflectancia.shape[2])
    return


@app.cell
def _(Reflectancia, ep, ortho, plt):
    # View result in Raw RGB and surface refectance (6S Model)
    fig, axv = plt.subplots(1, 2, figsize=(10, 10))
    ep.plot_rgb(
        ortho, rgb=(2, 1, 0), ax=axv[0], title="GF-1 Raw RGB", stretch=True, str_clip=4
    )
    ep.plot_rgb(
        Reflectancia,
        rgb=(2, 1, 0),
        ax=axv[1],
        title="GF-1 6S Atmospheric Correction",
        stretch=True,
        str_clip=4,
    )
    return


@app.cell
def _(BAND, b1, b2, b3, b4, blue, green, nir, plt, red):
    # Histogram Display
    fig_1, axh = plt.subplots(4, 2, figsize=(10, 10))
    axh[0, 0].hist(BAND["B1"].flatten(), bins=20, range=(1, b1.max()), color="skyblue")
    axh[0, 0].set_title("Raw RGB")
    axh[0, 0].set_ylabel("Blue Band")
    axh[0, 1].hist(blue.flatten(), bins=20, color="skyblue")
    axh[0, 1].set_title("6S Atmosphgeric Correction")
    axh[1, 0].hist(BAND["B2"].flatten(), bins=20, range=(1, b2.max()), color="green")
    axh[1, 0].set_ylabel("Green Band")
    axh[1, 1].hist(green.flatten(), bins=20, color="green")
    axh[2, 0].hist(BAND["B3"].flatten(), bins=20, range=(1, b3.max()), color="red")
    axh[2, 0].set_ylabel("Red Band")
    axh[2, 1].hist(red.flatten(), bins=20, color="red")
    axh[3, 0].hist(BAND["B4"].flatten(), bins=20, range=(1, b4.max()), color="orange")
    axh[3, 0].set_ylabel("NIR Band")
    axh[3, 1].hist(nir.flatten(), bins=20, color="orange")
    return


@app.cell
def _(Reflectancia, dir_atcor, lista, metada1, rasterio):
    # Save atmospherically corrected image
    param = metada1
    param["dtype"] = "float32"
    fileo6s = dir_atcor / f"R6S_{lista[0]}"
    try:
        with rasterio.open(str(fileo6s), "w", **param) as dst:
            dst.write(Reflectancia)
    except NameError:
        print("variable no definido")
    except ValueError:
        print("el valor no es valido")
    else:
        print("se guardo satisfactoriamente")
    return


@app.cell
def _(np):
    # Define vegetation inde - VIs
    def computeNDVIband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_red = sat_data[2].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (band_nir - band_red) / (band_nir + band_red)
        return ndvi

    def computeGNDVIband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_green = sat_data[1].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        gndvi = (band_nir - band_green) / (band_nir + band_green)
        return gndvi

    def computeSRband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_red = sat_data[2].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        srvi = band_nir / band_red
        return srvi

    def computeEVIband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_blue = sat_data[0].astype(float)  ## leer banda blue
        band_red = sat_data[2].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        evi = (2.5 * (band_nir - band_red)) / (
            band_nir + 6.0 * band_red - 7.5 * band_blue + 1.0
        )
        return evi

    def computeCIgband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_green = sat_data[1].astype(float)  ## leer banda green
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        cigreen = (band_nir / band_green) - 1.0
        return cigreen

    def computeCIrepband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_red = sat_data[2].astype(float)  ## leer banda red
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        cirep = (band_nir / band_red) - 1.0
        return cirep

    def computeMCARIband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_green = sat_data[1].astype(float)  ## leer banda green
        band_red = sat_data[2].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        mcari = 1.2 * (2.5 * (band_nir - band_red) - 1.3 * (band_nir - band_green))
        return mcari

    def computeSAVIband(sat_data):
        ## indices de las bandas B,G,R,Nir -> 1,2,3,4
        band_red = sat_data[2].astype(float)  ## leer banda roja
        band_nir = sat_data[3].astype(float)  ## leer banda infraroja
        # Permitir la division por cero
        np.seterr(divide="ignore", invalid="ignore")
        savi = (1 + 0.5) * (band_nir - band_red) / (band_nir + band_red + 0.5)
        return savi

    return (
        computeCIgband,
        computeCIrepband,
        computeEVIband,
        computeGNDVIband,
        computeMCARIband,
        computeNDVIband,
        computeSAVIband,
        computeSRband,
    )


@app.cell
def _(
    Reflectancia,
    computeCIgband,
    computeCIrepband,
    computeEVIband,
    computeGNDVIband,
    computeMCARIband,
    computeNDVIband,
    computeSAVIband,
    computeSRband,
    ep,
    plt,
):
    # Plot and tabulation the VIs
    ndvi_band = computeNDVIband(Reflectancia)
    gndvi_band = computeGNDVIband(Reflectancia)
    srvi_band = computeSRband(Reflectancia)
    evi_band = computeEVIband(Reflectancia)
    cigreen_band = computeCIgband(Reflectancia)
    cirep_band = computeCIrepband(Reflectancia)
    mcari_band = computeMCARIband(Reflectancia)
    savi_band = computeSAVIband(Reflectancia)
    fig_2, ax = plt.subplots(3, 3, figsize=(10, 10))
    ep.plot_rgb(
        Reflectancia,
        rgb=(2, 1, 0),
        ax=ax[0, 0],
        title="Imagen GaoFen-1 - RGB",
        stretch=True,
        str_clip=4,
    )
    ep.plot_bands(ndvi_band, cmap="RdYlGn", ax=ax[0, 1], title="NDVI", vmin=-1, vmax=1)
    ep.plot_bands(
        gndvi_band, cmap="RdYlGn", ax=ax[0, 2], title="GNDVI", vmin=-1, vmax=1
    )
    ep.plot_bands(evi_band, cmap="RdYlGn", ax=ax[1, 0], title="EVI", vmin=-1, vmax=1)
    ep.plot_bands(
        cigreen_band, cmap="RdYlGn", ax=ax[1, 1], title="CI-green", vmin=-1, vmax=5
    )
    ep.plot_bands(
        cirep_band, cmap="RdYlGn", ax=ax[1, 2], title="CI-rep", vmin=-1, vmax=10
    )
    ep.plot_bands(mcari_band, cmap="RdYlGn", ax=ax[2, 0], title="MCARI")
    ep.plot_bands(savi_band, cmap="RdYlGn", ax=ax[2, 1], title="SAVI")
    ep.plot_bands(srvi_band, cmap="RdYlGn", ax=ax[2, 2], title="SR", vmin=0, vmax=10)
    plt.savefig("F:/IVs-GF1-1200dpi.png")
    return


@app.cell
def _(Reflectancia, data1, pd, plt, rasterio):
    # Plot spectral signature of coverages
    import pyproj

    proj = pyproj.Transformer.from_crs(4326, data1.crs, always_xy=True)
    pes_pixel = pd.DataFrame()  # Transforma geograficas a UTM
    wavelengths = [450, 520, 620, 770]
    xlon1 = -73.0596008  # Longitd de onda central de las bandas GaoFen-1
    ylat1 = -3.98170897
    ##Coordenada de pixel - Cultivos
    xs1, ys1 = proj.transform(xlon1, ylat1)  # E=610484.063
    row1, col1 = rasterio.transform.rowcol(data1.transform, xs1, ys1)  # N=9325505.625
    print(row1, col1)
    xlon2 = -73.3044981  # convierte UTM a (fil, columna)
    ylat2 = -3.73590708
    ##Coordenada de pixel - Suelo (Playa)
    xs2, ys2 = proj.transform(xlon2, ylat2)  # E=610484.063
    row2, col2 = rasterio.transform.rowcol(data1.transform, xs2, ys2)  # N=9325505.625
    print(row2, col2)
    xlon3 = -73.3202919  # convierte UTM a (fil, columna)
    ylat3 = -3.82773775
    ##Coordenada de pixel - Agua (Laguna)
    xs3, ys3 = proj.transform(xlon3, ylat3)  # E=610484.063
    row3, col3 = rasterio.transform.rowcol(data1.transform, xs3, ys3)  # N=9325505.625
    print(row3, col3)
    pes_pixel["wavelengths"] = wavelengths  # convierte UTM a (fil, columna)
    pes_pixel["r_Cultivo"] = Reflectancia[:, row1, col1]
    pes_pixel["r_Suelo"] = Reflectancia[:, row2, col2]
    pes_pixel["r_Agua"] = Reflectancia[:, row3, col3]
    print(pes_pixel)
    fig_3 = plt.figure(figsize=(15, 5))
    ax1 = fig_3.add_subplot(1, 2, 1)
    pes_pixel.plot(ax=ax1, x="wavelengths", y="r_Cultivo", kind="line")
    ax1.set_title("Spectra Signature - Cultivos")
    ax1.grid("on")
    ax2 = fig_3.add_subplot(1, 2, 2)
    pes_pixel.plot(ax=ax2, x="wavelengths", y="r_Suelo", kind="line")
    ax2.set_title("Spectra Signature - Suelo")
    ax2.grid("on")
    return


@app.cell
def _(Reflectancia, mo):
    pixel_x = mo.ui.slider(
        start=0, stop=Reflectancia.shape[2] - 1, step=1, value=0, label="pixel_x"
    )
    pixel_y = mo.ui.slider(
        start=0, stop=Reflectancia.shape[1] - 1, step=1, value=0, label="pixel_y"
    )
    return pixel_x, pixel_y


@app.cell
def _(Reflectancia, data1, mo, np, pixel_x, pixel_y, plt):
    reflectance = Reflectancia[:, pixel_y.value, pixel_x.value]
    _wavelengths = np.array([450, 520, 620, 770])
    refl_band = Reflectancia[3, :, :]

    _fig = plt.figure(figsize=(15, 5))
    _ax1 = _fig.add_subplot(1, 2, 1)
    _ax1.plot(_wavelengths, reflectance)
    _ax1.set_title(f"Firma espectral ({pixel_x.value}, {pixel_y.value})")
    _ax1.set_xlabel("Wavelength, nm")
    _ax1.set_ylabel("Reflectancia")
    _ax1.grid("on")

    _ax2 = _fig.add_subplot(1, 2, 2)
    _plot = _ax2.imshow(refl_band, extent=data1.bounds, clim=(0, 0.2))
    _ax2.set_title("Localizador de pixel Gaofen-1")
    _fig.colorbar(_plot, ax=_ax2, aspect=20).set_label(
        "Reflectance", rotation=90, labelpad=20
    )
    _ax2.ticklabel_format(useOffset=False, style="plain")
    for label in _ax2.get_xticklabels():
        label.set_rotation(90)
    _ax2.plot(
        data1.bounds[0] + pixel_x.value,
        data1.bounds[3] - pixel_y.value,
        "s",
        markersize=5,
        color="red",
    )
    _ax2.set_xlim(data1.bounds[0], data1.bounds[1])
    _ax2.set_ylim(data1.bounds[2], data1.bounds[3])
    _fig.tight_layout()

    mo.vstack([mo.hstack([pixel_x, pixel_y], justify="start"), _fig])
    return


if __name__ == "__main__":
    app.run()
