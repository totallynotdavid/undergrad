import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import importlib.util
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from Py6S import AeroProfile, AtmosProfile, GroundReflectance, SixS, Wavelength

    # El paquete `6s-bin` trae el ejecutable de 6S ya compilado. Su módulo
    # Python no se importa en Python 3.14, así que se ubica el ejecutable sin
    # importarlo.
    _sixs_bin = importlib.util.find_spec("sixs_bin")
    SIXS_PATH = Path(_sixs_bin.submodule_search_locations[0]) / "sixsV1.1"

    # Py6S escribe la entrada de cada corrida en un archivo temporal y lo borra
    # al terminar. Se guarda junto al cuaderno y no en el directorio del sistema.
    tempfile.tempdir = str(Path(__file__).resolve().parent)

    BANDS_UM = {
        "azul (0.485 µm)": 0.485,
        "rojo (0.675 µm)": 0.675,
        "NIR (0.865 µm)": 0.865,
        "vapor de agua (0.94 µm)": 0.94,
    }
    return (
        AeroProfile,
        AtmosProfile,
        BANDS_UM,
        GroundReflectance,
        SIXS_PATH,
        SixS,
        Wavelength,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Clase 0: transferencia radiativa con 6S a través de Py6S

    El modelo **6S** (Second Simulation of a Satellite Signal in the Solar
    Spectrum) calcula la radiancia que un sensor satelital mide sobre una
    superficie, dados la atmósfera, los aerosoles y la geometría de
    observación. **Py6S** es la interfaz de Python que escribe la entrada de 6S,
    ejecuta el programa y lee sus resultados.

    Este cuaderno ejecuta 6S al construir la página. El ejecutable viene
    compilado en el paquete `6s-bin`, de modo que no hace falta instalar nada
    ni descargar datos. Se estudia cómo cambia la señal que llega al sensor con
    tres variables:

    - el espesor óptico de aerosoles a 550 nm (AOD),
    - el contenido de vapor de agua de la columna atmosférica,
    - la geometría, aquí el ángulo cenital solar.
    """)
    return


@app.cell
def _(
    AeroProfile,
    AtmosProfile,
    GroundReflectance,
    SIXS_PATH,
    SixS,
    Wavelength,
):
    SURFACE_REFLECTANCE = 0.25
    OZONE_CM_ATM = 0.25

    def run_6s(wavelength_um, aod=0.2, water=2.5, solar_z=40.0):
        """Ejecuta 6S para una combinación y devuelve sus salidas."""
        s = SixS(str(SIXS_PATH))
        s.wavelength = Wavelength(wavelength_um)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, OZONE_CM_ATM)
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(
            SURFACE_REFLECTANCE
        )
        s.aot550 = aod
        s.geometry.solar_z = solar_z
        s.geometry.solar_a = 0.0
        s.geometry.view_z = 0.0
        s.geometry.view_a = 0.0
        # La fecha fija la distancia Tierra-Sol del escenario.
        s.geometry.month = 7
        s.geometry.day = 14
        s.run()
        return s.outputs

    return OZONE_CM_ATM, SURFACE_REFLECTANCE, run_6s


@app.cell
def _(OZONE_CM_ATM, SURFACE_REFLECTANCE, mo):
    mo.md(f"""
    ## Una corrida

    Se fija el caso de referencia: AOD de 0.2, columna de vapor de agua de
    2.5 g/cm², ángulo cenital solar de 40°, ozono de {OZONE_CM_ATM} cm-atm y
    superficie de reflectancia {SURFACE_REFLECTANCE}. Las salidas de 6S en la
    banda roja son:
    """)
    return


@app.cell
def _(pd, run_6s):
    _o = run_6s(0.675)
    pd.DataFrame(
        {
            "salida": [
                "Reflectancia aparente (sensor)",
                "Radiancia en el sensor",
                "Irradiancia solar directa",
                "Irradiancia solar difusa",
                "Transmitancia total por dispersión",
                "Transmitancia gaseosa total",
                "Albedo esférico de la atmósfera",
            ],
            "valor": [
                _o.apparent_reflectance,
                _o.apparent_radiance,
                _o.direct_solar_irradiance,
                _o.diffuse_solar_irradiance,
                _o.transmittance_total_scattering.total,
                _o.total_gaseous_transmittance,
                _o.spherical_albedo.total,
            ],
            "unidad": [
                "adimensional",
                "W m⁻² sr⁻¹ µm⁻¹",
                "W m⁻² µm⁻¹",
                "W m⁻² µm⁻¹",
                "adimensional",
                "adimensional",
                "adimensional",
            ],
        }
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Malla de simulaciones

    Se ejecuta 6S para cada combinación de AOD, vapor de agua, ángulo cenital
    solar y banda. La tabla muestra las primeras filas de la malla completa.
    """)
    return


@app.cell
def _(BANDS_UM, pd, run_6s):
    AOD_VALUES = [0.05, 0.2, 0.5, 1.0]
    WATER_VALUES = [0.5, 2.5, 4.5]
    SOLAR_ZENITH_VALUES = [20.0, 40.0, 60.0]

    _rows = []
    for _band, _wavelength in BANDS_UM.items():
        for _aod in AOD_VALUES:
            for _water in WATER_VALUES:
                for _solar_z in SOLAR_ZENITH_VALUES:
                    _o = run_6s(_wavelength, _aod, _water, _solar_z)
                    _rows.append(
                        {
                            "banda": _band,
                            "aod": _aod,
                            "agua_g_cm2": _water,
                            "zenital_solar": _solar_z,
                            "reflectancia_aparente": _o.apparent_reflectance,
                            "radiancia": _o.apparent_radiance,
                            "transmitancia_agua": _o.transmittance_water.total,
                        }
                    )
    grid = pd.DataFrame(_rows)
    grid.head(12)
    return AOD_VALUES, SOLAR_ZENITH_VALUES, WATER_VALUES, grid


@app.cell
def _(AOD_VALUES, SOLAR_ZENITH_VALUES, WATER_VALUES, grid, mo):
    mo.md(f"""
    La malla tiene **{len(grid)}** corridas: {len(AOD_VALUES)} valores de AOD,
    {len(WATER_VALUES)} de vapor de agua, {len(SOLAR_ZENITH_VALUES)} de ángulo
    cenital solar y {grid["banda"].nunique()} bandas.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Efecto de los aerosoles

    Con vapor de agua de 2.5 g/cm² y ángulo cenital solar de 40°, la
    reflectancia aparente en el sensor cambia con el AOD así:
    """)
    return


@app.cell
def _(grid):
    _fixed = grid[(grid["agua_g_cm2"] == 2.5) & (grid["zenital_solar"] == 40.0)]
    aod_table = _fixed.pivot(
        index="aod", columns="banda", values="reflectancia_aparente"
    ).round(4)
    aod_table
    return (aod_table,)


@app.cell
def _(np, plt, run_6s):
    _wavelengths = np.linspace(0.40, 1.00, 31)
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4.5))
    for _aod in (0.05, 0.2, 0.5, 1.0):
        _radiance = [
            run_6s(float(_w), aod=_aod).apparent_radiance for _w in _wavelengths
        ]
        ax_spectrum.plot(_wavelengths, _radiance, label=f"AOD = {_aod}")
    ax_spectrum.set_xlabel("Longitud de onda (µm)")
    ax_spectrum.set_ylabel(
        "Radiancia en el sensor ($W\\,m^{-2}\\,sr^{-1}\\,\\mu m^{-1}$)"
    )
    ax_spectrum.set_title("Espectro VNIR según el espesor óptico de aerosoles")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend()
    fig_spectrum.tight_layout()
    fig_spectrum
    return


@app.cell
def _(aod_table, mo):
    _blue = aod_table["azul (0.485 µm)"]
    _nir = aod_table["NIR (0.865 µm)"]
    mo.md(f"""
    El AOD actúa de dos formas. Aumenta la luz que la atmósfera dispersa hacia
    el sensor sin tocar la superficie, y atenúa la luz que viene de ella. Sobre
    una superficie de reflectancia 0.25 domina la atenuación: de AOD 0.05 a 1.0
    la reflectancia aparente baja de {_blue.iloc[0]} a {_blue.iloc[-1]} en el
    azul y de {_nir.iloc[0]} a {_nir.iloc[-1]} en el NIR. El efecto es mayor en
    el azul porque los aerosoles dispersan más en longitudes de onda cortas. El
    espectro de arriba usa vapor de agua de 2.5 g/cm² y ángulo cenital solar de
    40°.

    ## Efecto del vapor de agua

    El vapor de agua absorbe en bandas puntuales. A 0.94 µm la absorción es
    fuerte, mientras que en el azul y el rojo es despreciable. Esto se ve en la
    transmitancia del agua para un AOD de 0.2 y un ángulo cenital solar de 40°:
    """)
    return


@app.cell
def _(grid):
    _fixed = grid[(grid["aod"] == 0.2) & (grid["zenital_solar"] == 40.0)]
    water_table = _fixed.pivot(
        index="agua_g_cm2", columns="banda", values="transmitancia_agua"
    ).round(4)
    water_table
    return


@app.cell
def _(BANDS_UM, np, plt, run_6s):
    _water_range = np.linspace(0.25, 5.0, 20)
    fig_water, ax_water = plt.subplots(figsize=(8, 4.5))
    for _band in ("NIR (0.865 µm)", "vapor de agua (0.94 µm)"):
        _reflectance = [
            run_6s(BANDS_UM[_band], water=float(_w)).apparent_reflectance
            for _w in _water_range
        ]
        ax_water.plot(_water_range, _reflectance, label=_band)
    ax_water.set_xlabel("Vapor de agua de la columna (g/cm²)")
    ax_water.set_ylabel("Reflectancia aparente")
    ax_water.set_title("Reflectancia aparente según el vapor de agua")
    ax_water.grid(alpha=0.25)
    ax_water.legend()
    fig_water.tight_layout()
    fig_water
    return


@app.cell
def _(mo):
    mo.md("""
    ## Efecto de la geometría

    Un ángulo cenital solar mayor reduce la irradiancia que llega al suelo y
    alarga el camino de la luz por la atmósfera. La radiancia en el sensor, para
    AOD de 0.2 y vapor de agua de 2.5 g/cm², es:
    """)
    return


@app.cell
def _(grid, plt):
    _fixed = grid[(grid["aod"] == 0.2) & (grid["agua_g_cm2"] == 2.5)]
    fig_geometry, ax_geometry = plt.subplots(figsize=(8, 4.5))
    for _band, _group in _fixed.groupby("banda", sort=False):
        ax_geometry.plot(
            _group["zenital_solar"], _group["radiancia"], marker="o", label=_band
        )
    ax_geometry.set_xlabel("Ángulo cenital solar (°)")
    ax_geometry.set_ylabel(
        "Radiancia en el sensor ($W\\,m^{-2}\\,sr^{-1}\\,\\mu m^{-1}$)"
    )
    ax_geometry.set_title("Radiancia según el ángulo cenital solar")
    ax_geometry.grid(alpha=0.25)
    ax_geometry.legend()
    fig_geometry.tight_layout()
    fig_geometry
    return


if __name__ == "__main__":
    app.run()
