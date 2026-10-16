import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from pathlib import Path
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from tsunamis import legacy, mansinha, okada
    from tsunamis.faults import (
        InversionFormat,
        fault_plane,
        local_xy,
        metres_per_degree,
        papazachos_scaling,
        subfaults_from_inversion,
        to_local,
    )
    from tsunamis.grid import read_grid
    from tsunamis.kajiura import kajiura_filter

    PACKAGE = Path(__file__).resolve().parent
    plt.rcParams.update({"figure.dpi": 70, "savefig.dpi": 70, "font.size": 9})
    return (
        InversionFormat,
        PACKAGE,
        dedent,
        fault_plane,
        kajiura_filter,
        legacy,
        local_xy,
        mansinha,
        metres_per_degree,
        mo,
        np,
        okada,
        papazachos_scaling,
        plt,
        read_grid,
        subfaults_from_inversion,
        to_local,
    )


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
# De la falla al tsunami: deformación de Okada y filtro de Kajiura

Este notebook reúne en Python los programas de los laboratorios del curso
*Tópicos avanzados en tsunamis*, que estaban escritos en Fortran y MATLAB:

1. el modelo de Okada para la deformación vertical del fondo marino
   (`def_oka.f`, de Koshimura, modificado por C. Jiménez), vectorizado sobre la
   grilla con NumPy;
2. una comprobación cruzada con el modelo de Mansinha y Smylie (`deform.f`);
3. la geometría de la falla y sus subfallas (`fault_plane_n.m`, `falla.m` y
   `fuji2oka.m`);
4. la batimetría GEBCO leída con sus coordenadas correctas;
5. el filtro de Kajiura, que convierte la deformación del fondo en la
   elevación inicial de la superficie del mar.

Todo se calcula al construir la página; no se descarga nada.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
## 1. Validación contra la salida del programa en Fortran

El laboratorio 2 guarda `deform_a.grd`, la salida de `def_oka.f` para la falla
de `pfalla.inp`: una subfalla de 160 × 70 km, strike 310°, dip 18°, rake 63° y
3 m de deslizamiento, con el borde superior a 5 km de profundidad. Aquí se
repite el cálculo con el código nuevo, en la misma ventana de 485 × 553 celdas
y con el mismo tamaño de celda (926,6244 m), y se compara celda por celda.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(PACKAGE, dedent, legacy, mo, np):
    pfalla = legacy.read_pfalla(PACKAGE / "lab2" / "pfalla.inp")
    referencia = np.loadtxt(PACKAGE / "lab2" / "deform_a.grd")
    calculada = legacy.run_def_oka(pfalla)

    _dif = np.abs(calculada - referencia)
    _max_abs = _dif.max()
    _max_ref = np.abs(referencia).max()
    _fuertes = np.abs(referencia) >= 0.1
    _rel_punto = (_dif[_fuertes] / np.abs(referencia[_fuertes])).max()
    _i, _j = np.unravel_index(_dif.argmax(), _dif.shape)

    mo.md(
        dedent(
            f"""
| | valor |
|---|---|
| celdas comparadas | {referencia.shape[0]} × {referencia.shape[1]} |
| máximo de la referencia / mínimo | {referencia.max():.3f} m / {referencia.min():.3f} m |
| máximo de la calculada / mínimo | {calculada.max():.5f} m / {calculada.min():.5f} m |
| **diferencia absoluta máxima** | **{_max_abs:.2e} m** (celda {_i}, {_j}) |
| diferencia relativa al máximo de la referencia | {_max_abs / _max_ref:.2e} |
| diferencia relativa máxima donde \\|referencia\\| ≥ 0,1 m | {_rel_punto:.2e} |

`deform_a.grd` se escribió con formato `F7.3`, así que cada dato ya lleva un
redondeo de hasta 5 × 10⁻⁴ m, y `def_oka.f` trabaja en precisión simple. La
diferencia máxima es de ese orden. La diferencia relativa punto a punto solo
tiene sentido donde la deformación es apreciable; cerca de cero el redondeo del
archivo domina.
"""
        )
    )
    return calculada, pfalla, referencia


@app.cell(hide_code=True)
def _(calculada, plt, referencia):
    _fig, _ejes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for _ax, _campo, _t in (
        (_ejes[0], referencia, "deform_a.grd (Fortran)"),
        (_ejes[1], calculada, "NumPy"),
    ):
        _im = _ax.imshow(_campo.T, origin="lower", cmap="RdBu_r", vmin=-1.3, vmax=1.3)
        _ax.set_title(_t)
        _ax.set_xlabel("celda i (este)")
        _ax.set_ylabel("celda j (norte)")
    _fig.colorbar(_im, ax=_ejes[:2], label="uz (m)", shrink=0.8)
    _ejes[2].hist((calculada - referencia).ravel() * 1e3, bins=60, color="0.4")
    _ejes[2].set_yscale("log")
    _ejes[2].set_title("NumPy − Fortran")
    _ejes[2].set_xlabel("diferencia (mm)")
    _ejes[2].set_ylabel("celdas")
    _fig
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
**Una inconsistencia del laboratorio original.** `def_oka.f` fija el tamaño de
celda en 926,6244 m (30″), pero `grid_a.grd` del laboratorio 2 tenía celdas de
15″ (≈ 463 m). La ventana de 485 × 553 celdas se había dimensionado para 15″
(1,4 L × 1,6 L), de modo que `deform_a.grd` quedaba dibujada al doble de escala
respecto a la batimetría. Para reproducir el archivo hay que conservar la
convención; el resto del notebook usa coordenadas geográficas reales.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
## 2. Comprobación cruzada: Okada y Mansinha–Smylie

`deform.f` implementa el modelo de Mansinha y Smylie, que supone un sólido de
Poisson (μ = λ). El modelo de Okada del código usa las velocidades de la corteza
de `def_oka.f` (Vp = 4,82 km/s, Vs = 2,78 km/s). Para la misma falla
rectangular de la sección anterior se esperan campos casi idénticos.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(dedent, mansinha, mo, np, okada, pfalla, plt):
    _, _, _slip, _L, _W, _strike, _dip, _rake, _H = pfalla[0]
    falla_ref = okada.Subfault(0.0, 0.0, _L, _W, _H, _strike, _dip, _rake, _slip)
    _x = np.linspace(-250e3, 250e3, 251)
    _y = np.linspace(-150e3, 300e3, 226)
    _xx, _yy = np.meshgrid(_x, _y)
    _okada = okada.vertical_displacement(_xx, _yy, [falla_ref])
    _okada_poisson = okada.vertical_displacement(_xx, _yy, [falla_ref], ratio=0.5)
    _ms = mansinha.vertical_displacement(_xx, _yy, falla_ref)

    _fig, _ejes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    _im = _ejes[0].pcolormesh(
        _x / 1e3, _y / 1e3, _okada, cmap="RdBu_r", vmin=-1.3, vmax=1.3, shading="auto"
    )
    _ejes[0].set_title("Okada (corteza de def_oka.f)")
    _ejes[1].pcolormesh(
        _x / 1e3, _y / 1e3, _ms, cmap="RdBu_r", vmin=-1.3, vmax=1.3, shading="auto"
    )
    _ejes[1].set_title("Mansinha–Smylie")
    _fig.colorbar(_im, ax=_ejes[:2], label="uz (m)", location="bottom", shrink=0.6)
    for _ax in _ejes[:2]:
        _ax.set_aspect("equal")
        _ax.set_xlabel("este (km)")
        _ax.set_ylabel("norte (km)")

    _az = np.radians(_strike + 90.0)
    _s = np.linspace(-150e3, 250e3, 401)
    _cx = 0.5 * _L * np.sin(np.radians(_strike)) + 0.5 * _W * np.cos(
        np.radians(_dip)
    ) * np.sin(_az)
    _cy = 0.5 * _L * np.cos(np.radians(_strike)) + 0.5 * _W * np.cos(
        np.radians(_dip)
    ) * np.cos(_az)
    _px, _py = _cx + _s * np.sin(_az), _cy + _s * np.cos(_az)
    _ejes[2].plot(
        _s / 1e3, okada.vertical_displacement(_px, _py, [falla_ref]), label="Okada"
    )
    _ejes[2].plot(
        _s / 1e3,
        mansinha.vertical_displacement(_px, _py, falla_ref),
        "--",
        label="Mansinha–Smylie",
    )
    _ejes[2].set_xlabel("distancia al centro, perpendicular al strike (km)")
    _ejes[2].set_ylabel("uz (m)")
    _ejes[2].set_title("Perfil")
    _ejes[2].legend()
    _ejes[2].grid(alpha=0.3)

    _d_corteza = np.abs(_okada - _ms).max()
    _d_poisson = np.abs(_okada_poisson - _ms).max()
    mo.vstack(
        [
            _fig,
            mo.md(
                dedent(
                    f"""
Diferencia máxima entre ambos campos: **{_d_corteza:.1e} m** con la corteza de
`def_oka.f` (el máximo de la deformación es {_okada.max():.2f} m), y
**{_d_poisson:.1e} m** si Okada se evalúa con el sólido de Poisson, donde las
dos formulaciones deben coincidir.
"""
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
## 3. Geometría de la fuente

`falla.m` estima el tamaño de la ruptura a partir de la magnitud con las
relaciones de escala de Papazachos et al. (2004) para fallas de desplazamiento
inclinado en zonas de subducción, ajustadas para 6,7 ≤ Mw ≤ 9,2. El
deslizamiento se obtiene del momento sísmico con una rigidez de 4 × 10¹⁰ N/m².
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo, papazachos_scaling):
    _filas = []
    for _mw in (7.0, 7.5, 8.0, 8.5, 9.0):
        _e = papazachos_scaling(_mw)
        _filas.append(
            f"| {_mw:.1f} | {_e.length_km:.0f} | {_e.width_km:.0f} | "
            f"{_e.slip_m:.2f} | {_e.moment_nm:.2e} |"
        )
    mo.md(
        dedent(
            """
| Mw | L (km) | W (km) | deslizamiento (m) | M0 (N m) |
|---|---|---|---|---|
"""
        )
        + "\n".join(_filas)
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
## 4. Terremoto de Pisco de 2007: ocho subfallas

La guía del laboratorio 3 (`lab3.pdf`, tabla I) describe la fuente como
ocho subfallas cuadradas de 45 km: cuatro a lo largo del strike y dos a lo largo
del buzamiento, con strike 325°, dip 18° y rake 63°. Cada fila da la esquina
inferior izquierda de la subfuente, como la tabla de la guía, su profundidad
(14,00 o 27,91 km) y su deslizamiento. `subfaults_from_inversion` convierte esa
tabla a subfallas de Okada y trata la longitud geográfica de cada fila por
separado: `fuji2oka.m` preguntaba `if xo < 0` sobre el vector completo, y por
eso una tabla con longitudes geográficas de ambos signos se convertía igual para
todas las filas.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(
    InversionFormat,
    PACKAGE,
    dedent,
    fault_plane,
    mo,
    np,
    subfaults_from_inversion,
):
    _tabla = np.loadtxt(PACKAGE / "data" / "pisco_2007.txt")
    subfallas = subfaults_from_inversion(_tabla, InversionFormat.FAULT)
    contornos = [
        fault_plane(
            _s.lon,
            _s.lat,
            length=_s.length,
            width=_s.width,
            depth=_s.depth,
            strike=_s.strike,
            dip=_s.dip,
            rake=_s.rake,
            slip=_s.slip,
        ).outline
        for _s in subfallas
    ]
    _area = sum(_s.length * _s.width for _s in subfallas)
    _m0 = 4.0e10 * sum(_s.length * _s.width * _s.slip for _s in subfallas)
    mw_fuente = (np.log10(_m0) - 9.1) / 1.5
    _filas = [
        f"| {_k + 1} | {_s.lon:.3f} | {_s.lat:.3f} | {_s.depth / 1e3:.2f} | {_s.slip:.1f} |"
        for _k, _s in enumerate(subfallas)
    ]
    mo.md(
        dedent(
            """
| subfalla | lon (°) | lat (°) | profundidad (km) | deslizamiento (m) |
|---|---|---|---|---|
"""
        )
        + "\n".join(_filas)
        + dedent(
            f"""

Área total {_area / 1e6:.0f} km², momento {_m0:.2e} N m, equivalente a
Mw {mw_fuente:.2f} con rigidez de 4 × 10¹⁰ N/m².
"""
        )
    )
    return contornos, mw_fuente, subfallas


@app.cell(hide_code=True)
def _(PACKAGE, local_xy, metres_per_degree, np, read_grid, subfallas, to_local):
    grilla = read_grid(PACKAGE / "data" / "estudio.npz")
    lon0 = float(grilla.lon.mean())
    lat0 = float(grilla.lat.mean())
    _lon, _lat = np.meshgrid(grilla.lon, grilla.lat)
    este, norte = local_xy(_lon, _lat, lon0, lat0)
    locales = to_local(subfallas, lon0, lat0)
    _per_lon, _per_lat = metres_per_degree(lat0)
    paso_x = _per_lon * float(np.diff(grilla.lon).mean())
    paso_y = _per_lat * float(np.diff(grilla.lat).mean())
    return este, grilla, lat0, locales, lon0, norte, paso_x, paso_y


@app.cell(hide_code=True)
def _(dedent, este, grilla, mo, norte, locales, okada):
    uplift = okada.vertical_displacement(este, norte, locales)
    _mar = grilla.elevation < 0
    mo.md(
        dedent(
            f"""
## 5. Deformación del fondo marino

La grilla de estudio es un recorte de GEBCO ({grilla.source}) de
{grilla.lon[0]:.2f}° a {grilla.lon[-1]:.2f}° de longitud y de
{grilla.lat[0]:.2f}° a {grilla.lat[-1]:.2f}° de latitud, con celdas de 30″
({len(grilla.lon)} × {len(grilla.lat)}). Las coordenadas salen del encabezado
del archivo original, sin reconstruirlas a mano. La deformación se evalúa en
cada celda con Okada; la costa es el contorno de elevación 0 m.

Sobre el mar, el levantamiento máximo es de {uplift[_mar].max():.2f} m y el
descenso máximo, {uplift[_mar].min():.2f} m.
"""
        )
    )
    return (uplift,)


@app.cell(hide_code=True)
def _(contornos, grilla, np, plt, uplift):
    _fig, _ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    _lim = np.abs(uplift).max()
    _im = _ax.pcolormesh(
        grilla.lon,
        grilla.lat,
        uplift,
        cmap="RdBu_r",
        vmin=-_lim,
        vmax=_lim,
        shading="auto",
    )
    _ax.contour(
        grilla.lon, grilla.lat, grilla.elevation, levels=[0], colors="k", linewidths=1
    )
    _ax.contour(
        grilla.lon,
        grilla.lat,
        grilla.elevation,
        levels=[-6000, -4000, -2000, -200],
        colors="0.4",
        linewidths=0.4,
    )
    for _c in contornos:
        _ax.plot(_c[:, 0], _c[:, 1], color="darkgreen", linewidth=0.8)
    _ax.set_aspect(1 / np.cos(np.radians(grilla.lat.mean())))
    _ax.set_xlabel("longitud (°)")
    _ax.set_ylabel("latitud (°)")
    _ax.set_title("Deformación vertical del fondo (Okada, 8 subfallas)")
    _fig.colorbar(_im, ax=_ax, label="uz (m)", shrink=0.8)
    _fig
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
## 6. Del fondo a la superficie: el filtro de Kajiura

El mar no copia una deformación súbita del fondo. Para un océano incompresible
de profundidad $h$, la elevación inicial de la superficie tiene como
transformada de Fourier la del levantamiento dividida por $\cosh(kh)$, con $k$
el número de onda. Las longitudes de onda grandes frente a la profundidad pasan
casi sin cambio; las cortas se atenúan. Como $h$ varía, el filtro se aplica a
varios niveles de profundidad y cada celda toma el campo interpolado a su
profundidad local. Sobre tierra no se filtra.

Es una aproximación más simple que la de Abbate et al. (2024, *Natural Hazards
and Earth System Sciences* 24, 2773), quienes escriben el filtro de Kajiura
para un fondo que se levanta de forma súbita y lo evalúan celda por celda con
la profundidad local. Aquí solo se interpola entre campos filtrados con
profundidad constante.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(grilla, kajiura_filter, paso_x, paso_y, uplift):
    condicion_inicial = kajiura_filter(uplift, grilla.depth, paso_x, paso_y)
    return (condicion_inicial,)


@app.cell(hide_code=True)
def _(condicion_inicial, contornos, grilla, np, plt, uplift):
    _fig, _ejes = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
    _lim = np.abs(uplift).max()
    _aspecto = 1 / np.cos(np.radians(grilla.lat.mean()))
    for _ax, _campo, _t in (
        (_ejes[0], uplift, "Deformación del fondo"),
        (_ejes[1], condicion_inicial, "Condición inicial (Kajiura)"),
    ):
        _im = _ax.pcolormesh(
            grilla.lon,
            grilla.lat,
            _campo,
            cmap="RdBu_r",
            vmin=-_lim,
            vmax=_lim,
            shading="auto",
        )
        _ax.set_title(_t)
    _fig.colorbar(_im, ax=_ejes[:2], label="m", shrink=0.8)
    _dif = condicion_inicial - uplift
    _l = np.abs(_dif).max()
    _im = _ejes[2].pcolormesh(
        grilla.lon, grilla.lat, _dif, cmap="PuOr", vmin=-_l, vmax=_l, shading="auto"
    )
    _ejes[2].set_title("Diferencia")
    _fig.colorbar(_im, ax=_ejes[2], label="m", shrink=0.8)
    for _ax in _ejes:
        _ax.contour(
            grilla.lon,
            grilla.lat,
            grilla.elevation,
            levels=[0],
            colors="k",
            linewidths=0.8,
        )
        for _c in contornos:
            _ax.plot(_c[:, 0], _c[:, 1], color="darkgreen", linewidth=0.5)
        _ax.set_aspect(_aspecto)
        _ax.set_xlabel("longitud (°)")
    _ejes[0].set_ylabel("latitud (°)")
    _fig
    return


@app.cell(hide_code=True)
def _(dedent, mo, subfallas):
    azimut_perfil = (subfallas[0].strike + 90.0) % 360.0
    mo.md(
        dedent(
            f"""
## 7. Perfil perpendicular a la falla

El perfil pasa por el centro de la fuente en la dirección del buzamiento
(azimut strike + 90° = {azimut_perfil:.0f}°). Arriba, la batimetría con la línea
de costa; abajo, la deformación del fondo y la condición inicial filtrada.
"""
        )
    )
    return (azimut_perfil,)


@app.cell(hide_code=True)
def _(
    azimut_perfil,
    condicion_inicial,
    contornos,
    este,
    grilla,
    lat0,
    local_xy,
    lon0,
    norte,
    np,
    plt,
    uplift,
):
    _cx, _cy = local_xy(*np.mean([_c[:4] for _c in contornos], axis=(0, 1)), lon0, lat0)
    _az = np.radians(azimut_perfil)
    _s = np.arange(-120e3, 260e3, 500.0)
    _px, _py = _cx + _s * np.sin(_az), _cy + _s * np.cos(_az)
    _i = np.abs(este[0][None, :] - _px[:, None]).argmin(axis=1)
    _j = np.abs(norte[:, 0][None, :] - _py[:, None]).argmin(axis=1)
    _dentro = (
        (_px >= este[0, 0])
        & (_px <= este[0, -1])
        & (_py >= norte[0, 0])
        & (_py <= norte[-1, 0])
    )
    _s, _i, _j = _s[_dentro], _i[_dentro], _j[_dentro]

    _fig, (_a, _b, _c) = plt.subplots(
        3,
        1,
        figsize=(8, 7),
        sharex=True,
        constrained_layout=True,
        height_ratios=[1, 1.3, 0.8],
    )
    _z = grilla.elevation[_j, _i]
    _a.fill_between(_s / 1e3, _z, _z.min(), color="0.85")
    _a.plot(_s / 1e3, _z, color="0.3")
    _a.axhline(0, color="k", linewidth=0.8)
    _a.set_ylabel("elevación (m)")
    _a.set_title("Batimetría y topografía")
    _b.plot(_s / 1e3, uplift[_j, _i], label="deformación del fondo")
    _b.plot(
        _s / 1e3, condicion_inicial[_j, _i], "--", label="condición inicial (Kajiura)"
    )
    _b.axhline(0, color="k", linewidth=0.5)
    _b.set_ylabel("m")
    _b.legend()
    _c.plot(_s / 1e3, (condicion_inicial - uplift)[_j, _i] * 100, color="tab:red")
    _c.axhline(0, color="k", linewidth=0.5)
    _c.set_ylabel("Kajiura − fondo (cm)")
    _c.set_xlabel(f"distancia al centro de la fuente, azimut {azimut_perfil:.0f}° (km)")
    _costa = _s[np.flatnonzero(np.diff(np.sign(_z)))]
    for _ax in (_a, _b, _c):
        if len(_costa):
            _ax.axvline(_costa[0] / 1e3, color="tab:red", linestyle=":")
        _ax.grid(alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(condicion_inicial, dedent, grilla, mo, uplift):
    _mar = grilla.elevation < 0
    mo.md(
        dedent(
            f"""
| sobre el mar | máximo (m) | mínimo (m) |
|---|---|---|
| deformación del fondo | {uplift[_mar].max():.2f} | {uplift[_mar].min():.2f} |
| condición inicial filtrada | {condicion_inicial[_mar].max():.2f} | {condicion_inicial[_mar].min():.2f} |

Con subfallas de 45 km y una profundidad del agua de pocos kilómetros sobre la
fuente, la longitud de onda de la deformación es mucho mayor que la profundidad
del agua ($kh \\ll 1$ con $h$ la profundidad del agua) y el filtro cambia muy
poco el campo: la diferencia es de centímetros y se concentra en los bordes de
la ruptura, donde el levantamiento cambia más rápido. El filtro pesa más con
fuentes pequeñas o aguas profundas. Esta elevación es la condición inicial que
un modelo de propagación (por ejemplo, aguas someras lineales) recibiría para
seguir el tsunami hasta la costa.
"""
        )
    )
    return


if __name__ == "__main__":
    app.run()
