import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import base64

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dinamica_molecular import analisis, animacion, lennard_jones, tight_binding

    plt.rcParams["figure.dpi"] = 72

    return analisis, animacion, base64, lennard_jones, mo, np, plt, tight_binding


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dinámica molecular: argón líquido y cúmulos de potasio

    Un fluido de Lennard-Jones se integra con velocity Verlet en una caja
    cúbica periódica, siguiendo el método de Verlet (1967). Las fuerzas usan la
    convención de mínima imagen y las posiciones nunca se reflejan en las
    paredes, de modo que la fuerza y la frontera son coherentes. Al final se
    usa el mismo integrador con un segundo potencial, de enlace fuerte en
    segundo momento, para pequeños cúmulos de potasio.

    El potencial de pares es

    $$
    V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}
    - \left(\frac{\sigma}{r}\right)^{6}\right],
    $$

    truncado en $r_c = 2.5\,\sigma$ y desplazado para que tanto $V$ como la
    fuerza se anulen en $r_c$. Se trabaja en unidades reducidas
    ($\sigma = \varepsilon = m = 1$); para el argón, $\sigma = 3.405$ Å y
    $\varepsilon/k_B = 119.8$ K.

    Todo se calcula al construir la página: sin datos externos.
    """)
    return


@app.cell
def _(lennard_jones, np):
    # Punto de la tabla I de Verlet: densidad 0.85 y T ~ 0.78.
    DENSIDAD = 0.85
    T_OBJETIVO = 0.78
    DT = 0.004
    PASOS_EQUILIBRADO = 1500
    PASOS_PRODUCCION = 4000
    CADA = 10

    pos0, LADO = lennard_jones.red_fcc(4, DENSIDAD)
    N = pos0.shape[0]
    vel0 = lennard_jones.velocidades_maxwell(N, T_OBJETIVO, np.random.default_rng(1))

    equilibrado = lennard_jones.integrar(
        pos0,
        vel0,
        LADO,
        DT,
        PASOS_EQUILIBRADO,
        t_objetivo=T_OBJETIVO,
        cada=PASOS_EQUILIBRADO,
    )
    corrida = lennard_jones.integrar(
        equilibrado.pos_final,
        equilibrado.vel_final,
        LADO,
        DT,
        PASOS_PRODUCCION,
        cada=CADA,
    )
    return (
        CADA,
        DENSIDAD,
        DT,
        LADO,
        N,
        PASOS_EQUILIBRADO,
        PASOS_PRODUCCION,
        T_OBJETIVO,
        corrida,
    )


@app.cell(hide_code=True)
def _(DENSIDAD, DT, LADO, N, PASOS_EQUILIBRADO, PASOS_PRODUCCION, mo):
    mo.md(f"""
    ## Sistema simulado

    | | |
    |---|---|
    | Partículas | {N} (red fcc de 4³ celdas, fundida) |
    | Densidad $\\rho^*$ | {DENSIDAD} |
    | Lado de la caja | {LADO:.3f} $\\sigma$ |
    | Paso $\\Delta t$ | {DT} |
    | Equilibrado (con reescalado de $T$) | {PASOS_EQUILIBRADO} pasos |
    | Producción (energía constante) | {PASOS_PRODUCCION} pasos |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conservación de la energía

    Velocity Verlet es simpléctico: la energía total $E = K + U$ no crece ni
    decae, solo oscila alrededor de un valor fijo. En la producción no hay
    termostato, así que $E(t)$ es la comprobación de que las fuerzas y el
    integrador son coherentes. El truncamiento con fuerza desplazada evita los
    saltos de energía cuando un par cruza $r_c$.
    """)
    return


@app.cell
def _(corrida, plt):
    _n = corrida.posiciones.shape[1]
    _m = slice(None, None, 5)
    _t = corrida.tiempo[_m]
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    _ax1.plot(_t, (corrida.energia_cinetica / _n)[_m], label="cinética $K/N$")
    _ax1.plot(_t, (corrida.energia_potencial / _n)[_m], label="potencial $U/N$")
    _ax1.plot(_t, (corrida.energia_total / _n)[_m], label="total $E/N$", color="black")
    _ax1.set_xlabel("tiempo $t$ (unidades reducidas)")
    _ax1.set_ylabel("energía por partícula ($\\varepsilon$)")
    _ax1.legend(fontsize=8)
    _desv = (corrida.energia_total - corrida.energia_total[0]) / abs(
        corrida.energia_total[0]
    )
    _ax2.plot(_t, _desv[_m], color="black", linewidth=0.8)
    _ax2.set_xlabel("tiempo $t$ (unidades reducidas)")
    _ax2.set_ylabel("$(E(t)-E(0))/|E(0)|$")
    _fig
    return


@app.cell(hide_code=True)
def _(corrida, mo, np):
    _total = corrida.energia_total
    _desv = np.max(np.abs(_total - _total[0])) / abs(_total[0])
    mo.md(f"""
    La desviación máxima respecto a la energía inicial es de
    **{_desv:.1e}** en valor relativo a lo largo de {len(_total) - 1} pasos. El
    ensayo `tests/test_lennard_jones.py` fija una cota de $5\\times10^{{-4}}$ para
    una corrida corta y comprueba que un paso de 0.03 la supera.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temperatura

    $T = 2K / [3(N-1)]$, con $3(N-1)$ grados de libertad porque el momento
    total es nulo. En una corrida microcanónica $T$ fluctúa alrededor de su
    media; no está fijada.
    """)
    return


@app.cell
def _(corrida, plt):
    _fig, _ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    _ax.plot(corrida.tiempo[::5], corrida.temperatura[::5], linewidth=0.8)
    _ax.axhline(
        corrida.temperatura.mean(), color="black", linestyle="--", linewidth=0.8
    )
    _ax.set_xlabel("tiempo $t$ (unidades reducidas)")
    _ax.set_ylabel("$T^*$")
    _fig
    return


@app.cell(hide_code=True)
def _(DENSIDAD, LADO, corrida, lennard_jones, mo, np):
    _n = corrida.posiciones.shape[1]
    _u = (
        np.mean(
            [
                lennard_jones.energia_truncada(p, LADO, lennard_jones.CORTE)
                for p in corrida.posiciones
            ]
        )
        / _n
    )
    _cola = lennard_jones.correccion_cola(DENSIDAD)
    mo.md(f"""
    Temperatura media: $\\langle T^*\\rangle$ = **{corrida.temperatura.mean():.3f}**,
    con desviación estándar {corrida.temperatura.std():.3f}.

    Como comprobación contra Verlet (1967, tabla I), con $\\rho^* = 0.85$ y
    $T^* = 0.782$ la energía potencial por partícula es $-6.04$. Con las
    configuraciones de esta corrida, para el potencial truncado sin desplazar
    en $r_c = 2.5$ se obtiene $U/N$ = {_u:.3f}; sumando la corrección de cola
    de largo alcance ($g = 1$ más allá de $r_c$), {_cola:.3f}, resulta
    **{_u + _cola:.3f}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Función de distribución radial

    $g(r)$ es la densidad local relativa a la de un gas ideal a distancia $r$
    de una partícula. Se promedia sobre las instantáneas de la producción y se
    calcula hasta $L/2$, donde la mínima imagen deja de ser única. El pico
    cerca de $r \approx \sigma$ y los picos siguientes, que se amortiguan hasta
    $g \to 1$, son la estructura de capas de un líquido.
    """)
    return


@app.cell
def _(LADO, analisis, corrida, plt):
    _r, _g = analisis.g_de_r(corrida.posiciones[len(corrida.posiciones) // 4 :], LADO)
    _fig, _ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    _ax.plot(_r, _g)
    _ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    _ax.set_xlabel("$r / \\sigma$")
    _ax.set_ylabel("$g(r)$")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Difusión

    El desplazamiento cuadrático medio (MSD) se calcula con las posiciones sin
    envolver en la caja, promediando sobre las partículas y sobre todos los
    orígenes de tiempo de la corrida. Para tiempos largos cumple la relación de
    Einstein

    $$
    \langle |\mathbf r(t) - \mathbf r(0)|^2 \rangle = 6\,D\,t + b,
    $$

    así que $D$ es un sexto de la pendiente del tramo lineal. A tiempos cortos
    el movimiento es balístico ($\propto t^2$) y ese tramo se excluye del
    ajuste.
    """)
    return


@app.cell
def _(analisis, corrida, plt):
    _msd = analisis.desplazamiento_cuadratico_medio(corrida.posiciones)
    tiempo_msd = corrida.tiempo_cuadros[: _msd.size]
    VENTANA = (1.0, float(tiempo_msd[-1]))
    D, ordenada = analisis.coeficiente_difusion(tiempo_msd, _msd, *VENTANA)

    _fig, _ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    _ax.plot(tiempo_msd, _msd, label="MSD")
    _ax.plot(
        tiempo_msd,
        6 * D * tiempo_msd + ordenada,
        "--",
        color="black",
        label=f"ajuste: $D^* = {D:.3f}$",
    )
    _ax.axvspan(*VENTANA, color="gray", alpha=0.12)
    _ax.set_xlabel("tiempo $t$ (unidades reducidas)")
    _ax.set_ylabel("MSD ($\\sigma^2$)")
    _ax.legend(fontsize=8)
    _fig
    return D, VENTANA


@app.cell(hide_code=True)
def _(D, N, VENTANA, mo, np):
    # La unidad de D es sigma * sqrt(epsilon / m), en m^2/s para el argón.
    _sigma = 3.405e-10
    _v0 = np.sqrt(119.8 * 1.380649e-23 / (39.948 * 1.66053907e-27))
    mo.md(f"""
    Con el ajuste en la zona sombreada ($t \\in [{VENTANA[0]:.0f}, {VENTANA[1]:.0f}]$),
    $D^*$ = **{D:.3f}** en unidades reducidas. Para el argón
    ($\\sigma$ = 3.405 Å, $\\varepsilon/k_B$ = 119.8 K, $m$ = 39.948 u) la unidad
    de $D$ es $\\sigma\\sqrt{{\\varepsilon/m}}$ = {_sigma * _v0:.2e} m²/s, de modo que
    $D$ = **{D * _sigma * _v0:.2e} m²/s**.

    Es un solo valor de una sola corrida de {N} partículas: la incertidumbre
    estadística no se ha estimado.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Animación

    Las partículas que empiezan en la capa $z < L/4$ se proyectan sobre el
    plano $xy$, con las posiciones envueltas en la caja. Cada una conserva el
    color de su coordenada $x$ inicial, de modo que la mezcla de colores es la
    difusión que mide el MSD.
    """)
    return


@app.cell
def _(LADO, animacion, base64, corrida, mo):
    _cuadros = corrida.posiciones[::13][:30]
    gif = animacion.gif_trayectoria(_cuadros, LADO, pixeles=200)
    mo.image(
        "data:image/gif;base64," + base64.b64encode(gif).decode("ascii"),
        width=200,
        alt="Proyección xy de las partículas de una capa de la caja",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cúmulos de potasio: enlace fuerte en segundo momento

    Para metales como el potasio un potencial de pares no basta: la energía de
    enlace de un átomo no es la suma de sus pares, sino que satura con el
    número de vecinos. La aproximación del segundo momento de la banda de
    enlace fuerte lo recoge con una raíz cuadrada:

    $$
    E = \sum_i \Big[ A \sum_{j\neq i} e^{-p\,(r_{ij}/r_0 - 1)}
    - \sqrt{\textstyle\sum_{j\neq i} \xi^2\, e^{-2q\,(r_{ij}/r_0 - 1)}}\;\Big].
    $$

    El primer término es una repulsión de pares y el segundo la energía de
    banda. Los parámetros ($A$, $\xi$, $r_0$, $p$, $q$) para K son los del
    código original de este paquete. Los cúmulos regulares de 2, 3 y 4 átomos
    (dímero, triángulo y tetraedro) permiten ver la dependencia con la
    coordinación: la energía por átomo baja al pasar de 1 a 3 vecinos.
    """)
    return


@app.cell
def _(np, plt, tight_binding):
    _d = np.linspace(3.3, 6.0, 300)
    _fig, _ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    _nombres = {2: "dímero", 3: "triángulo", 4: "tetraedro"}
    for _n, _nombre in _nombres.items():
        _e = [
            tight_binding.energia_por_atomo(tight_binding.simplex_regular(_n, _x))
            for _x in _d
        ]
        _ax.plot(_d, _e, label=f"{_nombre} ($N={_n}$): mín. {min(_e):.3f} eV")
    _ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    _ax.set_xlabel("distancia entre vecinos (Å)")
    _ax.set_ylabel("energía por átomo (eV)")
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El mismo esquema de velocity Verlet, con fuerzas analíticas del potencial,
    mueve un tetraedro de K perturbado. Como en el caso de Lennard-Jones, la
    energía total es la comprobación: la cinética y la potencial se
    intercambian y la suma permanece constante. Unidades: eV, Å y masa atómica;
    el paso es 0.5 en unidades de 10.18 fs.
    """)
    return


@app.cell
def _(np, plt, tight_binding):
    _rng = np.random.default_rng(4)
    _pos = tight_binding.simplex_regular(4, 4.3) + _rng.normal(scale=0.1, size=(4, 3))
    _vel = _rng.normal(scale=0.01, size=(4, 3))
    _vel -= _vel.mean(axis=0)
    _paso = 0.5
    _ke, _pe = tight_binding.integrar(_pos, _vel, dt=_paso, n_pasos=2000)
    _m = slice(None, None, 5)
    _t = (_paso * tight_binding.TIEMPO * np.arange(_ke.size) / 1000)[_m]  # ps
    _fig, _ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    _ax.plot(_t, _ke[_m], label="cinética")
    _ax.plot(_t, (_pe - _pe.mean())[_m], label="potencial (menos su media)")
    _ax.plot(
        _t,
        (_ke + _pe - (_ke + _pe).mean())[_m],
        label="total (menos su media)",
        color="black",
    )
    _ax.set_xlabel("tiempo (ps)")
    _ax.set_ylabel("energía (eV)")
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Referencias

    - L. Verlet, *Computer "Experiments" on Classical Fluids. I.
      Thermodynamical Properties of Lennard-Jones Molecules*, Phys. Rev.
      **159**, 98 (1967). doi:10.1103/PhysRev.159.98
    """)
    return


if __name__ == "__main__":
    app.run()
