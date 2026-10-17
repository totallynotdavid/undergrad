import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from inspect import cleandoc
    from itertools import pairwise

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go

    return cleandoc, go, mo, np, pairwise, plt


@app.cell
def _():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            "Este cuaderno necesita numba para ejecutar el muestreo de "
            "Metropolis en tiempo razonable. Instálalo con "
            "`uv sync --package mecanica-estadistica`."
        ) from exc

    return (njit,)


@app.cell(hide_code=True)
def _(cleandoc, mo):
    mo.md(
        cleandoc(r"""
        # Modelo de Ising 3D: muestreo de Metropolis

        Hamiltoniano (sin campo externo), con $J=1$ y $k_B=1$:
        $$
        H(\sigma) = -J\sum_{\langle i,j \rangle} s_i s_j,
        \qquad s_i \in \{-1,+1\}
        $$

        Regla de aceptación de Metropolis para un cambio local con $\Delta E$:
        $$
        P_{\text{aceptar}} = \min\left(1, e^{-\beta \Delta E}\right)
        $$

        Los resultados de esta página se calculan al exportar el cuaderno. Para
        cada tamaño $L$ se simula una red cúbica de $L^3$ espines con condiciones
        de frontera periódicas, a muchas temperaturas, y se estiman las
        cantidades que siguen (con $N=L^3$ y $m=\sum_i s_i/N$):

        $$
        \langle |m| \rangle,\qquad
        \langle e \rangle,\qquad
        \chi = \beta N\left(\langle m^2\rangle - \langle |m|\rangle^2\right),\qquad
        C = \beta^2 N\left(\langle e^2\rangle - \langle e\rangle^2\right),
        $$
        $$
        U_4 = 1 - \frac{\langle m^4\rangle}{3\langle m^2\rangle^2}.
        $$
    """)
    )
    return


@app.cell
def _(njit, np):
    @njit
    def simular(L: int, beta: float, barridos: int, semilla: int):
        """Metropolis secuencial desde la red ordenada.

        Un barrido visita los N sitios una vez, en orden. Devuelve e = E/N y
        m = M/N después de cada barrido, y la red final.
        """
        np.random.seed(semilla)
        spins = np.ones((L, L, L), dtype=np.int8)
        n_sitios = L * L * L
        # Vecinos con frontera periódica, sin módulo dentro del bucle.
        mas = (np.arange(L) + 1) % L
        menos = (np.arange(L) - 1) % L
        # dE solo toma los valores 4, 8, 12 cuando el cambio es desfavorable.
        boltzmann = np.exp(-beta * np.array([0.0, 4.0, 8.0, 12.0]))

        energia = -3 * n_sitios
        magnetizacion = n_sitios
        e = np.empty(barridos, dtype=np.float64)
        m = np.empty(barridos, dtype=np.float64)

        for t in range(barridos):
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        s = spins[i, j, k]
                        vecinos = (
                            spins[mas[i], j, k]
                            + spins[menos[i], j, k]
                            + spins[i, mas[j], k]
                            + spins[i, menos[j], k]
                            + spins[i, j, mas[k]]
                            + spins[i, j, menos[k]]
                        )
                        dE = 2 * s * vecinos
                        if dE <= 0 or np.random.random() < boltzmann[dE // 4]:
                            spins[i, j, k] = -s
                            energia += dE
                            magnetizacion -= 2 * s
            e[t] = energia / n_sitios
            m[t] = magnetizacion / n_sitios

        return e, m, spins

    @njit
    def barrer_temperaturas(
        L: int, temperaturas: np.ndarray, replicas: int, barridos: int, semilla: int
    ):
        n_t = temperaturas.shape[0]
        e = np.empty((n_t, replicas, barridos), dtype=np.float64)
        m = np.empty((n_t, replicas, barridos), dtype=np.float64)
        for a in range(n_t):
            for r in range(replicas):
                e[a, r], m[a, r], _ = simular(
                    L, 1.0 / temperaturas[a], barridos, semilla + 1000 * a + r
                )
        return e, m

    return barrer_temperaturas, simular


@app.cell
def _(np):
    TAMANOS = (4, 6, 8, 10, 12)
    REPLICAS = 6
    BARRIDOS = 4000
    DESCARTE = 800
    SEMILLA = 7

    # Rejilla fina alrededor de la temperatura crítica esperada (~4.5).
    TEMPERATURAS = np.round(
        np.concatenate([[3.0, 3.4, 3.8], np.arange(4.1, 4.95, 0.1), [5.3, 5.8]]), 2
    )
    # Intervalo de temperaturas donde se ajusta el cruce de U4.
    AJUSTE = (4.2, 4.8)
    # Referencias: la primera es el valor del cuaderno; la segunda, Hasenbusch.
    BETA_NOTEBOOK = 0.2217
    BETA_HASENBUSCH = 0.22165463
    U4_HASENBUSCH = 1.0 - 1.60357 / 3.0
    return (
        AJUSTE,
        BARRIDOS,
        BETA_HASENBUSCH,
        BETA_NOTEBOOK,
        DESCARTE,
        REPLICAS,
        SEMILLA,
        TAMANOS,
        TEMPERATURAS,
        U4_HASENBUSCH,
    )


@app.cell
def _(
    BARRIDOS,
    DESCARTE,
    REPLICAS,
    SEMILLA,
    TAMANOS,
    TEMPERATURAS,
    barrer_temperaturas,
    np,
):
    def momentos_por_replica(L: int):
        """Observables (n_T, replicas) para el tamaño L, con burn-in descartado."""
        e, m = barrer_temperaturas(L, TEMPERATURAS, REPLICAS, BARRIDOS, SEMILLA)
        e = e[:, :, DESCARTE:]
        m = m[:, :, DESCARTE:]
        n = L**3
        beta = (1.0 / TEMPERATURAS)[:, None]
        m_abs = np.abs(m).mean(axis=2)
        m2 = (m**2).mean(axis=2)
        m4 = (m**4).mean(axis=2)
        return {
            "|m|": m_abs,
            "e": e.mean(axis=2),
            "chi": beta * n * (m2 - m_abs**2),
            "C": beta**2 * n * e.var(axis=2),
            "U4": 1.0 - m4 / (3.0 * m2**2),
        }

    observables = {L: momentos_por_replica(L) for L in TAMANOS}
    return (observables,)


@app.cell
def _(AJUSTE, TAMANOS, TEMPERATURAS, np, observables, pairwise):
    def cruce(u_pequena: np.ndarray, u_grande: np.ndarray) -> float:
        """Cero de la recta que ajusta U4(L grande) - U4(L pequeña) cerca de Tc."""
        ventana = (AJUSTE[0] <= TEMPERATURAS) & (AJUSTE[1] >= TEMPERATURAS)
        pendiente, ordenada = np.polyfit(
            TEMPERATURAS[ventana], (u_grande - u_pequena)[ventana], 1
        )
        return float(-ordenada / pendiente)

    rng = np.random.default_rng(0)
    cruces = {}
    for _chico, _grande in pairwise(TAMANOS):
        _u_c = observables[_chico]["U4"]
        _u_g = observables[_grande]["U4"]
        _estimado = cruce(_u_c.mean(axis=1), _u_g.mean(axis=1))
        # Bootstrap sobre las réplicas independientes.
        _remuestras = []
        for _ in range(400):
            _idx_c = rng.integers(0, _u_c.shape[1], _u_c.shape[1])
            _idx_g = rng.integers(0, _u_g.shape[1], _u_g.shape[1])
            _remuestras.append(
                cruce(_u_c[:, _idx_c].mean(axis=1), _u_g[:, _idx_g].mean(axis=1))
            )
        cruces[(_chico, _grande)] = (_estimado, float(np.std(_remuestras)))
    return (cruces,)


@app.cell(hide_code=True)
def _(
    AJUSTE,
    BARRIDOS,
    BETA_HASENBUSCH,
    BETA_NOTEBOOK,
    DESCARTE,
    REPLICAS,
    TAMANOS,
    cleandoc,
    cruces,
    mo,
):
    # Las filas llevan la misma sangría que el resto del texto para que
    # cleandoc la quite de todas las líneas.
    _filas = "\n        ".join(
        f"| {a} y {b} | {t:.3f} ± {err:.3f} | {1 / t:.4f} ± {err / t**2:.4f} |"
        for (a, b), (t, err) in cruces.items()
    )
    mo.md(
        cleandoc(
            f"""
        ## Resultados de la simulación

        Parámetros: $L \\in \\{{{", ".join(str(L) for L in TAMANOS)}\\}}$,
        {REPLICAS} réplicas independientes por temperatura, {BARRIDOS} barridos
        por réplica y {DESCARTE} barridos de termalización descartados. Cada
        réplica parte de la red ordenada. Las barras de error son la desviación
        estándar de la media entre réplicas.

        ### Cruce del cumulante de Binder

        Para $T>T_c$ el cumulante $U_4$ decrece con $L$ y para $T<T_c$ crece, de
        modo que las curvas de dos tamaños se cortan cerca de $T_c$. Cada cruce
        es el cero de una recta ajustada a $U_4(L_2)-U_4(L_1)$ en
        ${AJUSTE[0]}\\le T\\le {AJUSTE[1]}$. La incertidumbre es un bootstrap
        sobre las réplicas.

        | Tamaños | $T_\\times$ | $\\beta_\\times = 1/T_\\times$ |
        | --- | --- | --- |
        {_filas}

        Referencias: la aproximación $\\beta_c \\approx {BETA_NOTEBOOK}$
        ($T_c \\approx {1 / BETA_NOTEBOOK:.3f}$) y
        $\\beta_c = {BETA_HASENBUSCH}(8)$ de Hasenbusch
        (Phys. Rev. B 82, 174433, 2010), es decir
        $T_c = {1 / BETA_HASENBUSCH:.4f}$.
        """
        )
    )
    return


@app.cell
def _(
    BETA_HASENBUSCH,
    BETA_NOTEBOOK,
    TAMANOS,
    TEMPERATURAS,
    U4_HASENBUSCH,
    np,
    observables,
    plt,
):
    _paneles = [
        ("|m|", r"$\langle |m| \rangle$"),
        ("e", r"$\langle e \rangle = \langle E/N \rangle$"),
        ("chi", r"$\chi$"),
        ("C", r"$C$"),
        ("U4", r"$U_4$"),
    ]
    _colores = plt.cm.viridis(np.linspace(0.0, 0.9, len(TAMANOS)))

    _fig, _ejes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    _ejes = _ejes.ravel()
    # The sixth axis displays the zoom below.
    for _eje, (_clave, _etiqueta) in zip(_ejes, _paneles, strict=False):
        for _L, _color in zip(TAMANOS, _colores, strict=True):
            _datos = observables[_L][_clave]
            _eje.errorbar(
                TEMPERATURAS,
                _datos.mean(axis=1),
                yerr=_datos.std(axis=1, ddof=1) / np.sqrt(_datos.shape[1]),
                marker="o",
                ms=3,
                lw=1,
                capsize=2,
                color=_color,
                label=f"L={_L}",
            )
        _eje.axvline(1 / BETA_HASENBUSCH, color="k", ls="--", lw=1)
        _eje.axvline(1 / BETA_NOTEBOOK, color="tab:red", ls=":", lw=1)
        _eje.set_ylabel(_etiqueta)
        _eje.grid(alpha=0.3)
        _eje.set_xlabel("$T$ ($J/k_B$)")
    _ejes[4].axhline(U4_HASENBUSCH, color="tab:orange", lw=1)
    _ejes[0].legend()

    _zoom = _ejes[5]
    _ventana = (TEMPERATURAS >= 4.0) & (TEMPERATURAS <= 5.0)
    for _L, _color in zip(TAMANOS, _colores, strict=True):
        _datos = observables[_L]["U4"]
        _zoom.errorbar(
            TEMPERATURAS[_ventana],
            _datos.mean(axis=1)[_ventana],
            yerr=(_datos.std(axis=1, ddof=1) / np.sqrt(_datos.shape[1]))[_ventana],
            marker="o",
            ms=3,
            lw=1,
            capsize=2,
            color=_color,
            label=f"L={_L}",
        )
    _zoom.axvline(1 / BETA_HASENBUSCH, color="k", ls="--", lw=1, label="Hasenbusch")
    _zoom.axvline(1 / BETA_NOTEBOOK, color="tab:red", ls=":", lw=1, label="0.2217")
    _zoom.axhline(U4_HASENBUSCH, color="tab:orange", lw=1, label="$U_4^*$")
    _zoom.set_ylabel(r"$U_4$ (ampliación)")
    _zoom.set_xlabel("$T$ ($J/k_B$)")
    _zoom.grid(alpha=0.3)
    _zoom.legend(fontsize=8)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(BETA_HASENBUSCH, U4_HASENBUSCH, cleandoc, cruces, mo):
    _t_min = min(t for t, _ in cruces.values())
    _t_max = max(t for t, _ in cruces.values())
    mo.md(
        cleandoc(
            rf"""
        Líneas verticales: $T_c=1/\beta_c$ con el $\beta_c$ de Hasenbusch
        (discontinua negra) y con $\beta_c\approx 0.2217$ (punteada roja). La
        línea naranja es $U_4^*=1-1.60357/3 = {U4_HASENBUSCH:.4f}$, el valor
        universal del cumulante en $T_c$ que sale de $U_4^*=1.60357(1)$ en la
        forma $\langle m^4\rangle/\langle m^2\rangle^2$ de Hasenbusch.

        - $\langle |m| \rangle$ pasa de $\approx 1$ a $\approx 0$. La transición se
          afila al crecer $L$.
        - $\chi$ y $C$ tienen un máximo cerca de $T_c$ que crece con $L$. En
          volumen finito ninguno diverge.
        - Los cruces de $U_4$ caen entre ${_t_min:.2f}$ y ${_t_max:.2f}$, por
          debajo de $T_c={1 / BETA_HASENBUSCH:.3f}$, y $U_4$ en $T_c$ queda por
          encima de $U_4^*$ para todo $L$ simulado. Es el efecto de las
          correcciones a la escala de tamaño finito: cerca de $T_c$,
          $U_4$ difiere de $U_4^*$ en un término $\propto L^{{-\omega}}$.
          Hasenbusch halla $\omega=0.832(6)$ y, para el modelo de Ising de
          espín 1/2, una amplitud negativa para la razón de momentos, que
          equivale a un $U_4$ por encima de $U_4^*$. Con $L\le 12$ y pocas
          réplicas, los cruces quedan entre
          ${1 / BETA_HASENBUSCH - _t_max:.2f}$ y
          ${1 / BETA_HASENBUSCH - _t_min:.2f}$ por debajo de $T_c$, y los de los
          tamaños mayores tienen una incertidumbre estadística más grande.
        """
        )
    )
    return


@app.cell
def _(np, simular):
    L_VISTA = 8
    TEMPERATURAS_VISTA = (3.0, 4.0, 4.5, 5.0, 7.0)

    def red_en_equilibrio(temperatura: float) -> np.ndarray:
        return simular(L_VISTA, 1.0 / temperatura, 1500, 11)[2]

    redes = {T: red_en_equilibrio(T) for T in TEMPERATURAS_VISTA}
    return L_VISTA, TEMPERATURAS_VISTA, redes


@app.cell
def _(L_VISTA, TEMPERATURAS_VISTA, go, np, redes):
    _x, _y, _z = np.indices((L_VISTA,) * 3).reshape(3, -1)

    def _traza(T: float, signo: int, nombre: str, color: str):
        _sel = redes[T].ravel() == signo
        return go.Scatter3d(
            x=_x[_sel],
            y=_y[_sel],
            z=_z[_sel],
            mode="markers",
            name=nombre,
            marker={"size": 5, "color": color, "opacity": 0.85},
        )

    def _cuadro(T: float):
        return [
            _traza(T, 1, "up", "crimson"),
            _traza(T, -1, "down", "royalblue"),
        ]

    _primero = TEMPERATURAS_VISTA[0]
    fig3d = go.Figure(
        data=_cuadro(_primero),
        frames=[go.Frame(data=_cuadro(T), name=f"{T:g}") for T in TEMPERATURAS_VISTA],
    )
    fig3d.update_layout(
        title=f"Configuración 3D en equilibrio (L={L_VISTA}) para distintas T",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        height=560,
        legend_title_text="Spin",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Reproducir",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1200, "redraw": True}}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "T = "},
                "steps": [
                    {
                        "label": f"{T:g}",
                        "method": "animate",
                        "args": [
                            [f"{T:g}"],
                            {"mode": "immediate", "frame": {"redraw": True}},
                        ],
                    }
                    for T in TEMPERATURAS_VISTA
                ],
            }
        ],
    )
    fig3d
    return


@app.cell(hide_code=True)
def _(L_VISTA, cleandoc, mo):
    mo.md(
        cleandoc(
            f"""
        Cada cuadro es una red de $L={L_VISTA}$ después de 1500 barridos a la
        temperatura indicada en el deslizador. Por debajo de $T_c\\approx 4.51$
        domina un solo signo, cerca de $T_c$ aparecen dominios de ambos signos y
        por encima los espines son casi independientes.

        Notas:
        - Se usan condiciones de frontera periódicas.
        - El tiempo de Monte Carlo se mide en barridos completos: un intento de
          cambio por sitio, recorriendo la red en orden.
        - Con $L\\le 12$ las curvas tienen efectos de tamaño finito marcados. La
          comparación con $T_c$ del límite termodinámico es cualitativa.
        """
        )
    )
    return


if __name__ == "__main__":
    app.run()
