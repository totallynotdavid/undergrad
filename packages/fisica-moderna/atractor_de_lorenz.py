import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atractor de Lorenz

    $$
    \dot x = \sigma (y - x),\qquad
    \dot y = x(\rho - z) - y,\qquad
    \dot z = xy - \beta z.
    $$

    Con $\sigma = 10$, $\beta = 8/3$ y $\rho = 28$ el sistema es caótico. El
    cuaderno integra con Runge–Kutta de cuarto orden (RK4) escrito con NumPy y
    calcula tres cosas sobre el atractor: el mayor exponente de Lyapunov, el
    mapa de Lorenz y el diagrama de bifurcación en $\rho$.
    """)
    return


@app.cell
def _(mo):
    sigma = mo.ui.slider(0.0, 30.0, value=10.0, step=0.1, label="sigma")
    rho = mo.ui.slider(0.0, 60.0, value=28.0, step=0.1, label="rho")
    beta = mo.ui.slider(0.5, 5.0, value=8.0 / 3.0, step=0.01, label="beta")
    t_end = mo.ui.slider(10, 120, value=60, step=1, label="t_fin")
    return beta, rho, sigma, t_end


@app.cell
def _(np):
    def lorenz(s, sigma, rho, beta):
        """Evalúa el campo para uno o varios estados."""
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def rk4(s, dt, p):
        k1 = lorenz(s, *p)
        k2 = lorenz(s + dt / 2 * k1, *p)
        k3 = lorenz(s + dt / 2 * k2, *p)
        k4 = lorenz(s + dt * k3, *p)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrar(s0, dt, pasos, p):
        """Devuelve todos los estados, incluido el inicial."""
        s = np.asarray(s0, dtype=np.float64)
        historia = np.empty((pasos + 1, *s.shape))
        historia[0] = s
        for i in range(pasos):
            s = rk4(s, dt, p)
            historia[i + 1] = s
        return historia

    return integrar, lorenz, rk4


@app.cell
def _(beta, integrar, rho, sigma, t_end):
    DT = 0.01
    ESTADO_INICIAL = (0.1, 0.0, 0.0)
    parametros = (sigma.value, rho.value, beta.value)
    puntos = integrar(ESTADO_INICIAL, DT, round(t_end.value / DT), parametros)
    return DT, ESTADO_INICIAL, parametros, puntos


@app.cell
def _(plt, puntos):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.plot(
        puntos[:, 0],
        puntos[:, 1],
        puntos[:, 2],
        color="k",
        alpha=0.7,
        linewidth=0.9,
        antialiased=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    return (fig,)


@app.cell
def _(beta, fig, mo, rho, sigma, t_end):
    mo.hstack(
        [
            mo.vstack([sigma, rho, beta, t_end], align="stretch"),
            fig,
        ],
        widths=[1, 2],
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Orden del integrador

    RK4 tiene error global $O(\Delta t^4)$: en el régimen asintótico, al reducir
    el paso a la mitad el error en $t = 4$ cae $2^4 = 16$ veces. Se compara con
    una solución de referencia de paso $0.0005$. Con pasos grandes el error aún
    no está en ese régimen y el cociente es mayor.
    """)
    return


@app.cell
def _(ESTADO_INICIAL, integrar, mo, np, parametros):
    def estado_final(dt, t=4.0):
        return integrar(ESTADO_INICIAL, dt, round(t / dt), parametros)[-1]

    _referencia = estado_final(0.0005)
    _pasos = [0.02, 0.01, 0.005, 0.0025]
    _errores = [np.linalg.norm(estado_final(_dt) - _referencia) for _dt in _pasos]
    _filas = [f"| {_pasos[0]} | {_errores[0]:.3e} | |"]
    _filas += [
        f"| {_dt} | {_e:.3e} | {_e0 / _e:.1f} |"
        for _dt, _e0, _e in zip(_pasos[1:], _errores, _errores[1:])
    ]
    _tabla = "\n".join(_filas)
    mo.md(
        f"""
    | Δt | error en t = 4 | cociente con el paso anterior |
    | ---: | ---: | ---: |
    {_tabla}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mayor exponente de Lyapunov

    Dos trayectorias que empiezan a una distancia $d_0$ se separan como
    $d(t) \approx d_0\,e^{\lambda_1 t}$ mientras $d$ es pequeña. En el atractor
    $d$ satura en su tamaño, de modo que la separación se renormaliza: cada
    intervalo $\tau$ se mide el factor $d/d_0$, se acumula $\ln(d/d_0)$ y se
    devuelve la segunda trayectoria a distancia $d_0$ de la primera, en la
    misma dirección. Entonces

    $$
    \lambda_1 \approx \frac{1}{N\tau}\sum_{k=1}^{N} \ln\frac{d_k}{d_0}.
    $$

    Es el algoritmo del mayor exponente de Lyapunov que describe Skokos
    (*Lect. Notes Phys.* 790, 2010, doi 10.1007/978-3-642-04458-8_2, versión
    arXiv:0811.0882): el promedio de los coeficientes de expansión locales
    $\ln\alpha_k/\tau$, aquí con dos órbitas vecinas en lugar de las ecuaciones
    variacionales, alternativa que el mismo texto menciona.

    Se parte de 32 puntos del atractor (la condición inicial evoluciona antes
    30 unidades de tiempo para llegar a él) con $d_0 = 10^{-8}$,
    $\tau = 0.5$ y $N = 400$ (200 unidades de tiempo).
    """)
    return


@app.cell
def _(DT, integrar, np, parametros, rk4):
    D0 = 1e-8
    TAU = 0.5
    BLOQUES = 400

    rng = np.random.default_rng(0)
    semillas = rng.uniform([-15.0, -20.0, 5.0], [15.0, 20.0, 40.0], size=(32, 3)).T
    en_atractor = integrar(semillas, DT, 3000, parametros)[-1]

    def lyapunov(a, p, dt=DT):
        """Devuelve el exponente acumulado tras cada bloque."""
        direccion = rng.normal(size=a.shape)
        b = a + D0 * direccion / np.linalg.norm(direccion, axis=0)
        suma = np.zeros(a.shape[1])
        acumulado = np.empty((BLOQUES, a.shape[1]))
        for k in range(BLOQUES):
            for _ in range(round(TAU / dt)):
                a = rk4(a, dt, p)
                b = rk4(b, dt, p)
            diferencia = b - a
            d = np.linalg.norm(diferencia, axis=0)
            suma += np.log(d / D0)
            b = a + diferencia * (D0 / d)
            acumulado[k] = suma / ((k + 1) * TAU)
        return acumulado

    acumulado = lyapunov(en_atractor, parametros)
    lambdas = acumulado[-1]
    return BLOQUES, TAU, acumulado, en_atractor, lambdas


@app.cell
def _(BLOQUES, TAU, acumulado, lambdas, mo, np, plt):
    _t = TAU * np.arange(1, BLOQUES + 1)
    fig_lyap, ax_lyap = plt.subplots(figsize=(8, 4))
    ax_lyap.plot(_t, acumulado, color="0.6", linewidth=0.6)
    ax_lyap.plot(_t, acumulado.mean(axis=1), "k", linewidth=2, label="promedio")
    ax_lyap.axhline(0.0, color="r", linewidth=0.8)
    ax_lyap.set_ylim(-0.5, 2.5)
    ax_lyap.set_xlabel("Tiempo t")
    ax_lyap.set_ylabel(r"$\lambda_1$ acumulado")
    ax_lyap.set_title("Mayor exponente de Lyapunov (32 pares de trayectorias)")
    ax_lyap.legend()
    ax_lyap.grid(True, alpha=0.3)
    fig_lyap.tight_layout()

    _media = lambdas.mean()
    _error = lambdas.std(ddof=1) / np.sqrt(lambdas.size)
    mo.vstack(
        [
            mo.md(
                f"""
    $\\lambda_1$ = {_media:.3f} ± {_error:.3f} (error estándar de los 32 pares;
    desviación entre pares {lambdas.std(ddof=1):.3f}). Un valor positivo indica
    dependencia sensible de las condiciones iniciales.
    """
            ),
            fig_lyap,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Separación sin renormalizar

    Un solo par de trayectorias, sin renormalizar, muestra las dos fases: un
    crecimiento exponencial y la saturación en el tamaño del atractor. La recta
    es $d_0 e^{\lambda_1 t}$ con el $\lambda_1$ de arriba.
    """)
    return


@app.cell
def _(D0, DT, en_atractor, integrar, lambdas, np, parametros, plt):
    _a0 = en_atractor[:, 0]
    _par = integrar(np.stack([_a0, _a0 + [D0, 0.0, 0.0]], axis=1), DT, 4000, parametros)
    _d = np.linalg.norm(_par[:, :, 1] - _par[:, :, 0], axis=1)
    _t = DT * np.arange(_d.size)

    fig_sep, ax_sep = plt.subplots(figsize=(8, 4))
    ax_sep.semilogy(_t, _d, "k", linewidth=1.2, label="distancia entre trayectorias")
    ax_sep.semilogy(
        _t, D0 * np.exp(lambdas.mean() * _t), "r--", label=r"$d_0 e^{\lambda_1 t}$"
    )
    ax_sep.set_ylim(1e-9, 1e3)
    ax_sep.set_xlabel("Tiempo t")
    ax_sep.set_ylabel("d(t)")
    ax_sep.legend()
    ax_sep.grid(True, which="both", alpha=0.3)
    fig_sep.tight_layout()
    fig_sep
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapa de Lorenz

    Si $z_n$ es el $n$-ésimo máximo local de $z(t)$, el gráfico de $z_{n+1}$
    frente a $z_n$ cae casi sobre una curva con un solo pico. Esa curva es un
    mapa unidimensional: para la sucesión de máximos, el flujo tridimensional
    se reduce a una función de una variable. Guckenheimer y Williams
    (*Publ. Math. IHÉS* 50, 1979, doi 10.1007/bf02684769) hacen lo mismo con su
    modelo geométrico del atractor: la sección de Poincaré se reduce a un mapa
    $f$ de un intervalo en sí mismo, y su dinámica simbólica determina la del
    atractor. Aquí el mapa se obtiene numéricamente de las ecuaciones de
    Lorenz. Los máximos se refinan con una parábola por los tres puntos
    alrededor de cada uno.
    """)
    return


@app.cell
def _(DT, en_atractor, integrar, np, parametros):
    _historia = integrar(en_atractor, DT, 10_000, parametros)[:, 2, :]
    _z0, _z1, _z2 = _historia[:-2], _historia[1:-1], _historia[2:]
    _es_maximo = (_z1 > _z0) & (_z1 >= _z2)
    # Refina cada máximo con la parábola que pasa por sus tres puntos vecinos.
    _vertice = _z1 + (_z2 - _z0) ** 2 / (8 * (2 * _z1 - _z0 - _z2))

    zn, zn1 = [], []
    for _j in range(_historia.shape[1]):
        _maximos = _vertice[_es_maximo[:, _j], _j]
        zn.append(_maximos[:-1])
        zn1.append(_maximos[1:])
    zn = np.concatenate(zn)
    zn1 = np.concatenate(zn1)
    return zn, zn1


@app.cell
def _(mo, np, plt, zn, zn1):
    fig_mapa, ax_mapa = plt.subplots(figsize=(6, 6))
    ax_mapa.plot(zn, zn1, ".", markersize=2, color="k")
    _rango = np.array([zn.min(), zn.max()])
    ax_mapa.plot(_rango, _rango, "r--", linewidth=1, label=r"$z_{n+1} = z_n$")
    ax_mapa.set_xlabel("$z_n$")
    ax_mapa.set_ylabel("$z_{n+1}$")
    ax_mapa.set_title("Mapa de Lorenz")
    ax_mapa.set_aspect("equal")
    ax_mapa.legend()
    ax_mapa.grid(True, alpha=0.3)
    fig_mapa.tight_layout()

    # El punto fijo de la rama derecha se interpola donde z_{n+1} - z_n cambia
    # de signo. La pendiente se ajusta con los puntos situados a menos de 0.5.
    _critico = zn[np.argmax(zn1)]
    _der = zn > _critico
    _orden = np.argsort(zn[_der])
    _x, _y = zn[_der][_orden], zn1[_der][_orden]
    _i = np.nonzero(np.diff(np.sign(_y - _x)))[0][0]
    _fijo = _x[_i] - (_y[_i] - _x[_i]) * (_x[_i + 1] - _x[_i]) / (
        (_y[_i + 1] - _x[_i + 1]) - (_y[_i] - _x[_i])
    )
    _cerca = np.abs(_x - _fijo) < 0.5
    _pendiente = np.polyfit(_x[_cerca], _y[_cerca], 1)[0]
    mo.vstack(
        [
            mo.md(
                f"""
    {zn.size} pares de máximos. La curva tiene un pico agudo en $z_n$ ≈
    {_critico:.1f} y cruza la diagonal en $z^*$ ≈ {_fijo:.1f}, donde su
    pendiente ajustada es {_pendiente:.2f}. Como $|f'(z^*)| > 1$, el punto fijo
    es inestable: las sucesiones de máximos vecinas se separan en cada paso.
    """
            ),
            fig_mapa,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagrama de bifurcación en $\rho$

    Para cada $\rho$ se integra desde las cercanías del punto fijo
    $(\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho - 1)$, se descartan
    30 unidades de tiempo y se registran los máximos de $z$ en las 60
    siguientes. Un punto fijo estable aparece como un solo punto, una órbita
    periódica como pocos puntos y el caos como una nube.
    """)
    return


@app.cell
def _(DT, beta, np, rk4, sigma):
    _rhos = np.linspace(20.0, 230.0, 700)
    _semilla = np.sqrt(beta.value * (_rhos - 1))
    _s = np.array([_semilla, _semilla, _rhos - 1]) + 0.1
    _p = (sigma.value, _rhos, beta.value)

    for _ in range(3000):
        _s = rk4(_s, DT, _p)
    _z = np.empty((6000, _rhos.size))
    for _i in range(6000):
        _s = rk4(_s, DT, _p)
        _z[_i] = _s[2]

    _es_maximo = (_z[1:-1] > _z[:-2]) & (_z[1:-1] >= _z[2:])
    _i, _j = np.nonzero(_es_maximo)
    rho_maximos = _rhos[_j]
    z_maximos = _z[1:-1][_i, _j]
    return rho_maximos, z_maximos


@app.cell
def _(beta, plt, rho_maximos, sigma, z_maximos):
    fig_bif, ax_bif = plt.subplots(figsize=(9, 5))
    ax_bif.plot(rho_maximos, z_maximos, ",", color="k")
    ax_bif.axvline(28.0, color="C0", linewidth=0.8, label=r"$\rho = 28$")

    # La bifurcación de Hopf es válida para sigma > beta + 1.
    _s, _b = sigma.value, beta.value
    _rho_hopf = _s * (_s + _b + 3) / (_s - _b - 1) if _s > _b + 1 else None
    if _rho_hopf is not None:
        ax_bif.axvline(
            _rho_hopf,
            color="C3",
            linewidth=0.8,
            label=rf"Hopf $\rho_H$ = {_rho_hopf:.2f}",
        )
    ax_bif.set_xlabel(r"$\rho$")
    ax_bif.set_ylabel(r"máximos de $z$")
    ax_bif.set_title("Diagrama de bifurcación del sistema de Lorenz")
    ax_bif.legend(loc="lower right")
    ax_bif.grid(True, alpha=0.3)
    fig_bif.tight_layout()
    fig_bif
    return


if __name__ == "__main__":
    app.run()
