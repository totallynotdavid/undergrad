import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from animation import figura_animada

    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return figura_animada, go, md, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Integración de Monte Carlo

        En física estadística, un observable es un valor esperado sobre un
        espacio de configuraciones $R$ de dimensión alta,
        $\langle A\rangle = Z^{-1}\int e^{-\beta H(R)}A(R)\,dR$. Una malla
        regular de $n$ nodos por dirección necesita $N=n^d$ puntos, y una regla
        de orden $k$ tiene error $\sim N^{-k/d}$: se degrada con la dimensión
        $d$. Monte Carlo tiene error $\sim N^{-1/2}$ en cualquier $d$.

        Sea $\Omega$ una región de volumen $V_\Omega$ y $X_1,\ldots,X_N$ puntos
        independientes y uniformes en $\Omega$. Entonces

        $$
        I=\int_\Omega f(x)\,dx \approx \hat I_N = V_\Omega\,\frac1N\sum_{i=1}^N f(X_i),
        \qquad
        \sigma_{\hat I}=\frac{V_\Omega}{\sqrt N}\sqrt{\operatorname{Var}f(X)},
        $$

        y $\operatorname{Var}f(X)$ se estima con la varianza muestral
        $s_f^2$ (`ddof=1` en NumPy). El error estándar $V_\Omega s_f/\sqrt N$
        es el error típico de $\hat I_N$.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Tres integrales con valor conocido

        1. $\pi=\displaystyle\iint_{[-1,1]^2}\mathbf 1_{x^2+y^2\le1}\,dx\,dy$,
           con $V_\Omega=4$.
        2. $\pi=\displaystyle\int_0^2\sqrt{4-x^2}\,dx$ (un cuarto de círculo de
           radio 2), con $V_\Omega=2$.
        3. $1=\displaystyle\int_0^1\!\!\int_0^1 9x^2y^2\,dx\,dy$, con
           $V_\Omega=1$.
        """
    )
    return


@app.cell
def _(np):
    def muestra_pi(rng, n):
        puntos = rng.uniform(-1.0, 1.0, size=(n, 2))
        return 4.0, (np.sum(puntos**2, axis=1) <= 1.0).astype(float)

    def muestra_1d(rng, n):
        x = rng.uniform(0.0, 2.0, size=n)
        return 2.0, np.sqrt(4.0 - x**2)

    def muestra_2d(rng, n):
        x, y = rng.uniform(0.0, 1.0, size=(2, n))
        return 1.0, 9.0 * x**2 * y**2

    problemas = {
        "π (indicadora)": (muestra_pi, np.pi),
        "π (1D)": (muestra_1d, np.pi),
        "integral 2D": (muestra_2d, 1.0),
    }

    def estimador(muestra, rng, n):
        """Devuelve la estimación de la integral y su error estándar."""
        volumen, valores = muestra(rng, n)
        return volumen * valores.mean(), volumen * valores.std(ddof=1) / np.sqrt(n)

    return estimador, problemas


@app.cell
def _(estimador, md, np, problemas):
    N_TABLA = 10_000
    _rng = np.random.default_rng(42)
    _filas = []
    for _nombre, (_muestra, _exacto) in problemas.items():
        _estimacion, _se = estimador(_muestra, _rng, N_TABLA)
        _filas.append(
            f"| {_nombre} | {_estimacion:.5f} | {_exacto:.5f} "
            f"| {abs(_estimacion - _exacto):.5f} | {_se:.5f} |"
        )
    md(
        "\n".join(
            [
                f"$N={N_TABLA}$, semilla 42.",
                "",
                "| integral | estimación | valor exacto | error absoluto | error estándar |",
                "|---|---:|---:|---:|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        La animación sigue la primera integral. Cada cuadro agrega puntos
        uniformes en el cuadrado: azules si caen en el círculo, rojos si no.
        A la derecha, $\hat I_N = 4\cdot(\text{fracción dentro})$ oscila y se
        acerca a $\pi$ dentro de la banda $\pi\pm2\sigma_{\hat I}$, que se
        estrecha como $N^{-1/2}$.
        """
    )
    return


@app.cell
def _(figura_animada, go, np):
    _N_MAX = 5000
    _rng = np.random.default_rng(3)
    _puntos = np.round(_rng.uniform(-1.0, 1.0, size=(_N_MAX, 2)), 3)
    _dentro = (np.sum(_puntos**2, axis=1) <= 1.0).astype(int)
    _estimacion = 4.0 * np.cumsum(_dentro) / np.arange(1, _N_MAX + 1)
    _cortes = np.unique(np.round(np.geomspace(10, _N_MAX, 30)).astype(int))
    _marcas = np.unique(np.round(np.geomspace(10, _N_MAX, 200)).astype(int))
    _p = np.pi / 4
    _sigma = 4.0 * np.sqrt(_p * (1 - _p) / _marcas)

    # Cada lote de puntos es una traza y los cuadros sólo cambian su
    # opacidad: los puntos viajan una vez en la página, no en cada cuadro.
    _lotes = [
        go.Scatter(
            x=_puntos[_a:_b, 0],
            y=_puntos[_a:_b, 1],
            mode="markers",
            marker={
                "size": 4,
                "opacity": int(_b == _cortes[0]),
                "color": _dentro[_a:_b],
                "colorscale": [[0, "#ef553b"], [1, "#636efa"]],
                "cmin": 0,
                "cmax": 1,
            },
            hoverinfo="skip",
            showlegend=False,
        )
        for _a, _b in zip(np.r_[0, _cortes[:-1]], _cortes, strict=True)
    ]
    _fijas = [
        go.Scatter(
            x=np.r_[_marcas, _marcas[::-1]],
            y=np.round(np.r_[np.pi + 2 * _sigma, (np.pi - 2 * _sigma)[::-1]], 4),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.15)",
            line={"width": 0},
            xaxis="x2",
            yaxis="y2",
            name="π ± 2σ",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[_marcas[0], _N_MAX],
            y=[np.pi, np.pi],
            mode="lines",
            line={"color": "#8a94a6", "dash": "dash"},
            xaxis="x2",
            yaxis="y2",
            name="π",
        ),
    ]

    def _curva(n):
        _k = _marcas[_marcas <= n]
        return {"x": _k, "y": np.round(_estimacion[_k - 1], 4)}

    def _punto(n):
        return {"x": [n], "y": [round(_estimacion[n - 1], 4)]}

    def _rotulo(n):
        return {"text": [f"estimación = {_estimacion[n - 1]:.4f}"]}

    _animadas = [
        go.Scatter(
            **_curva(_cortes[0]),
            mode="lines",
            line={"color": "#636efa"},
            xaxis="x2",
            yaxis="y2",
            name="estimación",
        ),
        go.Scatter(
            **_punto(_cortes[0]),
            mode="markers",
            marker={"size": 9, "color": "#636efa"},
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        ),
        go.Scatter(
            x=[12],
            y=[3.95],
            **_rotulo(_cortes[0]),
            mode="text",
            textposition="middle right",
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        ),
    ]
    _cuadros = [
        [{"marker": {"opacity": int(_j <= _i)}} for _j in range(len(_lotes))]
        + [_curva(_c), _punto(_c), _rotulo(_c)]
        for _i, _c in enumerate(_cortes)
    ]
    _primera = len(_lotes) + len(_fijas)
    figura_animada(
        _lotes + _fijas + _animadas,
        _cuadros,
        _cortes,
        animadas=[*range(len(_lotes)), *range(_primera, _primera + 3)],
        ms_por_cuadro=300,
        formato="{}",
        prefijo="N = ",
        sufijo="",
        title="Estimación de π con puntos al azar",
        shapes=[
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": -1,
                "y0": -1,
                "x1": 1,
                "y1": 1,
                "line": {"color": "#2a3f5f", "width": 1},
            }
        ],
        xaxis={"domain": [0, 0.4], "range": [-1.05, 1.05], "constrain": "domain"},
        yaxis={"range": [-1.05, 1.05], "scaleanchor": "x"},
        xaxis2={
            "domain": [0.5, 1],
            "type": "log",
            "range": [1, np.log10(1.25 * _N_MAX)],
        },
        yaxis2={"anchor": "x2", "range": [2.2, 4.1], "title": {"text": "estimación"}},
        legend={"x": 1, "xanchor": "right", "y": 0.02, "yanchor": "bottom"},
        height=480,
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El error absoluto de una corrida es del tamaño del error estándar,
        pero fluctúa de una corrida a otra.

        ## Convergencia: error $\sim N^{-1/2}$

        Para ver la ley de potencias hay que promediar esa fluctuación.
        Repetimos cada estimación 200 veces con muestras nuevas, para
        $N$ entre $10^2$ y $10^5$, y graficamos la raíz del error cuadrático
        medio (RMS) contra $N$. El ajuste de $\log(\mathrm{RMS})$ contra $\log N$
        da la pendiente.
        """
    )
    return


@app.cell
def _(estimador, np, problemas):
    TAMANOS = np.unique(np.logspace(2, 5, 7).astype(int))
    REPETICIONES = 200

    _rng = np.random.default_rng(2024)
    rms = {}
    error_estandar = {}
    for _nombre, (_muestra, _exacto) in problemas.items():
        _rms, _se = [], []
        for _n in TAMANOS:
            _corridas = np.array(
                [estimador(_muestra, _rng, int(_n)) for _ in range(REPETICIONES)]
            )
            _rms.append(np.sqrt(np.mean((_corridas[:, 0] - _exacto) ** 2)))
            _se.append(_corridas[:, 1].mean())
        rms[_nombre] = np.array(_rms)
        error_estandar[_nombre] = np.array(_se)

    pendientes = {
        nombre: np.polyfit(np.log(TAMANOS), np.log(valores), 1)[0]
        for nombre, valores in rms.items()
    }
    return REPETICIONES, TAMANOS, error_estandar, pendientes, rms


@app.cell
def _(TAMANOS, error_estandar, pendientes, plt, rms):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for _a, _nombre in zip(_ax, rms, strict=True):
        _a.loglog(TAMANOS, rms[_nombre], "o-", label="RMS de 200 corridas")
        _a.loglog(TAMANOS, error_estandar[_nombre], "s--", label="error estándar")
        _a.loglog(
            TAMANOS,
            rms[_nombre][0] * (TAMANOS[0] / TAMANOS) ** 0.5,
            ":",
            color="k",
            label=r"$N^{-1/2}$",
        )
        _a.set_title(f"{_nombre}: pendiente {pendientes[_nombre]:.3f}")
        _a.set_xlabel(r"$N$")
        _a.grid(True, which="both", alpha=0.4)
    _ax[0].set_ylabel("error")
    _ax[0].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md, pendientes):
    _pendientes = ", ".join(f"{p:.2f}" for p in pendientes.values())
    md(
        rf"""
        Las pendientes ajustadas ({_pendientes}) son cercanas a $-1/2$ en
        los tres problemas: el exponente no depende de la dimensión (2, 1 y 2
        variables) ni del integrando. El error estándar calculado desde una
        sola muestra coincide con el RMS medido, así que se puede reportar sin
        repetir la simulación.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Reducción de varianza: muestreo por importancia

        Con $N^{-1/2}$ no se puede mejorar la pendiente, pero sí el factor
        $\sqrt{\operatorname{Var}}$. Si se muestrea con una densidad $p(x)$ en
        lugar de la uniforme,

        $$
        I=\int f(x)\,dx=\int \frac{f(x)}{p(x)}\,p(x)\,dx
        =\mathbb E_p\!\left[\frac{f(X)}{p(X)}\right],
        \qquad
        \hat I_N=\frac1N\sum_{i=1}^N\frac{f(X_i)}{p(X_i)},\ \ X_i\sim p.
        $$

        La varianza se anula si $p\propto f$, y crece si $p$ se aparta de $f$
        o es menor que $f$ donde $f$ es grande. En el caso
        $f=9x^2y^2$ sobre el cuadrado unitario usamos la familia
        $p_a(x,y)=(a+1)^2x^ay^a$, que se muestrea con la inversa de la
        distribución acumulada, $x=u^{1/(a+1)}$. El caso $a=0$ es el muestreo
        uniforme. El peso es $w=f/p_a=\dfrac{9}{(a+1)^2}(xy)^{2-a}$, y

        $$
        \operatorname{Var}_{p_a}(w)=\frac{81}{(a+1)^2(5-a)^2}-1.
        $$

        Esa varianza es 2.24 para $a=0$ y se anula en $a=2$, donde
        $p_a\propto f$ y el peso es constante.
        """
    )
    return


@app.cell
def _(np):
    def importancia(rng, n, a):
        """Estimación de la integral 2D con muestras de p_a y su error estándar."""
        x, y = rng.uniform(size=(2, n)) ** (1.0 / (a + 1.0))
        pesos = 9.0 / (a + 1.0) ** 2 * (x * y) ** (2.0 - a)
        return pesos.mean(), pesos.std(ddof=1) / np.sqrt(n)

    def varianza_teorica(a):
        return 81.0 / ((a + 1.0) ** 2 * (5.0 - a) ** 2) - 1.0

    return importancia, varianza_teorica


@app.cell
def _(importancia, np, varianza_teorica):
    N_IMPORTANCIA = 10_000
    EXPONENTES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    _rng = np.random.default_rng(7)
    sigma_medida = np.array(
        [
            np.std([importancia(_rng, N_IMPORTANCIA, a)[0] for _ in range(200)], ddof=1)
            for a in EXPONENTES
        ]
    )
    return EXPONENTES, N_IMPORTANCIA, sigma_medida


@app.cell
def _(EXPONENTES, N_IMPORTANCIA, np, plt, sigma_medida, varianza_teorica):
    _a_fino = np.linspace(0.0, 4.0, 200)
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.semilogy(
        _a_fino,
        np.sqrt(np.maximum(varianza_teorica(_a_fino), 1e-12) / N_IMPORTANCIA),
        label="teoría",
    )
    _ax.semilogy(
        EXPONENTES[sigma_medida > 0],
        sigma_medida[sigma_medida > 0],
        "o",
        label="desviación de 200 corridas",
    )
    _ax.set_xlabel("exponente $a$ de $p_a$")
    _ax.set_ylabel(rf"desviación del estimador, $N={N_IMPORTANCIA}$")
    _ax.set_title("Muestreo por importancia, $f=9x^2y^2$")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(EXPONENTES, md, sigma_medida):
    _idx = {float(a): i for i, a in enumerate(EXPONENTES)}
    _uniforme = sigma_medida[_idx[0.0]]
    _unidad = sigma_medida[_idx[1.0]]
    md(
        rf"""
        Con muestreo uniforme ($a=0$) la desviación del estimador es
        {_uniforme:.4f}. Con $a=1$ (densidad lineal en cada coordenada) baja a
        {_unidad:.4f}, es decir, {(_uniforme / _unidad) ** 2:.1f} veces menos
        varianza con el mismo $N$. Con $a=2$ el peso es constante y la
        desviación es cero salvo error de redondeo. Más allá de $a=2$ la
        varianza vuelve a subir, porque $p_a$ cae más rápido que $f$ cerca de
        $x=0$. Este caso se eligió porque se conoce $p\propto f$; en un problema
        real $f$ no se conoce de antemano, y el esfuerzo se va en encontrar una
        $p$ que lo aproxime y se pueda muestrear. Esa es la idea detrás de
        Metropolis cuando $p$ es el factor de Boltzmann.
        """
    )
    return


if __name__ == "__main__":
    app.run()
