import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo

    return dedent, mo


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            """
    # Volumen de la hiperesfera en $n$ dimensiones

    Objetivo: estimar numéricamente el volumen de la bola unitaria
    $$B_n(1)=\\{x\\in\\mathbb{R}^n:\\|x\\|\\le 1\\}$$
    con Monte Carlo por rechazo en el cubo $[-1,1]^n$, compararlo con la fórmula
    exacta y entender por qué el método falla cuando $n$ crece.

    Los resultados de esta página se calculan al exportar el cuaderno.
    """
        )
    )
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            """
    ## Resultado analítico

    La integral gaussiana en una dimensión es:
    $$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx = \\sqrt{\\pi}.$$

    En $n$ dimensiones:
    $$\\int_{\\mathbb{R}^n} e^{-\\|x\\|^2}\\,d^n x = (\\sqrt{\\pi})^n = \\pi^{n/2}.$$

    En coordenadas hiperesféricas:
    $$\\int_{\\mathbb{R}^n} e^{-\\|x\\|^2}\\,d^n x
    = S_{n-1}\\int_0^{\\infty} e^{-r^2}r^{n-1}dr
    = S_{n-1}\\,\\frac{1}{2}\\Gamma\\!\\left(\\frac{n}{2}\\right),$$
    de donde
    $$S_{n-1}=\\frac{2\\pi^{n/2}}{\\Gamma(n/2)}.$$

    Integrando el área radialmente, el volumen de radio $R$ es
    $$V_n(R)=\\frac{\\pi^{n/2}}{\\Gamma\\!\\left(\\frac{n}{2}+1\\right)}R^n.$$

    Para $R=1$:
    $$V_n(1)=\\frac{\\pi^{n/2}}{\\Gamma\\!\\left(\\frac{n}{2}+1\\right)}.$$
    """
        )
    )
    return


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    return math, np, plt


@app.cell
def _():
    try:
        import numba as nb
    except ImportError as exc:
        raise ImportError(
            "Este cuaderno necesita numba para el muestreo Monte Carlo. "
            "Instálalo con `uv sync --package mecanica-estadistica`."
        ) from exc

    return (nb,)


@app.cell
def _(nb, np):
    @nb.njit
    def aciertos_acumulados(
        dim: int, puntos_control: np.ndarray, semilla: int
    ) -> np.ndarray:
        """Puntos del cubo [-1, 1]^dim que caen dentro de la bola unitaria.

        Devuelve el número acumulado de aciertos al llegar a cada tamaño de
        `puntos_control` (creciente), con una sola corrida de muestras.
        """
        np.random.seed(semilla)
        salida = np.empty(puntos_control.shape[0], dtype=np.int64)
        aciertos = 0
        hechos = 0
        for c in range(puntos_control.shape[0]):
            for _ in range(puntos_control[c] - hechos):
                r2 = 0.0
                for _d in range(dim):
                    x = np.random.uniform(-1.0, 1.0)
                    r2 += x * x
                    if r2 > 1.0:
                        break
                if r2 <= 1.0:
                    aciertos += 1
            hechos = puntos_control[c]
            salida[c] = aciertos
        return salida

    return (aciertos_acumulados,)


@app.cell
def _(math):
    def volumen_exacto(dim: int) -> float:
        return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)

    def tasa_exacta(dim: int) -> float:
        """Fracción del cubo [-1, 1]^dim que ocupa la bola: V_n / 2^n."""
        return volumen_exacto(dim) / 2.0**dim

    return tasa_exacta, volumen_exacto


@app.cell
def _(aciertos_acumulados, np, tasa_exacta, volumen_exacto):
    N_MUESTRAS = 1_000_000
    DIMENSIONES = np.arange(1, 21)
    SEMILLA = 12345

    aciertos = np.array(
        [
            aciertos_acumulados(int(n), np.array([N_MUESTRAS]), SEMILLA + int(n))[0]
            for n in DIMENSIONES
        ]
    )
    exacto = np.array([volumen_exacto(int(n)) for n in DIMENSIONES])
    tasa = np.array([tasa_exacta(int(n)) for n in DIMENSIONES])
    tasa_mc = aciertos / N_MUESTRAS
    estimado = tasa_mc * 2.0**DIMENSIONES
    # La desviación típica esperada es 2^n sqrt(p (1 - p) / N).
    sigma = 2.0**DIMENSIONES * np.sqrt(tasa * (1.0 - tasa) / N_MUESTRAS)
    error_relativo = np.abs(estimado - exacto) / exacto
    error_esperado = sigma / exacto
    return (
        DIMENSIONES,
        N_MUESTRAS,
        aciertos,
        error_esperado,
        error_relativo,
        estimado,
        exacto,
        sigma,
        tasa,
        tasa_mc,
    )


@app.cell
def _(DIMENSIONES, exacto, math, np):
    n_pico = int(DIMENSIONES[np.argmax(exacto)])
    _n_fino = np.linspace(1.0, 10.0, 90001)
    _v_fino = np.exp(
        _n_fino / 2.0 * math.log(math.pi)
        - np.array([math.lgamma(x / 2 + 1) for x in _n_fino])
    )
    n_pico_continuo = float(_n_fino[np.argmax(_v_fino)])
    return n_pico, n_pico_continuo


@app.cell(hide_code=True)
def _(DIMENSIONES, N_MUESTRAS, dedent, estimado, exacto, mo, n_pico, aciertos, tasa):
    # Las filas llevan la misma sangría que el texto para que dedent la quite.
    _filas = "\n    ".join(
        f"| {n} | {ex:.6f} | {est:.6f} | {int(h)} | {p:.3e} |"
        for n, ex, est, h, p in zip(DIMENSIONES, exacto, estimado, aciertos, tasa)
        if n in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    )
    mo.md(
        dedent(
            f"""
    ## Monte Carlo frente a la fórmula exacta

    Para cada dimensión se sortean $N={N_MUESTRAS:,}$ puntos uniformes en el
    cubo $[-1,1]^n$, de volumen $2^n$. Si $k$ caen dentro de la bola, la
    estimación es $\\hat V_n = 2^n k/N$.

    | $n$ | $V_n$ exacto | $\\hat V_n$ | aciertos $k$ | $V_n/2^n$ |
    | --- | --- | --- | --- | --- |
    {_filas}

    El volumen exacto es máximo en $n={n_pico}$, con
    $V_{{{n_pico}}}=8\\pi^2/15\\approx{exacto[n_pico - 1]:.4f}$. Para $n>5$ la
    bola unitaria pierde volumen y $V_n\\to 0$.
    """
        )
    )
    return


@app.cell
def _(
    DIMENSIONES,
    N_MUESTRAS,
    error_esperado,
    error_relativo,
    estimado,
    exacto,
    n_pico,
    n_pico_continuo,
    plt,
    sigma,
    tasa,
    tasa_mc,
):
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4.5))

    _ax[0].plot(DIMENSIONES, exacto, "k-", label="Exacto")
    _ax[0].errorbar(
        DIMENSIONES,
        estimado,
        yerr=sigma,
        fmt="o",
        color="tab:blue",
        capsize=2,
        label=f"Monte Carlo ($N={N_MUESTRAS:,}$)",
    )
    _ax[0].axvline(n_pico, color="tab:red", ls="--", lw=1)
    _ax[0].annotate(
        f"máximo en $n={n_pico}$\n($n\\approx{n_pico_continuo:.2f}$ continuo)",
        xy=(n_pico, exacto[n_pico - 1]),
        xytext=(n_pico + 5, exacto[n_pico - 1] * 0.8),
        arrowprops={"arrowstyle": "->"},
    )
    _ax[0].set_xlabel("Dimensión $n$")
    _ax[0].set_ylabel("$V_n$")
    _ax[0].set_title("Volumen de la bola unitaria")
    _ax[0].set_xticks(DIMENSIONES[::2])
    _ax[0].grid(alpha=0.3)
    _ax[0].legend(loc="lower left")

    _ax[1].semilogy(DIMENSIONES, tasa, "k-", label="$V_n/2^n$ (exacta)")
    _visibles = tasa_mc > 0
    _ax[1].semilogy(
        DIMENSIONES[_visibles],
        tasa_mc[_visibles],
        "o",
        color="tab:blue",
        label="Medida ($k/N$)",
    )
    _ax[1].axhline(1.0 / N_MUESTRAS, color="tab:red", ls="--", lw=1)
    _ax[1].text(1.5, 1.4 / N_MUESTRAS, "$1/N$: menos de un acierto esperado")
    _ax[1].set_xlabel("Dimensión $n$")
    _ax[1].set_ylabel("Tasa de aceptación")
    _ax[1].set_title("Aceptación del rechazo en el cubo")
    _ax[1].set_xticks(DIMENSIONES[::2])
    _ax[1].grid(alpha=0.3)
    _ax[1].legend()

    # Sin aciertos, la estimación es 0 y el error relativo es 1.
    # Se omite n = 1 porque la bola llena el segmento y no hay error.
    _ax[2].semilogy(
        DIMENSIONES[1:], error_esperado[1:], "k-", label="Esperado $\\sigma/V_n$"
    )
    _ax[2].semilogy(
        DIMENSIONES[1:],
        error_relativo[1:],
        "o",
        color="tab:red",
        label="Medido",
    )
    _ax[2].set_xlabel("Dimensión $n$")
    _ax[2].set_ylabel("Error relativo")
    _ax[2].set_title("Error con $N$ fijo")
    _ax[2].set_xticks(DIMENSIONES[::2])
    _ax[2].grid(alpha=0.3)
    _ax[2].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(DIMENSIONES, N_MUESTRAS, aciertos, dedent, mo, tasa):
    _sin_aciertos = ", ".join(str(int(n)) for n in DIMENSIONES[aciertos == 0])
    _filas = "\n    ".join(
        f"| {n} | {p:.3e} | {(1 - p) / (p * 0.01**2):.3e} |"
        for n, p in zip(DIMENSIONES, tasa)
        if n in (2, 5, 10, 15, 20)
    )
    mo.md(
        dedent(
            f"""
    ## Por qué falla en dimensión alta

    La tasa de aceptación es la fracción del cubo que ocupa la bola,
    $p_n=V_n/2^n$. Con la aproximación de Stirling para $\\Gamma(n/2+1)$,
    $$p_n \\approx \\frac{{1}}{{\\sqrt{{\\pi n}}}}\\left(\\frac{{\\pi e}}{{2n}}\\right)^{{n/2}},$$
    que cae más rápido que cualquier exponencial: el factor entre paréntesis ya
    es menor que 1 para $n\\ge 5$. Un punto uniforme del cubo cumple
    $\\|x\\|^2=\\sum_i x_i^2\\approx n/3$, que supera 1 para $n>3$ y se
    concentra cada vez más alrededor de ese valor: casi todo el volumen del
    cubo está en sus esquinas, fuera de la bola.

    El estimador $\\hat V_n=2^nk/N$ tiene error relativo típico
    $\\sqrt{{(1-p_n)/(p_nN)}}$. Para fijar ese error en $1\\%$ hacen falta
    $N\\approx(1-p_n)/(p_n\\,10^{{-4}})$ puntos:

    | $n$ | $p_n$ | $N$ para error del 1 % |
    | --- | --- | --- |
    {_filas}

    Con $N={N_MUESTRAS:,}$ la corrida no tiene ningún acierto para
    $n={_sin_aciertos}$. El estimador entonces vale $0$ y su error relativo es
    $100\\%$. Aumentar $N$ no arregla el método porque el costo crece como
    $1/p_n$, es decir, superexponencialmente en $n$.

    Esto es una muestra de la *maldición de la dimensionalidad*. Hinrichs,
    Prochno y Ullrich (J. Complexity 50, 2019) demuestran que, para integrar
    funciones suaves sobre dominios como el cubo o las bolas $\\ell_p$, el
    número de evaluaciones de la función que necesita un algoritmo lineal
    determinista en el peor caso crece exponencialmente con la dimensión. Su
    técnica usa que el volumen de un cuerpo convexo isótropo se concentra en
    una capa delgada alrededor de su radio típico. Ese resultado no trata el
    rechazo en el cubo; es el marco general en el que este fallo encaja.
    """
        )
    )
    return


@app.cell
def _(aciertos_acumulados, np, volumen_exacto):
    CONTROL = np.unique(np.logspace(2, 6, 17).astype(np.int64))
    DIMS_CONVERGENCIA = (2, 4, 8, 12)
    convergencia = {}
    for _n in DIMS_CONVERGENCIA:
        _k = aciertos_acumulados(_n, CONTROL, 99 + _n)
        _est = 2.0**_n * _k / CONTROL
        convergencia[_n] = np.abs(_est - volumen_exacto(_n)) / volumen_exacto(_n)
    return CONTROL, DIMS_CONVERGENCIA, convergencia


@app.cell
def _(CONTROL, DIMS_CONVERGENCIA, convergencia, np, plt, tasa_exacta):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _n in DIMS_CONVERGENCIA:
        _ax.loglog(
            CONTROL,
            np.maximum(convergencia[_n], 1e-5),
            marker="o",
            ms=3,
            label=f"$n={_n}$",
        )
        _p = tasa_exacta(_n)
        _ax.loglog(
            CONTROL,
            np.sqrt((1 - _p) / (_p * CONTROL)),
            "--",
            lw=1,
            color=_ax.lines[-1].get_color(),
        )
    _ax.set_xlabel("Número de muestras $N$")
    _ax.set_ylabel("Error relativo")
    _ax.set_title("Convergencia: medida (puntos) y esperada (guiones)")
    _ax.grid(alpha=0.3, which="both")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            """
    En cada dimensión el error decae como $N^{-1/2}$, pero la constante
    $\\sqrt{(1-p_n)/p_n}$ crece con $n$. La curva medida sigue a la esperada
    con las fluctuaciones propias de una sola corrida.
    """
        )
    )
    return


if __name__ == "__main__":
    app.run()
