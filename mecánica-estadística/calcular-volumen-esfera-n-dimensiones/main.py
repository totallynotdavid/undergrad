import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            """
    # Volumen de la hiperesfera en $n$ dimensiones

    Objetivo: estimar numéricamente el volumen de la bola unitaria
    $$B_n(1)=\\{x\\in\\mathbb{R}^n:\\|x\\|\\le 1\\}$$
    y compararlo con la fórmula exacta.
    """
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
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
    import numba as nb
    import numpy as np

    return math, nb, np, plt


@app.cell
def _(nb, np):
    @nb.njit
    def volumen_monte_carlo(dim: int, nsample: int) -> float:
        hits = 0
        for _ in range(nsample):
            x = np.random.uniform(-1.0, 1.0, dim)
            if np.sum(x * x) <= 1.0:
                hits += 1
        return (2.0**dim) * hits / nsample

    @nb.njit
    def barrido_muestras(dim: int, samples: np.ndarray) -> np.ndarray:
        out = np.zeros(samples.shape[0], dtype=np.float64)
        for i in range(samples.shape[0]):
            out[i] = volumen_monte_carlo(dim, int(samples[i]))
        return out

    return (barrido_muestras,)


@app.cell
def _(math, np):
    def volumen_exacto(dim: int) -> float:
        return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)

    def malla_muestras(nmax: int) -> np.ndarray:
        base = np.array([100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])
        filtered = base[base <= nmax]
        if filtered.size == 0:
            return np.array([100])
        return filtered

    return malla_muestras, volumen_exacto


@app.cell
def _(mo):
    dim = mo.ui.slider(start=2, stop=12, step=1, value=4, label="Dimensión n")
    nmax = mo.ui.slider(
        start=1000,
        stop=1000000,
        step=1000,
        value=100000,
        label="Muestras máximas",
    )
    return dim, nmax


@app.cell
def _(
    barrido_muestras,
    dim,
    malla_muestras,
    mo,
    nmax,
    np,
    plt,
    volumen_exacto,
):
    n = dim.value
    samples = malla_muestras(nmax.value)
    estimados = barrido_muestras(n, samples)
    exacto = volumen_exacto(n)
    errores = np.abs(estimados - exacto) / exacto

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot(samples, estimados, marker="o", label="Monte Carlo")
    ax[0].axhline(exacto, color="black", linestyle="--", label=f"Exacto = {exacto:.6f}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Número de muestras")
    ax[0].set_ylabel("Volumen")
    ax[0].set_title(f"Volumen de la bola unitaria en n={n}")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(samples, errores, marker="o", color="tab:red")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Número de muestras")
    ax[1].set_ylabel("Error relativo")
    ax[1].set_title("Convergencia del error")
    ax[1].grid(True)

    fig.tight_layout()

    resumen = mo.md(
        rf"""
        **Valor exacto:** $V_{{{n}}}(1)={exacto:.8f}$  
        **Mejor estimación actual:** ${estimados[-1]:.8f}$  
        **Error relativo final:** ${errores[-1]:.3e}$
        """
    )

    mo.vstack([mo.hstack([dim, nmax], justify="start"), resumen, fig])
    return


if __name__ == "__main__":
    app.run()
