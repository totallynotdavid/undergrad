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
    # Volumen de la hiperesfera de n-dimensiones

    Autor: David Duran<br>
    Año: 2023
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derivación

    ### Caso bidimensional

    Partimos del área de la función gaussiana: $e^{-x^2}$. Integremos, por ahora elevemos la integral al cuadrado:

    $$
    \left(\int_{-\infty}^{\infty} e^{-x^2} d x\right)^2=\left(\int_{-\infty}^{\infty} e^{-x_1^2} d x_1\right)\left(\int_{-\infty}^{\infty} e^{-x_2^2} d x_2\right)=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-\left(x_1^2+x_2^2\right)} d x_1 d x_2=\pi
    $$

    Trabajando con coordenadas polares $(r, \theta)$ en vez de cartesianas $\left(x_1, x_2\right)$. Reemplazando con:

    $$
    x_1^2+x_2^2=r^2;\quad x_1=r \sin \theta;\quad x_2=r \cos \theta
    $$

    Obtenemos:

    $$
    \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-\left(x_1^2+x_2^2\right)} d x_1 d x_2=\int_{r=0}^{\infty} \int_{\theta=0}^{2 \pi} e^{-r^2} r d r d \theta = \int_{u=0}^{\infty} \int_{\theta=0}^{2 \pi} e^{-u} \frac{1}{2} d u d \theta=\frac{1}{2} \int_0^{2 \pi} d \theta \int_0^{\infty} e^{-u} d u =\frac{1}{2} \cdot 2 \pi \cdot\left(-e^{-u}\right)_0^{\infty}= \pi
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Por lo tanto: $$\left(\int_{-\infty}^{\infty} e^{-x^2} d x\right)^2 = \pi$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La integral de la gaussiana es $\sqrt{\pi}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Caso tridimensional

    Elevemos la integral al cubo esta vez.

    $$
    \left(\int_{-\infty}^{\infty} e^{-x^2} d x\right)^3=\left(\int_{-\infty}^{\infty} e^{-x^2} d x_1\right)\left(\int_{-\infty}^{\infty} e^{-x_2^2} d x_2\right)\left(\int_{-\infty}^{\infty} e^{-x_1^2} d x_3\right)=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-\left(x_1^2+x_1^2+x_3^2\right)} d x_1 d x_2 d x_3
    $$

    Pasemos de trabajar en coordenadas polares $\left(x_1, x_2, x_3\right)_0$ a coordenadas esféricas $(r, \theta, \varphi)$
    $$
    \left(x_1^2+x_2^2+x_3^2\right)=r^2; \quad d x_1 d x_2 d x_3=r^2 \sin \theta d r d \theta d \varphi=r^2 d r d \Omega_2 \\
    $$
    Reemplazando
    $$
    \iiint e^{-\left(x_1^2+x_2^2+x_3^2\right)} d x_1 d x_2 d x_3=\underbrace{\int e^{-r^2} r^2 d r}_{\frac{\sqrt{\pi}}{4}} \underbrace{\int d \Omega_2}_{4 \pi}=(\sqrt{\pi})^3
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Caso n-dimensional

    $$
    \left(\int_{-\infty}^{\infty} e^{-x^2} d x\right)^n=\int_{-\infty}^{\infty} e^{-x_1^2} d x_1 \cdots \int_{-\infty}^{\infty} e^{-x_n^2} d x_n=\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} e^{-\left(x_1^2+\cdots+x_n^2\right)} d x_1 \cdots d x_n
    $$

    Notemos que: $$x_1^2+\cdots+x_n^2=r^2$$.

    Ahora, reescribiendo la integral:

    $$
    \int_{-\infty}^{\infty} e^{-r^2} r^{n-1} \int_{\text{n-esfera}} d \Omega_{n-1}
    $$

    Veámoslo por partes:

    $$
    \int_{-\infty}^{\infty} e^{-r^2} r^{n-1} =\frac{1}{2} \Gamma\left(\frac{n}{2}\right)
    $$

    Para área de la superficie $S_n$:

    $$
    S_n \cdot \frac{1}{2} \Gamma\left(\frac{n}{2}\right)=\pi^{n / 2} \Rightarrow S_n=\frac{\pi^{n / 2}}{\frac{1}{2} \Gamma\left(\frac{n}{2}\right)} \Rightarrow S_n(r)=\frac{\pi^{n / 2} r^{n-1}}{\frac{1}{2} \Gamma\left(\frac{n}{2}\right)}
    $$

    Para el volumen $V_n$:

    $$
    V_n(r) =\int S_n(r) d r=S_n \int r^{n-1} d r=\frac{S_n r^n}{n}=\frac{\pi^{n / 2} r^n}{\frac{n}{2} \Gamma\left(\frac{n}{2}\right)}=\frac{\pi^{n / 2} r^n}{\Gamma\left(\frac{n}{2}+1\right)}
    $$

    Finalmente, para el caso de una esfera n-dimensional con $r=1$. El volumen será $\frac{\pi^{n / 2}}{\Gamma\left(\frac{n}{2}+1\right)}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cálculo numérico
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import numba as nb

    return nb, np, plt


@app.cell
def _(nb, np):
    @nb.jit(nopython=True)
    def volumen_hiperesfera(dim, Nsample=1000000):
        hits = 0
        for n in range(Nsample):
            punto = np.random.uniform(-1.0, 1.0, dim)
            radio_cuadrado = np.sum(punto**2)
            if radio_cuadrado <= 1.0:
                hits += 1
        return hits * (2.0 ** dim / Nsample)

    return (volumen_hiperesfera,)


@app.cell
def _(nb, np, volumen_hiperesfera):
    @nb.jit(nopython=True)
    def calcular_volumenes(dim, sample_sizes):
        volumes = np.zeros(len(sample_sizes))
        for i, Nsample in enumerate(sample_sizes):
            volumes[i] = volumen_hiperesfera(dim, Nsample)
        return volumes

    return (calcular_volumenes,)


@app.cell
def _(calcular_volumenes, np, plt):
    def plot_volumen(dim, Nsample):
        plt.figure(figsize=(8, 6))
        candidate_samples = np.array([10, 100, 1000, 10000, 100000, 1000000])
        samples = candidate_samples[candidate_samples <= Nsample]
        if len(samples) == 0:
            samples = np.array([10])
        volumes = calcular_volumenes(dim, samples)
        plt.plot(samples, volumes, marker="o")
        plt.xscale("log")
        plt.xlabel("Número de muestras (escala logarítmica)")
        plt.ylabel("Volumen estimado")
        plt.title(f"Volumen de la hiperesfera {dim}-dimensional")
        plt.grid()
        plt.show()

    return (plot_volumen,)


@app.cell
def _(mo):
    dim = mo.ui.slider(start=2, stop=10, step=1, value=4, label="Dimensiones")
    nsample = mo.ui.slider(start=1000, stop=1000000, step=1000, value=100000, label="Muestras máximas")
    return dim, nsample


@app.cell
def _(dim, mo, nsample, plot_volumen):
    chart = plot_volumen(dim.value, nsample.value)
    mo.vstack(
        [
            mo.hstack([dim, nsample], justify="start"),
            chart,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
