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
    # Curva de Planck

    La radiancia espectral de un cuerpo negro a temperatura $T$ es

    $$
    B_\lambda(T) = \frac{2hc^2}{\lambda^5}\,\frac{1}{e^{hc/(\lambda k T)} - 1}.
    $$

    Con $\lambda$ en metros, la fórmula da W/(m²·sr·m). En todo el cuaderno se
    usa **W/(m²·sr·nm)**: la longitud de onda va en nm y el resultado se
    multiplica por $10^{-9}$, de modo que valores, etiquetas y límites de los
    ejes comparten una sola unidad.

    Boyer deduce este espectro solo con termodinámica aplicada al oscilador
    armónico: las dos primeras leyes dan el teorema de desplazamiento de Wien
    (la energía del oscilador es función de $T/\omega$) y la tercera ley, con
    la entropía como función analítica, fija la forma de Planck. A alta
    temperatura la energía del oscilador coincide con el resultado de
    equipartición, que es el límite de Rayleigh–Jeans (Boyer, *Eur. J. Phys.*
    40, 2019, doi 10.1088/1361-6404/aaf45b; se leyó la versión arXiv:1809.09093).
    """)
    return


@app.cell
def _(np):
    # Constantes SI exactas.
    H = 6.62607015e-34  # J s
    C = 299792458.0  # m/s
    K = 1.380649e-23  # J/K
    SIGMA = 5.670374419e-8  # W/(m^2 K^4)
    TEMPERATURAS = [2000, 6000, 10000, 14000]

    def planck(lam_nm, T):
        """Devuelve la radiancia en W/(m^2 sr nm) para longitudes en nm."""
        lam = lam_nm * 1e-9
        return 1e-9 * 2 * H * C**2 / lam**5 / np.expm1(H * C / (lam * K * T))

    def rayleigh_jeans(lam_nm, T):
        lam = lam_nm * 1e-9
        return 1e-9 * 2 * C * K * T / lam**4

    def wien(lam_nm, T):
        lam = lam_nm * 1e-9
        return 1e-9 * 2 * H * C**2 / lam**5 * np.exp(-H * C / (lam * K * T))

    return (
        C,
        H,
        K,
        SIGMA,
        TEMPERATURAS,
        planck,
        rayleigh_jeans,
        wien,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curvas para cuatro temperaturas

    Para $\lambda \to 0$ el exponente desborda a infinito y $B_\lambda$ da cero,
    que es el límite correcto.
    """)
    return


@app.cell
def _(TEMPERATURAS, np, planck, plt):
    lam_nm = np.linspace(1.0, 4000.0, 4000)
    fig_curvas, (ax_lineal, ax_log) = plt.subplots(1, 2, figsize=(12, 5))
    with np.errstate(over="ignore"):
        for _T in TEMPERATURAS:
            for _ax in (ax_lineal, ax_log):
                _ax.plot(lam_nm, planck(lam_nm, _T), linewidth=2, label=f"{_T} K")
    ax_lineal.set_title("Escala lineal", fontsize=13)
    ax_lineal.set_xlim(0, 2000)
    ax_lineal.set_ylim(0, 2.5e6)
    ax_log.set_title("Escala logarítmica en el eje vertical", fontsize=13)
    ax_log.set_yscale("log")
    ax_log.set_xlim(0, 4000)
    ax_log.set_ylim(1, 1e7)
    for _ax in (ax_lineal, ax_log):
        _ax.set_xlabel("Longitud de onda λ (nm)", fontsize=12)
        _ax.grid(True)
        _ax.legend()
    ax_lineal.set_ylabel(
        r"Radiancia $B_\lambda$ $(\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}})$",
        fontsize=12,
    )
    fig_curvas.suptitle("Distribución de Planck", fontsize=15)
    fig_curvas.tight_layout()
    fig_curvas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Límites de Rayleigh–Jeans y de Wien

    Con $x = hc/(\lambda k T)$:

    - $x \ll 1$ (λ grande): $e^x - 1 \approx x$ y
      $B_\lambda \to 2ckT/\lambda^4$ (Rayleigh–Jeans, equipartición clásica).
    - $x \gg 1$ (λ pequeña): $e^x - 1 \approx e^x$ y
      $B_\lambda \to (2hc^2/\lambda^5)\,e^{-x}$ (Wien).

    Las razones $B^{RJ}/B$ y $B^{W}/B$ valen $(e^x-1)/x$ y $1-e^{-x}$.
    """)
    return


@app.cell
def _(np, planck, plt, rayleigh_jeans, wien):
    T_LIMITES = 6000
    lam_log = np.geomspace(10.0, 1e5, 2000)
    with np.errstate(over="ignore"):
        b_planck = planck(lam_log, T_LIMITES)
        b_rj = rayleigh_jeans(lam_log, T_LIMITES)
        b_wien = wien(lam_log, T_LIMITES)

    fig_limites, (ax_ley, ax_razon) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, height_ratios=[2, 1]
    )
    ax_ley.loglog(lam_log, b_planck, "k", linewidth=2.5, label="Planck")
    ax_ley.loglog(lam_log, b_rj, "--", linewidth=1.8, label="Rayleigh–Jeans")
    ax_ley.loglog(lam_log, b_wien, "--", linewidth=1.8, label="Wien")
    ax_ley.set_ylim(1e-6, 1e9)
    ax_ley.set_ylabel(
        r"$B_\lambda$ $(\mathrm{W\,m^{-2}\,sr^{-1}\,nm^{-1}})$", fontsize=12
    )
    ax_ley.set_title(f"Planck y sus límites a {T_LIMITES} K", fontsize=15)
    ax_ley.legend()
    ax_ley.grid(True, which="both", alpha=0.3)

    ax_razon.semilogx(lam_log, b_rj / b_planck, label="Rayleigh–Jeans / Planck")
    ax_razon.semilogx(lam_log, b_wien / b_planck, label="Wien / Planck")
    ax_razon.axhline(1.0, color="k", linewidth=0.8)
    ax_razon.set_ylim(0, 2)
    ax_razon.set_xlabel("Longitud de onda λ (nm)", fontsize=12)
    ax_razon.set_ylabel("razón")
    ax_razon.legend()
    ax_razon.grid(True, which="both", alpha=0.3)
    fig_limites.tight_layout()
    fig_limites
    return T_LIMITES, b_planck, b_rj, b_wien, lam_log


@app.cell
def _(T_LIMITES, b_planck, b_rj, b_wien, lam_log, mo, np):
    # Rayleigh-Jeans sobreestima por menos de 5 % para lambda >= lam_rj.
    # Wien subestima por menos de 5 % para lambda <= lam_w.
    lam_rj = lam_log[np.abs(b_rj / b_planck - 1) < 0.05].min()
    lam_w = lam_log[np.abs(b_wien / b_planck - 1) < 0.05].max()
    mo.md(
        f"""
    A {T_LIMITES} K, Rayleigh–Jeans reproduce a Planck con error menor que 5 %
    para λ ≥ {lam_rj:.0f} nm y Wien para λ ≤ {lam_w:.0f} nm. Entre ambos
    valores ninguno de los dos límites basta y hay que usar la fórmula completa.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### La catástrofe ultravioleta

    Rayleigh–Jeans crece como $\lambda^{-4}$ y su integral diverge al acercarse
    $\lambda$ a cero. Con $M = \pi\int_{\lambda_{\min}}^\infty B_\lambda\,d\lambda$
    (exitancia radiante), el límite clásico no converge y Planck sí:
    """)
    return


@app.cell
def _(C, K, SIGMA, T_LIMITES, mo, np, planck):
    def exitancia(f, lam_min_nm, T, lam_max_nm=1e9):
        """Integra pi por f en W/(m^2) sobre una malla logarítmica."""
        lam = np.geomspace(lam_min_nm, lam_max_nm, 20_001)
        with np.errstate(over="ignore"):
            integrando = f(lam, T) * lam
        return np.pi * np.trapezoid(integrando, np.log(lam))

    def exitancia_rj(lam_min_nm, T):
        # La integral clásica diverge al extender la malla hasta lambda = 0.
        lam = lam_min_nm * 1e-9
        return np.pi * 2 * C * K * T / (3 * lam**3)

    filas = "\n".join(
        f"| {_lam_min} | {exitancia_rj(_lam_min, T_LIMITES):.3e} "
        f"| {exitancia(planck, _lam_min, T_LIMITES):.3e} |"
        for _lam_min in (1000, 100, 10, 1)
    )
    mo.md(
        f"""
    Exitancia a {T_LIMITES} K, con $\\sigma T^4$ = {SIGMA * T_LIMITES**4:.3e} W/m²:

    | λ_min (nm) | Rayleigh–Jeans (W/m²) | Planck (W/m²) |
    | ---: | ---: | ---: |
    {filas}
    """
    )
    return (exitancia,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Desplazamiento de Wien

    Derivando $B_\lambda$ respecto a $\lambda$ el máximo cumple
    $x = 5(1 - e^{-x})$ con $x = hc/(\lambda_{\max} k T)$. De ahí
    $\lambda_{\max} T = b = hc/(k x)$. Aquí se halla el máximo de la curva
    directamente, con una búsqueda de la sección áurea sobre $B_\lambda$, y se
    compara con $b/T$. La forma $B_\lambda = T^5 f(\lambda T)$ es la
    información que aporta el teorema de desplazamiento de Wien, que Boyer
    obtiene de las dos primeras leyes de la termodinámica.
    """)
    return


@app.cell
def _(C, H, K, mo, np, planck):
    def maximo(f, a, b, iteraciones=200):
        """Abscisa del máximo de una función unimodal en [a, b]."""
        razon = (np.sqrt(5) - 1) / 2
        c = b - razon * (b - a)
        d = a + razon * (b - a)
        for _ in range(iteraciones):
            if f(c) > f(d):
                b = d
            else:
                a = c
            c = b - razon * (b - a)
            d = a + razon * (b - a)
        return (a + b) / 2

    # Resuelve x = 5 (1 - exp(-x)) por iteración de punto fijo.
    x_wien = 5.0
    for _ in range(100):
        x_wien = 5 * (1 - np.exp(-x_wien))
    b_wien_mk = H * C / (K * x_wien)

    pasos = []
    for _T in (2000, 6000, 10000, 14000):
        _lam_max = maximo(lambda lam: planck(lam, _T), 10.0, 20000.0)
        pasos.append(
            f"| {_T} | {_lam_max:.2f} | {b_wien_mk / _T * 1e9:.2f} "
            f"| {planck(_lam_max, _T):.4g} |"
        )
    filas_wien = "\n".join(pasos)
    mo.md(
        f"""
    $x$ = {x_wien:.9f}, $b = hc/(kx)$ = {b_wien_mk * 1e3:.9f} × 10⁻³ m·K.

    | T (K) | λ_max numérico (nm) | b/T (nm) | B(λ_max) (W/(m²·sr·nm)) |
    | ---: | ---: | ---: | ---: |
    {filas_wien}
    """
    )
    return (maximo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stefan–Boltzmann

    La exitancia total es $M(T) = \pi\int_0^\infty B_\lambda\,d\lambda = \sigma T^4$.
    Se integra numéricamente con la regla del trapecio sobre una malla
    logarítmica de 1 nm a 1 m (el integrando es $B_\lambda\,\lambda$ respecto a
    $\ln\lambda$) y se compara con $\sigma T^4$, donde
    $\sigma = 2\pi^5 k^4/(15 h^3 c^2)$.
    """)
    return


@app.cell
def _(C, H, K, SIGMA, TEMPERATURAS, exitancia, mo, np, planck, plt):
    sigma_derivada = 2 * np.pi**5 * K**4 / (15 * H**3 * C**2)
    _filas = []
    _T_vals = np.linspace(1000, 15000, 60)
    _m_vals = np.array([exitancia(planck, 1.0, _T, 1e9) for _T in _T_vals])
    for _T in TEMPERATURAS:
        _m = exitancia(planck, 1.0, _T, 1e9)
        _filas.append(
            f"| {_T} | {_m:.6e} | {SIGMA * _T**4:.6e} | {_m / (SIGMA * _T**4) - 1:+.2e} |"
        )
    filas_sb = "\n".join(_filas)

    fig_sb, ax_sb = plt.subplots(figsize=(8, 5))
    ax_sb.loglog(_T_vals, _m_vals, "o", markersize=4, label="integral numérica")
    ax_sb.loglog(_T_vals, SIGMA * _T_vals**4, "k-", label=r"$\sigma T^4$")
    ax_sb.set_xlabel("Temperatura T (K)", fontsize=12)
    ax_sb.set_ylabel(r"Exitancia $M$ $(\mathrm{W\,m^{-2}})$", fontsize=12)
    ax_sb.legend()
    ax_sb.grid(True, which="both", alpha=0.3)
    fig_sb.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"""
    $\\sigma$ desde las constantes = {sigma_derivada:.9e} W/(m²·K⁴); valor
    usado = {SIGMA:.9e}.

    | T (K) | π∫B dλ (W/m²) | σT⁴ (W/m²) | error relativo |
    | ---: | ---: | ---: | ---: |
    {filas_sb}
    """
            ),
            fig_sb,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
