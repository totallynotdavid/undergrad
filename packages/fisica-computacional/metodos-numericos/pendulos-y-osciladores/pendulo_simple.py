import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return md, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Péndulo simple: período exacto y energía numérica

        Una masa en el extremo de una varilla rígida de longitud $\ell$ obedece

        $$
        \ddot\theta + \omega_0^2\sin\theta = 0,\qquad \omega_0^2=\frac{g}{\ell},
        $$

        con energía por unidad de masa y de $\ell^2$

        $$
        E = \tfrac12\dot\theta^2 + \omega_0^2\,(1-\cos\theta).
        $$

        Para ángulos pequeños $\sin\theta\approx\theta$ y el período es
        $T_0=2\pi/\omega_0$, independiente de la amplitud. Esta página mide
        cuánto falla esa aproximación y qué integrador numérico conserva mejor
        la energía.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Período exacto

        Integrar $E$ entre el ángulo máximo $\theta_0$ y $0$ da el período como
        una integral elíptica completa de primera especie:

        $$
        T = \frac{4}{\omega_0}\,K\!\left(m\right),
        \qquad
        K(m)=\int_0^{\pi/2}\frac{d\phi}{\sqrt{1-m\sin^2\phi}},
        \qquad
        m=\sin^2\frac{\theta_0}{2}.
        $$

        Para $\theta_0\to0$, $K(0)=\pi/2$ y se recupera $T_0$. Calculamos $K$
        con la media aritmético-geométrica,
        $K(m)=\pi/\left[2\,\mathrm{AGM}(1,\sqrt{1-m})\right]$, que converge en
        pocas iteraciones. Como comprobación independiente, integramos la
        ecuación no lineal con RK4 ($\Delta t=10^{-3}$ s) y medimos el período
        como cuatro veces el tiempo hasta el primer cruce por $\theta=0$.
        """
    )
    return


@app.cell
def _(np):
    G, LARGO = 9.81, 1.0
    OMEGA0 = np.sqrt(G / LARGO)
    T0 = 2 * np.pi / OMEGA0

    def agm(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        for _ in range(40):
            a, b = (a + b) / 2, np.sqrt(a * b)
        return a

    def elliptic_k(m):
        return np.pi / (2 * agm(1.0, np.sqrt(1.0 - m)))

    def periodo_exacto(theta0):
        return 4 / OMEGA0 * elliptic_k(np.sin(theta0 / 2) ** 2)

    def periodo_serie(theta0):
        """Dos primeros términos de la serie en la amplitud."""
        return T0 * (1 + theta0**2 / 16)

    def rhs(y):
        return np.array([y[1], -(OMEGA0**2) * np.sin(y[0])])

    def paso_rk4(y, dt):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def paso_verlet(y, dt):
        """Velocity Verlet: la aceleración sólo depende de la posición."""
        theta, omega = y
        a = -(OMEGA0**2) * np.sin(theta)
        theta_nuevo = theta + omega * dt + 0.5 * a * dt**2
        a_nuevo = -(OMEGA0**2) * np.sin(theta_nuevo)
        return np.array([theta_nuevo, omega + 0.5 * (a + a_nuevo) * dt])

    def periodo_numerico(theta0, dt=1e-3):
        y, t = np.array([theta0, 0.0]), 0.0
        while True:
            y_nuevo = paso_rk4(y, dt)
            if y_nuevo[0] <= 0.0:
                # Interpolación lineal del cruce dentro del último paso.
                return 4 * (t + dt * y[0] / (y[0] - y_nuevo[0]))
            y, t = y_nuevo, t + dt

    def energia(y):
        return 0.5 * y[..., 1] ** 2 + OMEGA0**2 * (1 - np.cos(y[..., 0]))

    return (
        OMEGA0,
        T0,
        energia,
        paso_rk4,
        paso_verlet,
        periodo_exacto,
        periodo_numerico,
        periodo_serie,
    )


@app.cell
def _(T0, md, np, periodo_exacto, periodo_numerico, periodo_serie):
    amplitudes_deg = np.array([5, 10, 30, 60, 90, 120, 150, 170])
    _theta0 = np.radians(amplitudes_deg)
    _exacto = periodo_exacto(_theta0)
    _numerico = np.array([periodo_numerico(a) for a in _theta0])
    _serie = periodo_serie(_theta0)
    _filas = [
        f"| {d} | {e:.5f} | {n:.5f} | {100 * abs(T0 - e) / e:.2f} | {100 * abs(s - e) / e:.3f} |"
        for d, e, n, s in zip(amplitudes_deg, _exacto, _numerico, _serie)
    ]
    md(
        "\n".join(
            [
                rf"$T_0={T0:.5f}$ s para $\ell=1$ m, $g=9.81$ m/s².",
                "",
                r"| $\theta_0$ [°] | $T$ exacto [s] | $T$ RK4 [s] | error de $T_0$ [%] | error de $T_0(1+\theta_0^2/16)$ [%] |",
                "|---:|---:|---:|---:|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell
def _(T0, np, periodo_exacto, periodo_serie, plt):
    _deg = np.linspace(1, 175, 300)
    _theta0 = np.radians(_deg)
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5))

    _ax[0].plot(_deg, periodo_exacto(_theta0) / T0, label="exacto, $K(m)$")
    _ax[0].plot(_deg, periodo_serie(_theta0) / T0, "--", label=r"$1+\theta_0^2/16$")
    _ax[0].axhline(1.0, color="k", ls=":", label="pequeñas oscilaciones")
    _ax[0].set_xlabel(r"$\theta_0$ [°]")
    _ax[0].set_ylabel(r"$T/T_0$")
    _ax[0].set_title("Período en función de la amplitud")
    _ax[0].set_ylim(0.95, 3.0)
    _ax[0].grid(True)
    _ax[0].legend()

    _exacto = periodo_exacto(_theta0)
    _ax[1].semilogy(_deg, np.abs(T0 - _exacto) / _exacto, label=r"$T_0$")
    _ax[1].semilogy(
        _deg,
        np.abs(periodo_serie(_theta0) - _exacto) / _exacto,
        "--",
        label=r"$T_0(1+\theta_0^2/16)$",
    )
    _ax[1].set_xlabel(r"$\theta_0$ [°]")
    _ax[1].set_ylabel("error relativo del período")
    _ax[1].set_title("Error de las aproximaciones")
    _ax[1].grid(True, which="both", alpha=0.4)
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El error de $T_0$ crece como $\theta_0^2/16$: 0.19 % a 10°, 6.8 % a
        60° y 43 % a 150°, donde el período real es 1.76 veces el lineal. El
        período RK4 coincide con el exacto en las cifras de la tabla. La
        corrección $1+\theta_0^2/16$ lo reduce a 0.4 % a 60°, pero a 150°
        todavía falla en 19 %: la serie en $\theta_0$ converge mal cerca de
        $\pi$, y por eso existen aproximaciones analíticas más elaboradas.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Deriva de la energía: velocity Verlet y RK4

        El péndulo no tiene disipación, así que $E$ debe ser constante.
        Comparamos dos integradores con $\theta_0=2$ rad, $\dot\theta_0=0$ y
        $\Delta t=0.1$ s durante 2000 s (unas 700 oscilaciones):

        - Velocity Verlet, de segundo orden y simpléctico: una evaluación de
          la fuerza por paso. Conserva exactamente una energía vecina a $E$,
          así que el error oscila y queda acotado.
        - RK4, de cuarto orden: cuatro evaluaciones por paso. Es más preciso
          en pocos períodos, pero no es simpléctico y su error se acumula.
        """
    )
    return


@app.cell
def _(energia, np, paso_rk4, paso_verlet):
    DT = 0.1
    PASOS = 20_000
    THETA_INICIAL = 2.0

    def trayectoria(paso):
        y = np.empty((PASOS + 1, 2))
        y[0] = (THETA_INICIAL, 0.0)
        for i in range(PASOS):
            y[i + 1] = paso(y[i], DT)
        return y

    _e0 = energia(np.array([THETA_INICIAL, 0.0]))
    t_energia = np.arange(PASOS + 1) * DT
    error_rk4 = (energia(trayectoria(paso_rk4)) - _e0) / _e0
    error_verlet = (energia(trayectoria(paso_verlet)) - _e0) / _e0
    return DT, PASOS, error_rk4, error_verlet, t_energia


@app.cell
def _(error_rk4, error_verlet, np, plt, t_energia):
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5))
    _ax[0].plot(t_energia, error_rk4, label="RK4")
    _ax[0].plot(t_energia, error_verlet, label="velocity Verlet", lw=0.6)
    _ax[0].set_xlabel("t [s]")
    _ax[0].set_ylabel(r"$(E-E_0)/E_0$")
    _ax[0].set_title("Error relativo de la energía")
    _ax[0].grid(True)
    _ax[0].legend()

    _ax[1].loglog(t_energia[1:], np.abs(error_rk4[1:]), label="RK4")
    _ax[1].loglog(
        t_energia[1:], np.abs(error_verlet[1:]), label="velocity Verlet", lw=0.6
    )
    _ax[1].set_xlabel("t [s]")
    _ax[1].set_ylabel(r"$|E-E_0|/E_0$")
    _ax[1].set_title("Escala logarítmica")
    _ax[1].grid(True, which="both", alpha=0.4)
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(DT, PASOS, error_rk4, error_verlet, md, np, t_energia):
    _inicio = t_energia <= 50.0
    md(
        "\n".join(
            [
                rf"$\Delta t={DT}$ s, {PASOS} pasos.",
                "",
                r"| integrador | $\max\lvert E-E_0\rvert/E_0$ hasta 50 s | $\max\lvert E-E_0\rvert/E_0$ hasta 2000 s |",
                "|---|---:|---:|",
                f"| RK4 | {np.abs(error_rk4[_inicio]).max():.1e} | {np.abs(error_rk4).max():.1e} |",
                f"| velocity Verlet | {np.abs(error_verlet[_inicio]).max():.1e} | {np.abs(error_verlet).max():.1e} |",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(error_rk4, error_verlet, md, np, t_energia):
    _cruce = t_energia[np.argmax(np.abs(error_rk4) > np.abs(error_verlet).max())]
    md(
        rf"""
        El error de Verlet oscila y no sobrepasa {np.abs(error_verlet).max():.1%}.
        El de RK4 es mucho menor al principio, pero crece en una sola
        dirección (RK4 disipa energía en este problema) y supera la cota de
        Verlet a los {_cruce:.0f} s. Para simular muchos períodos con un paso
        grueso conviene un integrador simpléctico; para un intervalo corto y
        exacto, RK4. RK4 cuesta cuatro evaluaciones de la fuerza por paso y
        Verlet una.
        """
    )
    return


if __name__ == "__main__":
    app.run()
