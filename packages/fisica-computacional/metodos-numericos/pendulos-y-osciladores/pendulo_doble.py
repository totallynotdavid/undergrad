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

    return md, mo, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Péndulo doble: Euler, RK4 y exponente de Lyapunov

        Dos masas $m_1, m_2$ cuelgan de varillas rígidas de longitud $\ell_1,
        \ell_2$. Los ángulos $(\theta_1,\theta_2)$ medidos desde la vertical y
        sus velocidades $(\omega_1,\omega_2)$ forman el estado
        $y=(\theta_1,\omega_1,\theta_2,\omega_2)$. Con $\delta=\theta_1-\theta_2$,
        las ecuaciones de Lagrange dan

        $$
        \dot\omega_1 =
        \frac{-g(2m_1+m_2)\sin\theta_1 - m_2 g\sin(\theta_1-2\theta_2)
        - 2\sin\delta\,m_2\left(\omega_2^2\ell_2+\omega_1^2\ell_1\cos\delta\right)}
        {\ell_1\left(2m_1+m_2-m_2\cos 2\delta\right)},
        $$

        $$
        \dot\omega_2 =
        \frac{2\sin\delta\left(\omega_1^2\ell_1(m_1+m_2)
        + g(m_1+m_2)\cos\theta_1+\omega_2^2\ell_2 m_2\cos\delta\right)}
        {\ell_2\left(2m_1+m_2-m_2\cos 2\delta\right)}.
        $$

        No hay disipación, así que la energía

        $$
        E = \tfrac12(m_1+m_2)\ell_1^2\omega_1^2 + \tfrac12 m_2\ell_2^2\omega_2^2
        + m_2\ell_1\ell_2\,\omega_1\omega_2\cos\delta
        - (m_1+m_2)g\ell_1\cos\theta_1 - m_2 g\ell_2\cos\theta_2
        $$

        es constante. Esa constante es la vara con que medimos a cada integrador.
        """
    )
    return


@app.cell
def _(np):
    M1, M2, L1, L2, G = 1.0, 2.0, 1.0, 1.0, 9.81
    Y0 = np.array([1.2, 0.0, 0.8, 0.0])  # theta1, omega1, theta2, omega2

    def rhs(y):
        theta1, omega1, theta2, omega2 = y
        delta = theta1 - theta2
        den = 2 * M1 + M2 - M2 * np.cos(2 * delta)
        domega1 = (
            -G * (2 * M1 + M2) * np.sin(theta1)
            - M2 * G * np.sin(theta1 - 2 * theta2)
            - 2 * np.sin(delta) * M2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
        ) / (L1 * den)
        domega2 = (
            2
            * np.sin(delta)
            * (
                omega1**2 * L1 * (M1 + M2)
                + G * (M1 + M2) * np.cos(theta1)
                + omega2**2 * L2 * M2 * np.cos(delta)
            )
        ) / (L2 * den)
        return np.array([omega1, domega1, omega2, domega2])

    def energy(y):
        theta1, omega1, theta2, omega2 = np.asarray(y).T
        kinetic = (
            0.5 * (M1 + M2) * L1**2 * omega1**2
            + 0.5 * M2 * L2**2 * omega2**2
            + M2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
        )
        potential = -(M1 + M2) * G * L1 * np.cos(theta1) - M2 * G * L2 * np.cos(theta2)
        return kinetic + potential

    return G, L1, L2, M1, M2, Y0, energy, rhs


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Dos integradores

        Euler explícito avanza con la pendiente al inicio del paso:

        $$
        y_{n+1} = y_n + \Delta t\,f(y_n).
        $$

        RK4 evalúa la pendiente cuatro veces dentro del paso y las combina:

        $$
        y_{n+1} = y_n + \tfrac{\Delta t}{6}\left(k_1+2k_2+2k_3+k_4\right),
        $$

        con $k_1=f(y_n)$, $k_2=f(y_n+\tfrac{\Delta t}{2}k_1)$,
        $k_3=f(y_n+\tfrac{\Delta t}{2}k_2)$ y $k_4=f(y_n+\Delta t\,k_3)$.
        El error global de Euler es $O(\Delta t)$ y el de RK4 es $O(\Delta t^4)$.
        """
    )
    return


@app.cell
def _(np, rhs):
    def step_euler(y, dt):
        return y + dt * rhs(y)

    def step_rk4(y, dt):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(step, y0, dt, nsteps):
        """Devuelve el estado en los instantes 0, dt, ..., nsteps*dt."""
        y = np.empty((nsteps + 1, 4))
        y[0] = y0
        for i in range(nsteps):
            y[i + 1] = step(y[i], dt)
        return y

    return integrate, step_euler, step_rk4


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Trayectoria con RK4

        Con $\Delta t=0.01$ s durante 2500 pasos (25 s) RK4 sigue el movimiento.
        """
    )
    return


@app.cell
def _(L1, L2, Y0, integrate, np, plt, step_rk4):
    dt_fino = 0.01
    pasos_fino = 2500
    tray_rk4 = integrate(step_rk4, Y0, dt_fino, pasos_fino)
    t_fino = np.arange(pasos_fino + 1) * dt_fino

    _x1 = L1 * np.sin(tray_rk4[:, 0])
    _y1 = -L1 * np.cos(tray_rk4[:, 0])
    _x2 = _x1 + L2 * np.sin(tray_rk4[:, 2])
    _y2 = _y1 - L2 * np.cos(tray_rk4[:, 2])

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5))
    _ax[0].plot(_x2, _y2, lw=0.8, color="tab:red")
    _ax[0].plot([0, _x1[-1], _x2[-1]], [0, _y1[-1], _y2[-1]], "o-", color="k")
    _ax[0].set_aspect("equal", "box")
    _ax[0].set_title("Camino de la masa 2 y configuración final")
    _ax[0].set_xlabel("x [m]")
    _ax[0].set_ylabel("y [m]")
    _ax[0].grid(True)

    _ax[1].plot(t_fino, tray_rk4[:, 0], label=r"$\theta_1(t)$")
    _ax[1].plot(t_fino, tray_rk4[:, 2], label=r"$\theta_2(t)$")
    _ax[1].set_xlabel("t [s]")
    _ax[1].set_ylabel("ángulo [rad]")
    _ax[1].set_title("Evolución angular (RK4)")
    _ax[1].grid(True)
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return dt_fino, pasos_fino, t_fino, tray_rk4


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Deriva de la energía

        Integramos las mismas condiciones iniciales con tres configuraciones y
        medimos $|E(t)-E(0)|$:

        - RK4 con $\Delta t=0.01$ s y 2500 pasos.
        - RK4 con $\Delta t=0.05$ s y 10 000 pasos (500 s).
        - Euler con $\Delta t=0.01$ s y 2500 pasos.
        """
    )
    return


@app.cell
def _(Y0, energy, integrate, np, pasos_fino, step_euler, step_rk4, t_fino, tray_rk4):
    dt_grueso = 0.05
    pasos_grueso = 10_000
    tray_rk4_grueso = integrate(step_rk4, Y0, dt_grueso, pasos_grueso)
    tray_euler = integrate(step_euler, Y0, t_fino[1], pasos_fino)
    t_grueso = np.arange(pasos_grueso + 1) * dt_grueso

    deriva_rk4 = np.abs(energy(tray_rk4) - energy(Y0))
    deriva_rk4_grueso = np.abs(energy(tray_rk4_grueso) - energy(Y0))
    deriva_euler = np.abs(energy(tray_euler) - energy(Y0))
    return (
        deriva_euler,
        deriva_rk4,
        deriva_rk4_grueso,
        dt_grueso,
        pasos_grueso,
        t_grueso,
        tray_euler,
    )


@app.cell(hide_code=True)
def _(
    deriva_euler,
    deriva_rk4,
    deriva_rk4_grueso,
    dt_fino,
    dt_grueso,
    md,
    pasos_fino,
    pasos_grueso,
):
    md(
        rf"""
        | integrador | $\Delta t$ [s] | pasos | $\max\,\lvert E-E_0\rvert$ [J] |
        |---|---:|---:|---:|
        | RK4 | {dt_fino} | {pasos_fino} | {deriva_rk4.max():.1e} |
        | RK4 | {dt_grueso} | {pasos_grueso} | {deriva_rk4_grueso.max():.1f} |
        | Euler | {dt_fino} | {pasos_fino} | {deriva_euler.max():.0f} |
        """
    )
    return


@app.cell
def _(
    deriva_euler,
    deriva_rk4,
    deriva_rk4_grueso,
    plt,
    t_fino,
    t_grueso,
):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.semilogy(t_fino[1:], deriva_euler[1:], label=r"Euler, $\Delta t=0.01$")
    _ax.semilogy(t_grueso[1:], deriva_rk4_grueso[1:], label=r"RK4, $\Delta t=0.05$")
    _ax.semilogy(t_fino[1:], deriva_rk4[1:], label=r"RK4, $\Delta t=0.01$")
    _ax.set_xlabel("t [s]")
    _ax.set_ylabel(r"$|E(t)-E(0)|$ [J]")
    _ax.set_title("Deriva de la energía total")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Euler suma energía en cada paso, desde el primero: la deriva ya es de
        unos 2 J tras un segundo y supera varias veces $\lvert E_0\rvert$
        antes de los 25 s, así que su trayectoria deja de representar al
        péndulo. RK4 con el paso fino conserva $E$ a $10^{-4}$ J. Con un paso
        cinco veces mayor la deriva llega a varios julios en 500 s: RK4 no es
        simpléctico, el error no queda acotado, y el paso debe elegirse según
        el tiempo que se quiere simular.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Exponente de Lyapunov máximo

        Una trayectoria de referencia $y(t)$ y otra que empieza a
        $d_0=10^{-6}$ de distancia en el espacio de fases evolucionan juntas.
        Cuando la separación $d(t)$ crece como $d_0e^{\lambda t}$, $\lambda>0$ es el
        exponente de Lyapunov máximo. Para que $d$ no sature en el tamaño del
        espacio de fases, cada $n$ pasos se acerca la segunda trayectoria a la
        primera, sobre la misma dirección, hasta la distancia $d_0$. Skokos
        (2010, sec. 5) describe este procedimiento de dos órbitas vecinas como
        la forma original de calcular el exponente máximo; después se
        sustituyó por la evolución de un vector de desviación con las
        ecuaciones variacionales, que admite pasos mayores y no obliga a
        elegir $d_0$. El estimador es

        $$
        \lambda(T) = \frac{1}{T}\sum_k \ln\frac{d_k}{d_0},
        $$

        donde $d_k$ es la separación justo antes de cada renormalización.
        Comparamos tres condiciones iniciales en reposo.
        """
    )
    return


@app.cell
def _(np, step_rk4):
    D0 = 1e-6

    def lyapunov(y0, dt=0.01, tiempo=100.0, cada=10):
        """Estimador de Benettin de lambda_max y su valor acumulado en el tiempo."""
        a = np.array(y0, dtype=float)
        b = a + np.array([D0, 0.0, 0.0, 0.0])
        suma = 0.0
        t_est, lam = [], []
        for i in range(1, int(round(tiempo / dt)) + 1):
            a = step_rk4(a, dt)
            b = step_rk4(b, dt)
            if i % cada == 0:
                d = np.linalg.norm(b - a)
                suma += np.log(d / D0)
                b = a + (b - a) * (D0 / d)
                t_est.append(i * dt)
                lam.append(suma / (i * dt))
        return np.array(t_est), np.array(lam)

    def separacion(y0, dt=0.01, tiempo=15.0):
        """Distancia entre dos trayectorias sin renormalizar."""
        a = np.array(y0, dtype=float)
        b = a + np.array([D0, 0.0, 0.0, 0.0])
        t_sep, dist = [0.0], [D0]
        for i in range(1, int(round(tiempo / dt)) + 1):
            a = step_rk4(a, dt)
            b = step_rk4(b, dt)
            t_sep.append(i * dt)
            dist.append(np.linalg.norm(b - a))
        return np.array(t_sep), np.array(dist)

    return lyapunov, separacion


@app.cell
def _(Y0, lyapunov, np):
    casos = {
        "ángulos pequeños (0.1, 0.1)": np.array([0.1, 0.0, 0.1, 0.0]),
        "moderado (1.2, 0.8)": Y0,
        "grande (2.5, 2.0)": np.array([2.5, 0.0, 2.0, 0.0]),
    }
    lambdas = {nombre: lyapunov(y0) for nombre, y0 in casos.items()}
    return casos, lambdas


@app.cell(hide_code=True)
def _(lambdas, md):
    _filas = [f"| {nombre} | {lam[-1]:.2f} |" for nombre, (_, lam) in lambdas.items()]
    md(
        "\n".join(
            [
                r"| $(\theta_1(0),\theta_2(0))$ [rad] | $\lambda_{\max}$ [1/s], $T=100$ s |",
                "|---|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell
def _(casos, lambdas, np, plt, separacion):
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5))

    for _nombre, (_t, _lam) in lambdas.items():
        _ax[0].plot(_t, _lam, label=_nombre)
    _ax[0].axhline(0.0, color="k", lw=0.5)
    _ax[0].set_xlabel("T [s]")
    _ax[0].set_ylabel(r"$\lambda(T)$ [1/s]")
    _ax[0].set_title("Convergencia del estimador de Benettin")
    _ax[0].grid(True)
    _ax[0].legend()

    _t, _d = separacion(casos["grande (2.5, 2.0)"])
    _lam_grande = lambdas["grande (2.5, 2.0)"][1][-1]
    _ax[1].semilogy(_t, _d, label=r"$d(t)$ sin renormalizar")
    _ax[1].semilogy(
        _t,
        1e-6 * np.exp(_lam_grande * _t),
        "--",
        label=rf"$d_0e^{{\lambda t}}$, $\lambda={_lam_grande:.2f}$",
    )
    _ax[1].set_xlabel("t [s]")
    _ax[1].set_ylabel("distancia en el espacio de fases")
    _ax[1].set_title(r"Separación de dos trayectorias a $10^{-6}$")
    _ax[1].grid(True, which="both", alpha=0.4)
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Para ángulos grandes $\lambda>0$: una diferencia de una millonésima en
        la condición inicial alcanza el tamaño del espacio de fases en pocos
        segundos, y el tiempo en que se puede predecir la trayectoria crece
        sólo como $\ln(1/d_0)/\lambda$. La gráfica de la derecha muestra que la
        separación sin renormalizar sigue esa exponencial hasta saturar.

        Para ángulos pequeños el movimiento es casi lineal y el estimador cae
        hacia cero al crecer $T$. En el caso moderado también decrece con $T$,
        lo que es consistente con $\lambda=0$ (movimiento regular), pero un
        estimador de tiempo finito no lo demuestra.

        ## Referencia

        Ch. Skokos, «The Lyapunov characteristic exponents and their
        computation» (2010), doi:10.1007/978-3-642-04458-8_2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
