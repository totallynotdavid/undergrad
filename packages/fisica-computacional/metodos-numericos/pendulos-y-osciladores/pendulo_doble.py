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
            r"""
# Péndulo doble: integración RK4

Con coordenadas angulares $(\theta_1,\theta_2)$ y velocidades
$(\omega_1,\omega_2)$, integramos el sistema no lineal clásico del
péndulo doble con RK4.
"""
        )
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(mo):
    theta1_0 = mo.ui.number(
        start=-3.14, stop=3.14, step=0.01, value=1.2, label="theta1(0)"
    )
    omega1_0 = mo.ui.number(
        start=-10.0, stop=10.0, step=0.01, value=0.0, label="omega1(0)"
    )
    theta2_0 = mo.ui.number(
        start=-3.14, stop=3.14, step=0.01, value=0.8, label="theta2(0)"
    )
    omega2_0 = mo.ui.number(
        start=-10.0, stop=10.0, step=0.01, value=0.0, label="omega2(0)"
    )

    m1 = mo.ui.number(start=0.1, stop=10.0, step=0.1, value=1.0, label="m1")
    m2 = mo.ui.number(start=0.1, stop=10.0, step=0.1, value=2.0, label="m2")
    l1 = mo.ui.number(start=0.1, stop=5.0, step=0.1, value=1.0, label="l1")
    l2 = mo.ui.number(start=0.1, stop=5.0, step=0.1, value=1.0, label="l2")
    g = mo.ui.number(start=0.1, stop=20.0, step=0.1, value=9.81, label="g")

    dt = mo.ui.number(start=0.001, stop=0.05, step=0.001, value=0.01, label="dt")
    nsteps = mo.ui.slider(start=200, stop=10000, step=100, value=2500, label="pasos")

    return dt, g, l1, l2, m1, m2, nsteps, omega1_0, omega2_0, theta1_0, theta2_0


@app.cell(hide_code=True)
def _(dt, g, l1, l2, m1, m2, mo, nsteps, omega1_0, omega2_0, theta1_0, theta2_0):
    mo.vstack(
        [
            mo.hstack([m1, m2, l1, l2, g], justify="start"),
            mo.hstack([theta1_0, omega1_0, theta2_0, omega2_0], justify="start"),
            mo.hstack([dt, nsteps], justify="start"),
        ]
    )
    return


@app.cell
def _(np):
    def rhs(theta1, omega1, theta2, omega2, m1, m2, l1, l2, g):
        delta = theta1 - theta2
        den = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        dtheta1 = omega1
        dtheta2 = omega2

        domega1 = (
            -g * (2 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2 * theta2)
            - 2
            * np.sin(delta)
            * m2
            * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * np.cos(delta))
        ) / (l1 * den)

        domega2 = (
            2
            * np.sin(delta)
            * (
                omega1 * omega1 * l1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2 * omega2 * l2 * m2 * np.cos(delta)
            )
        ) / (l2 * den)

        return dtheta1, domega1, dtheta2, domega2

    def integrate_double_pendulum(
        theta1_0, omega1_0, theta2_0, omega2_0, m1, m2, l1, l2, g, dt, nsteps
    ):
        t = np.arange(nsteps + 1) * dt
        theta1 = np.zeros(nsteps + 1)
        omega1 = np.zeros(nsteps + 1)
        theta2 = np.zeros(nsteps + 1)
        omega2 = np.zeros(nsteps + 1)

        theta1[0] = theta1_0
        omega1[0] = omega1_0
        theta2[0] = theta2_0
        omega2[0] = omega2_0

        for i in range(nsteps):
            y1 = (theta1[i], omega1[i], theta2[i], omega2[i])

            k1 = rhs(*y1, m1, m2, l1, l2, g)
            y2 = tuple(y1[j] + 0.5 * dt * k1[j] for j in range(4))
            k2 = rhs(*y2, m1, m2, l1, l2, g)
            y3 = tuple(y1[j] + 0.5 * dt * k2[j] for j in range(4))
            k3 = rhs(*y3, m1, m2, l1, l2, g)
            y4 = tuple(y1[j] + dt * k3[j] for j in range(4))
            k4 = rhs(*y4, m1, m2, l1, l2, g)

            theta1[i + 1] = y1[0] + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            omega1[i + 1] = y1[1] + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            theta2[i + 1] = y1[2] + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            omega2[i + 1] = y1[3] + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        return t, theta1, omega1, theta2, omega2

    return integrate_double_pendulum


@app.cell
def _(
    dt,
    g,
    integrate_double_pendulum,
    l1,
    l2,
    m1,
    m2,
    nsteps,
    omega1_0,
    omega2_0,
    theta1_0,
    theta2_0,
):
    t, theta1, omega1, theta2, omega2 = integrate_double_pendulum(
        theta1_0.value,
        omega1_0.value,
        theta2_0.value,
        omega2_0.value,
        m1.value,
        m2.value,
        l1.value,
        l2.value,
        g.value,
        dt.value,
        int(nsteps.value),
    )
    return omega1, omega2, t, theta1, theta2


@app.cell
def _(l1, l2, np, theta1, theta2):
    x1 = l1.value * np.sin(theta1)
    y1 = -l1.value * np.cos(theta1)
    x2 = x1 + l2.value * np.sin(theta2)
    y2 = y1 - l2.value * np.cos(theta2)
    return x1, x2, y1, y2


@app.cell
def _(mo, t):
    frame = mo.ui.slider(
        start=0, stop=len(t) - 1, step=1, value=min(300, len(t) - 1), label="frame"
    )
    return (frame,)


@app.cell
def _(frame, mo, plt, t, theta1, theta2, x1, x2, y1, y2):
    i = frame.value

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], "o-", lw=2)
    ax[0].plot(x2[: i + 1], y2[: i + 1], "r-", lw=1)
    reach = max(abs(x2).max(), abs(y2).max()) + 0.2
    ax[0].set_xlim(-reach, reach)
    ax[0].set_ylim(-reach, reach)
    ax[0].set_aspect("equal", "box")
    ax[0].set_title(f"Configuración instantánea (t={t[i]:.2f} s)")
    ax[0].grid(True)

    ax[1].plot(t, theta1, label=r"$\theta_1(t)$")
    ax[1].plot(t, theta2, label=r"$\theta_2(t)$")
    ax[1].axvline(t[i], color="k", ls="--", lw=1)
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("ángulo [rad]")
    ax[1].set_title("Evolución angular")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    mo.vstack([frame, fig])
    return


if __name__ == "__main__":
    app.run()
