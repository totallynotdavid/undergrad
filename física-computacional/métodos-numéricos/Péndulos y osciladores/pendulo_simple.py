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
    # Péndulo simple: integración RK4

    Modelo linealizado:
    $$
    \ddot\theta + \omega_0^2\theta = 0,\qquad \omega_0^2 = g/\ell.
    $$

    Sistema de primer orden:
    $$
    \dot\theta = \omega,\qquad \dot\omega = -(g/\ell)\theta.
    $$
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
    theta0 = mo.ui.number(start=-3.14, stop=3.14, step=0.01, value=0.2, label="theta(0)")
    omega0 = mo.ui.number(start=-10.0, stop=10.0, step=0.01, value=0.0, label="omega(0)")
    length = mo.ui.number(start=0.2, stop=10.0, step=0.1, value=1.0, label="l")
    g = mo.ui.number(start=0.1, stop=20.0, step=0.1, value=9.81, label="g")
    dt = mo.ui.number(start=0.001, stop=0.1, step=0.001, value=0.01, label="dt")
    tf = mo.ui.number(start=1.0, stop=60.0, step=1.0, value=20.0, label="t_final")
    return dt, g, length, omega0, tf, theta0


@app.cell(hide_code=True)
def _(dt, g, length, mo, omega0, tf, theta0):
    mo.hstack([theta0, omega0, length, g, dt, tf], justify="start")
    return


@app.cell
def _(np):
    def integrate_pendulum_rk4(theta0, omega0, length, g, dt, tf):
        n = int(tf / dt)
        t = np.linspace(0.0, n * dt, n + 1)
        theta = np.zeros(n + 1)
        omega = np.zeros(n + 1)
        theta[0] = theta0
        omega[0] = omega0

        def f_theta(_theta, _omega):
            return _omega

        def f_omega(_theta, _omega):
            return -(g / length) * _theta

        for i in range(n):
            k1_theta = dt * f_theta(theta[i], omega[i])
            k1_omega = dt * f_omega(theta[i], omega[i])

            k2_theta = dt * f_theta(theta[i] + 0.5 * k1_theta, omega[i] + 0.5 * k1_omega)
            k2_omega = dt * f_omega(theta[i] + 0.5 * k1_theta, omega[i] + 0.5 * k1_omega)

            k3_theta = dt * f_theta(theta[i] + 0.5 * k2_theta, omega[i] + 0.5 * k2_omega)
            k3_omega = dt * f_omega(theta[i] + 0.5 * k2_theta, omega[i] + 0.5 * k2_omega)

            k4_theta = dt * f_theta(theta[i] + k3_theta, omega[i] + k3_omega)
            k4_omega = dt * f_omega(theta[i] + k3_theta, omega[i] + k3_omega)

            theta[i + 1] = theta[i] + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
            omega[i + 1] = omega[i] + (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6

        x = length * np.sin(theta)
        y = -length * np.cos(theta)
        return t, theta, omega, x, y

    return (integrate_pendulum_rk4,)


@app.cell
def _(dt, g, integrate_pendulum_rk4, length, omega0, tf, theta0):
    t, theta, omega, x, y = integrate_pendulum_rk4(
        theta0.value,
        omega0.value,
        length.value,
        g.value,
        dt.value,
        tf.value,
    )
    return omega, t, theta, x, y


@app.cell
def _(mo, t):
    frame = mo.ui.slider(start=0, stop=len(t) - 1, step=1, value=min(150, len(t) - 1), label="frame")
    return (frame,)


@app.cell
def _(frame, mo, omega, plt, t, theta, x, y):
    i = frame.value

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    ax[0].plot([0, x[i]], [0, y[i]], "k-", lw=2)
    ax[0].plot(x[i], y[i], "ro")
    ax[0].set_xlim(-1.2 * max(1.0, abs(x).max()), 1.2 * max(1.0, abs(x).max()))
    ax[0].set_ylim(-1.2 * max(1.0, abs(y).max()), 0.2)
    ax[0].set_aspect("equal", "box")
    ax[0].set_title(f"Configuración instantánea (t={t[i]:.2f} s)")
    ax[0].grid(True)

    ax[1].plot(t, theta, label=r"$\theta(t)$")
    ax[1].plot(t, omega, label=r"$\omega(t)$")
    ax[1].axvline(t[i], color="k", ls="--", lw=1)
    ax[1].set_xlabel("t [s]")
    ax[1].set_title("Espacio de estados temporal")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    mo.vstack([frame, fig])
    return


if __name__ == "__main__":
    app.run()
