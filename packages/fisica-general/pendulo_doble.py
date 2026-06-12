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
# Péndulo doble acoplado: integración de Euler

Este notebook muestra un modelo didáctico de dos ángulos acoplados
$(\theta_1,\theta_2)$ con integración explícita de Euler.
"""
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
## Modelo usado

Se integra el sistema:

$$
\ddot\theta_1 = -\frac{m_2 l_2}{l_1(m_1+m_2)}\cos(\theta_1-\theta_2)\dot\theta_2^2
-\frac{g}{l_1}\sin\theta_1 + \frac{b}{l_1(m_1+m_2)}\theta_1,
$$

$$
\ddot\theta_2 = -\frac{l_1}{l_2}\cos(\theta_1-\theta_2)\dot\theta_1^2
-\frac{g}{l_2}\sin\theta_2 + \frac{b}{m_2 l_2}\theta_2.
$$

Con Euler explícito:
$$
\theta(t+\Delta t)=\theta(t)+\dot\theta(t)\Delta t,
\qquad
\dot\theta(t+\Delta t)=\dot\theta(t)+\ddot\theta(t)\Delta t.
$$

Nota: Euler es simple pero acumula error numérico; aquí lo usamos por claridad pedagógica.
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
def _(np):
    def metodo_euler(
        theta1_0, theta2_0, omega1_0, omega2_0, m1, m2, l1, l2, g, b, dt, num_pasos
    ):
        theta1 = np.zeros(num_pasos)
        theta2 = np.zeros(num_pasos)
        omega1 = np.zeros(num_pasos)
        omega2 = np.zeros(num_pasos)

        theta1[0] = theta1_0
        theta2[0] = theta2_0
        omega1[0] = omega1_0
        omega2[0] = omega2_0

        for i in range(1, num_pasos):
            theta1_acc = (
                -m2
                * l2
                / (l1 * (m1 + m2))
                * np.cos(theta1[i - 1] - theta2[i - 1])
                * omega2[i - 1] ** 2
                - g / l1 * np.sin(theta1[i - 1])
                + b / (l1 * (m1 + m2)) * theta1[i - 1]
            )
            theta2_acc = (
                -l1 / l2 * np.cos(theta1[i - 1] - theta2[i - 1]) * omega1[i - 1] ** 2
                - g / l2 * np.sin(theta2[i - 1])
                + b / (m2 * l2) * theta2[i - 1]
            )

            theta1[i] = theta1[i - 1] + omega1[i - 1] * dt
            theta2[i] = theta2[i - 1] + omega2[i - 1] * dt
            omega1[i] = omega1[i - 1] + theta1_acc * dt
            omega2[i] = omega2[i - 1] + theta2_acc * dt

        return theta1, theta2, omega1, omega2

    return (metodo_euler,)


@app.cell
def _(mo):
    m1 = mo.ui.number(start=0.1, stop=10.0, step=0.1, value=1.0, label="m1")
    m2 = mo.ui.number(start=0.1, stop=10.0, step=0.1, value=2.0, label="m2")
    l1 = mo.ui.number(start=0.1, stop=5.0, step=0.1, value=1.0, label="l1")
    l2 = mo.ui.number(start=0.1, stop=5.0, step=0.1, value=2.0, label="l2")
    g = mo.ui.number(start=0.1, stop=20.0, step=0.1, value=9.81, label="g")
    b = mo.ui.number(start=-10.0, stop=10.0, step=0.1, value=-2.5, label="b")

    theta1_0 = mo.ui.number(
        start=-3.14, stop=3.14, step=0.01, value=0.1, label="theta1(0)"
    )
    theta2_0 = mo.ui.number(
        start=-3.14, stop=3.14, step=0.01, value=0.2, label="theta2(0)"
    )
    omega1_0 = mo.ui.number(
        start=-10.0, stop=10.0, step=0.01, value=0.0, label="omega1(0)"
    )
    omega2_0 = mo.ui.number(
        start=-10.0, stop=10.0, step=0.01, value=0.0, label="omega2(0)"
    )

    dt = mo.ui.number(start=0.001, stop=0.05, step=0.001, value=0.01, label="dt")
    num_pasos = mo.ui.slider(
        start=200, stop=5000, step=100, value=1000, label="num_pasos"
    )

    return b, dt, g, l1, l2, m1, m2, num_pasos, omega1_0, omega2_0, theta1_0, theta2_0


@app.cell(hide_code=True)
def _(b, dt, g, l1, l2, m1, m2, mo, num_pasos, omega1_0, omega2_0, theta1_0, theta2_0):
    mo.vstack(
        [
            mo.hstack([m1, m2, l1, l2, g, b], justify="start"),
            mo.hstack([theta1_0, theta2_0, omega1_0, omega2_0], justify="start"),
            mo.hstack([dt, num_pasos], justify="start"),
        ]
    )
    return


@app.cell
def _(
    b,
    dt,
    g,
    l1,
    l2,
    m1,
    m2,
    metodo_euler,
    num_pasos,
    np,
    omega1_0,
    omega2_0,
    theta1_0,
    theta2_0,
):
    theta1, theta2, omega1, omega2 = metodo_euler(
        theta1_0.value,
        theta2_0.value,
        omega1_0.value,
        omega2_0.value,
        m1.value,
        m2.value,
        l1.value,
        l2.value,
        g.value,
        b.value,
        dt.value,
        int(num_pasos.value),
    )
    t = np.arange(int(num_pasos.value)) * dt.value

    x1 = l1.value * np.sin(theta1)
    y1 = -l1.value * np.cos(theta1)
    x2 = x1 + l2.value * np.sin(theta2)
    y2 = y1 - l2.value * np.cos(theta2)

    return t, theta1, theta2, omega1, omega2, x1, x2, y1, y2


@app.cell
def _(mo, t):
    frame = mo.ui.slider(
        start=0, stop=len(t) - 1, step=1, value=min(200, len(t) - 1), label="frame"
    )
    return (frame,)


@app.cell
def _(frame, mo, plt, t, theta1, theta2, x1, x2, y1, y2):
    idx = frame.value

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot([0, x1[idx], x2[idx]], [0, y1[idx], y2[idx]], "o-", lw=2)
    ax[0].plot(x2[: idx + 1], y2[: idx + 1], "r-", lw=1)
    reach = max(abs(x2).max(), abs(y2).max()) + 0.2
    ax[0].set_xlim(-reach, reach)
    ax[0].set_ylim(-reach, reach)
    ax[0].set_aspect("equal", "box")
    ax[0].set_title(f"Configuración instantánea (t={t[idx]:.2f} s)")
    ax[0].grid(True)

    ax[1].plot(t, theta1, label=r"$\theta_1(t)$")
    ax[1].plot(t, theta2, label=r"$\theta_2(t)$")
    ax[1].axvline(t[idx], color="k", ls="--", lw=1)
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
