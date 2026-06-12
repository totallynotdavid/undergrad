import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # EDP de calor (1D): método explícito

    Consideramos la ecuación

    \[
    \frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2},
    \]

    con frontera de temperatura fija en una barra de longitud \(L\). Usamos diferencias finitas explícitas:

    \[
    T_i^{n+1}=T_i^n+\gamma\left(T_{i-1}^n-2T_i^n+T_{i+1}^n\right),
    \quad
    \gamma=\alpha\frac{\Delta t}{\Delta x^2}.
    \]

    Para estabilidad del esquema explícito en 1D se requiere \(\gamma \le 1/2\).
    """)
    return


@app.cell
def _(mo):
    alpha = mo.ui.slider(0.1, 20.0, value=10.0, step=0.1, label="Difusividad α")
    dt = mo.ui.slider(0.001, 0.1, value=0.02, step=0.001, label="Paso temporal Δt")
    dx = mo.ui.slider(0.25, 2.0, value=1.0, step=0.25, label="Paso espacial Δx")
    nx = mo.ui.slider(11, 101, value=11, step=2, label="Número de nodos espaciales")
    nt = mo.ui.slider(50, 600, value=101, step=1, label="Número de pasos temporales")
    return alpha, dt, dx, nt, nx


@app.cell
def _(alpha, dt, dx, mo, nt, nx):
    controls = mo.vstack(
        [
            mo.md("## Parámetros numéricos"),
            alpha,
            dt,
            dx,
            nx,
            nt,
        ]
    )
    controls
    return


@app.cell
def _(alpha, dt, dx):
    gamma = alpha.value * dt.value / (dx.value**2)
    is_stable = gamma <= 0.5
    return gamma, is_stable


@app.cell
def _(gamma, is_stable, mo):
    status = "cumplida" if is_stable else "violada"
    color = "green" if is_stable else "red"
    mo.md(
        f"""
    **Condición de estabilidad:**
    - \\(\\gamma = {gamma:.4f}\\)
    - Estado: <span style='color:{color}'>{status}</span> (se requiere \\(\\gamma \\le 0.5\\)).
    """
    )
    return


@app.cell
def _(alpha, dt, dx, np, nt, nx):
    x = np.arange(nx.value, dtype=float) * dx.value
    t = np.arange(nt.value, dtype=float) * dt.value

    T = np.zeros((nt.value, nx.value), dtype=float)
    T[:, 0] = 0.0
    T[:, -1] = 100.0

    gamma_fd = alpha.value * dt.value / (dx.value**2)
    for n_step in range(nt.value - 1):
        for i_pos in range(1, nx.value - 1):
            T[n_step + 1, i_pos] = T[n_step, i_pos] + gamma_fd * (
                T[n_step, i_pos - 1] - 2 * T[n_step, i_pos] + T[n_step, i_pos + 1]
            )
    return T, t, x


@app.cell
def _(mo, t):
    frame = mo.ui.slider(
        0, len(t) - 1, value=min(45, len(t) - 1), step=1, label="Índice temporal n"
    )
    return (frame,)


@app.cell
def _(frame):
    frame
    return


@app.cell
def _(T, frame, plt, t, x):
    idx = frame.value
    fig_profile, ax_profile = plt.subplots(figsize=(7, 4))
    ax_profile.plot(x, T[idx, :], lw=2)
    ax_profile.set_title(f"Perfil de temperatura en t = {t[idx]:.3f} s")
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel("T(x, t)")
    ax_profile.grid(alpha=0.3)
    fig_profile
    return


@app.cell
def _(T, plt, t, x):
    fig_multi, ax_multi = plt.subplots(figsize=(7, 4))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        n_curve = int(frac * (len(t) - 1))
        ax_multi.plot(x, T[n_curve, :], label=f"t={t[n_curve]:.2f} s")
    ax_multi.set_title("Evolución de perfiles de temperatura")
    ax_multi.set_xlabel("x")
    ax_multi.set_ylabel("T(x, t)")
    ax_multi.grid(alpha=0.3)
    ax_multi.legend()
    fig_multi
    return


if __name__ == "__main__":
    app.run()
