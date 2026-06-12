import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import odeint

    INITIAL_STATE = (0.1, 0.0, 0.0)
    POINT_DENSITY = 500
    return INITIAL_STATE, POINT_DENSITY, mo, np, odeint, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Atractor de Lorenz

    Ajusta parámetros y visualiza el atractor directamente en el notebook.
    """)
    return


@app.cell
def _(mo):
    sigma = mo.ui.slider(0.0, 30.0, value=10.0, step=0.1, label="sigma")
    rho = mo.ui.slider(0.0, 60.0, value=28.0, step=0.1, label="rho")
    beta = mo.ui.slider(0.5, 5.0, value=8.0 / 3.0, step=0.01, label="beta")
    t_end = mo.ui.slider(10, 120, value=60, step=1, label="t_fin")
    return beta, rho, sigma, t_end


@app.cell
def _(POINT_DENSITY, beta, mo, np, rho, sigma, t_end):
    @mo.cache
    def resolver_lorenz(sigma_val, rho_val, beta_val, t_end_val, density):
        # Cache avoids recomputing the full trajectory for unchanged parameters.
        puntos_tiempo = np.linspace(1, t_end_val, t_end_val * density)

        def sistema_lorenz(estado_actual, t):
            x, y, z = estado_actual
            dx_dt = sigma_val * (y - x)
            dy_dt = x * (rho_val - z) - y
            dz_dt = x * y - beta_val * z
            return [dx_dt, dy_dt, dz_dt]

        return puntos_tiempo, sistema_lorenz

    puntos_tiempo, sistema_lorenz = resolver_lorenz(
        sigma.value, rho.value, beta.value, t_end.value, POINT_DENSITY
    )
    return puntos_tiempo, sistema_lorenz


@app.cell
def _(INITIAL_STATE, odeint, puntos_tiempo, sistema_lorenz):
    puntos = odeint(sistema_lorenz, INITIAL_STATE, puntos_tiempo)
    return (puntos,)


@app.cell
def _(plt, puntos):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    x = puntos[:, 0]
    y = puntos[:, 1]
    z = puntos[:, 2]
    ax.plot(x, y, z, color="k", alpha=0.7, linewidth=0.9, antialiased=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis("off")
    return (fig,)


@app.cell
def _(beta, fig, mo, rho, sigma, t_end):
    mo.hstack(
        [
            mo.vstack([sigma, rho, beta, t_end], align="stretch"),
            fig,
        ],
        widths=[1, 2],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
