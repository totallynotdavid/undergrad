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
        # Dos masas y tres resortes: modos normales

        Dos masas $m_1, m_2$ se mueven en línea recta entre dos paredes. Un
        resorte $k_1$ une la pared izquierda con $m_1$, un resorte $k$ une las
        masas y un resorte $k_2$ une $m_2$ con la pared derecha. Con $x_1,x_2$
        los desplazamientos desde el equilibrio,

        $$
        m_1\ddot x_1 = -k_1x_1 + k\,(x_2-x_1),
        \qquad
        m_2\ddot x_2 = -k_2x_2 - k\,(x_2-x_1),
        $$

        o en forma matricial $M\ddot{\mathbf x} = -K\mathbf x$ con

        $$
        M=\begin{pmatrix} m_1 & 0\\ 0 & m_2\end{pmatrix},
        \qquad
        K=\begin{pmatrix} k_1+k & -k\\ -k & k_2+k\end{pmatrix}.
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Modos normales

        Buscamos soluciones $\mathbf x(t)=\mathbf v\cos(\omega t)$. Sustituir da el
        problema de valores propios generalizado $K\mathbf v=\omega^2M\mathbf v$.
        Con $A=M^{-1/2}KM^{-1/2}$, simétrica, se reduce a $A\mathbf u=\omega^2\mathbf u$
        y `numpy.linalg.eigh` entrega valores propios reales y vectores
        ortonormales. Los modos son $\mathbf v_i=M^{-1/2}\mathbf u_i$ y cumplen
        $\mathbf v_i^{T}M\mathbf v_j=\delta_{ij}$.

        Las coordenadas modales $q_i=\mathbf v_i^{T}M\,\mathbf x$ desacoplan el
        sistema: $\ddot q_i=-\omega_i^2q_i$. Cada modo oscila con una sola
        frecuencia, y cualquier movimiento es una superposición de ellos.
        """
    )
    return


@app.cell
def _(np):
    M1, M2 = 1.0, 2.0
    K1 = K2 = KC = 250.0

    masas = np.diag([M1, M2])
    rigidez = np.array([[K1 + KC, -KC], [-KC, K2 + KC]])

    _m_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(masas)))
    _valores, _u = np.linalg.eigh(_m_inv_sqrt @ rigidez @ _m_inv_sqrt)
    omega = np.sqrt(_valores)  # rad/s, de menor a mayor
    modos = _m_inv_sqrt @ _u  # columnas v_i con v_i^T M v_j = delta_ij
    # Forma de cada modo con amplitud máxima 1 y x_1 >= 0.
    formas = modos / abs(modos).max(axis=0) * np.sign(modos[0])
    return M1, M2, formas, masas, modos, omega, rigidez


@app.cell(hide_code=True)
def _(M1, M2, formas, md, np, omega):
    md(
        "\n".join(
            [
                rf"$m_1={M1}$ kg, $m_2={M2}$ kg, $k_1=k_2=k=250$ N/m.",
                "",
                "| modo | $\\omega$ [rad/s] | $f$ [Hz] | $x_1$ | $x_2$ |",
                "|---:|---:|---:|---:|---:|",
                *[
                    f"| {i + 1} | {omega[i]:.3f} | {omega[i] / (2 * np.pi):.3f} "
                    f"| {formas[0, i]:+.3f} | {formas[1, i]:+.3f} |"
                    for i in range(2)
                ],
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(M1, M2, formas, md):
    md(
        rf"""
        En el modo de menor frecuencia las masas se mueven en fase, con la
        masa mayor desplazándose más. En el de mayor frecuencia se mueven en
        oposición, estirando y comprimiendo el resorte central, y por eso su
        frecuencia es mayor. Los dos vectores son ortogonales respecto de $M$:
        ${M1}\cdot{formas[0, 0]:.3f}\cdot{formas[0, 1]:.3f}
        + {M2}\cdot{formas[1, 0]:.3f}\cdot({formas[1, 1]:.3f}) = 0$.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Integración temporal con RK4

        Integramos el sistema de cuatro ecuaciones de primer orden
        $\dot{\mathbf x}=\mathbf v,\ \dot{\mathbf v}=-M^{-1}K\mathbf x$ con RK4
        y paso $h=0.01$ s, partiendo del reposo con tres condiciones iniciales:

        1. $\mathbf x(0)$ igual al modo 1.
        2. $\mathbf x(0)$ igual al modo 2.
        3. $x_1(0)=1$ m, $x_2(0)=0$: una mezcla de ambos modos.

        La solución analítica desde el reposo es
        $\mathbf x(t)=\sum_i q_i(0)\,\mathbf v_i\cos\omega_it$, y la usamos para
        medir el error de RK4.
        """
    )
    return


@app.cell
def _(formas, masas, modos, np, omega, rigidez):
    H = 0.01
    PASOS = 2000  # 20 s

    _aceleracion = -np.linalg.solve(masas, rigidez)

    def rk4(x0, h=H, pasos=PASOS):
        def f(y):
            return np.concatenate([y[2:], _aceleracion @ y[:2]])

        y = np.empty((pasos + 1, 4))
        y[0] = np.concatenate([x0, [0.0, 0.0]])
        for i in range(pasos):
            k1 = f(y[i])
            k2 = f(y[i] + 0.5 * h * k1)
            k3 = f(y[i] + 0.5 * h * k2)
            k4 = f(y[i] + h * k3)
            y[i + 1] = y[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def analitica(x0, t):
        q0 = modos.T @ masas @ x0
        return (modos * q0) @ np.cos(np.outer(omega, t))

    t = np.arange(PASOS + 1) * H
    casos = {
        "modo 1": formas[:, 0],
        "modo 2": formas[:, 1],
        r"$x_1(0)=1,\ x_2(0)=0$": np.array([1.0, 0.0]),
    }
    trayectorias = {nombre: rk4(x0) for nombre, x0 in casos.items()}
    errores = {
        nombre: abs(trayectorias[nombre][:, :2].T - analitica(x0, t)).max()
        for nombre, x0 in casos.items()
    }
    return PASOS, analitica, errores, t, trayectorias


@app.cell
def _(plt, t, trayectorias):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for _a, (_nombre, _y) in zip(_ax, trayectorias.items()):
        _a.plot(t, _y[:, 0], label=r"$x_1(t)$")
        _a.plot(t, _y[:, 1], label=r"$x_2(t)$")
        _a.set_xlim(0, 3)
        _a.set_xlabel("t [s]")
        _a.set_title(_nombre)
        _a.grid(True)
    _ax[0].set_ylabel("desplazamiento [m]")
    _ax[0].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(errores, md):
    md(
        "\n".join(
            [
                "| condición inicial | error máximo de RK4 frente a la solución analítica [m] |",
                "|---|---:|",
                *[f"| {n} | {e:.1e} |" for n, e in errores.items()],
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El error de RK4 es un corrimiento de fase que se acumula durante los 20
        s y es mayor en el modo de mayor frecuencia. Si el movimiento empieza
        en un modo, las dos masas oscilan con la misma frecuencia y con una
        relación de amplitudes fija. En la mezcla, cada masa muestra las dos
        frecuencias a la vez.

        ## Los modos en la serie temporal

        Proyectar la serie numérica de la mezcla sobre los modos,
        $q_i(t)=\mathbf v_i^{T}M\,\mathbf x(t)$, recupera dos cosenos puros. Y el
        espectro de $x_1(t)$ tiene exactamente dos picos, en las frecuencias de
        los modos.
        """
    )
    return


@app.cell
def _(masas, modos, np, omega, plt, t, trayectorias):
    _y = trayectorias[r"$x_1(0)=1,\ x_2(0)=0$"]
    _q = modos.T @ masas @ _y[:, :2].T

    _ventana = np.hanning(len(t))
    _espectro = np.abs(np.fft.rfft(_y[:, 0] * _ventana))
    _freq = np.fft.rfftfreq(len(t), d=t[1] - t[0])

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].plot(t, _q[0], label=r"$q_1(t)$")
    _ax[0].plot(t, _q[1], label=r"$q_2(t)$")
    _ax[0].set_xlim(0, 3)
    _ax[0].set_xlabel("t [s]")
    _ax[0].set_ylabel(r"coordenada modal [m$\cdot$kg$^{1/2}$]")
    _ax[0].set_title("Coordenadas modales de la mezcla")
    _ax[0].grid(True)
    _ax[0].legend()

    _ax[1].semilogy(_freq, _espectro / _espectro.max())
    for _i, _w in enumerate(omega):
        _ax[1].axvline(
            _w / (2 * np.pi),
            color="tab:red",
            ls="--",
            lw=1,
            label=rf"$\omega_{_i + 1}/2\pi={_w / (2 * np.pi):.2f}$ Hz",
        )
    _ax[1].set_xlim(0, 8)
    _ax[1].set_ylim(1e-4, 2)
    _ax[1].set_xlabel("f [Hz]")
    _ax[1].set_ylabel("|FFT| normalizada")
    _ax[1].set_title(r"Espectro de $x_1(t)$")
    _ax[1].grid(True, which="both", alpha=0.4)
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
