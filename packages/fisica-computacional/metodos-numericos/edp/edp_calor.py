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
        # Ecuación de calor 1D: explícito, Crank–Nicolson y Fourier

        Una barra de longitud $L$ con difusividad $\alpha$ evoluciona según

        $$
        \frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2},
        \qquad
        T(0,t)=0,\quad T(L,t)=T_b,\quad T(x,0)=0\ \ (0\le x<L).
        $$

        Hay una solución analítica. Con $T=T_b\,x/L+u$, $u$ cumple la ecuación
        con extremos en cero y condición inicial $u(x,0)=-T_b\,x/L$, y el
        desarrollo en senos da

        $$
        T(x,t)=T_b\frac{x}{L}+\sum_{n=1}^{\infty}
        \frac{2T_b(-1)^n}{n\pi}\,
        \sin\frac{n\pi x}{L}\;e^{-\alpha(n\pi/L)^2t}.
        $$

        Cada modo decae con su propia constante de tiempo. Para $t>0$ la serie
        converge de inmediato, y la usamos como referencia exacta con 400
        términos.
        """
    )
    return


@app.cell
def _(np):
    ALFA, LARGO, T_BORDE, NODOS = 10.0, 10.0, 100.0, 21
    x = np.linspace(0.0, LARGO, NODOS)
    dx = x[1] - x[0]

    def fourier(t, terminos=400):
        n = np.arange(1, terminos + 1)[:, None]
        coef = 2 * T_BORDE * (-1.0) ** n / (n * np.pi)
        modos = (
            coef
            * np.sin(n * np.pi * x / LARGO)
            * np.exp(-ALFA * (n * np.pi / LARGO) ** 2 * t)
        )
        return T_BORDE * x / LARGO + modos.sum(axis=0)

    return ALFA, NODOS, T_BORDE, dx, fourier, x


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Esquemas

        Con $T_i^n\approx T(i\,\Delta x,\,n\,\Delta t)$ y el número

        $$
        r=\frac{\alpha\,\Delta t}{\Delta x^2}
        $$

        el **esquema explícito** (FTCS) es

        $$
        T_i^{n+1}=T_i^n+r\left(T_{i-1}^n-2T_i^n+T_{i+1}^n\right).
        $$

        Para el modo de Fourier de número de onda $k$, cada paso lo multiplica
        por $1-4r\sin^2(k\Delta x/2)$. El modo más oscilante
        ($k\Delta x=\pi$) se multiplica por $1-4r$, que tiene módulo mayor que 1
        si $r>1/2$: ese modo crece en cada paso. El esquema es estable sólo
        para $r\le1/2$, y con $r=1/2$ el modo más oscilante no decae.

        **Crank–Nicolson** promedia el laplaciano entre $n$ y $n+1$:

        $$
        T_i^{n+1}-\tfrac r2\,\delta^2T_i^{n+1}
        =T_i^{n}+\tfrac r2\,\delta^2T_i^{n},
        \qquad
        \delta^2T_i=T_{i-1}-2T_i+T_{i+1}.
        $$

        Cada paso resuelve un sistema tridiagonal. El factor de amplificación
        $\dfrac{1-2r\sin^2(k\Delta x/2)}{1+2r\sin^2(k\Delta x/2)}$ tiene módulo
        menor que 1 para todo $r$, y el error de truncamiento es
        $O(\Delta t^2+\Delta x^2)$ en lugar de $O(\Delta t+\Delta x^2)$.
        """
    )
    return


@app.cell
def _(ALFA, NODOS, T_BORDE, dx, np):
    def condicion_inicial():
        T = np.zeros(NODOS)
        T[-1] = T_BORDE
        return T

    def explicito(r, t_final):
        dt = r * dx**2 / ALFA
        pasos = round(t_final / dt)
        T = condicion_inicial()
        historia = [T.copy()]
        for _ in range(pasos):
            T[1:-1] += r * (T[:-2] - 2 * T[1:-1] + T[2:])
            historia.append(T.copy())
        return dt * np.arange(pasos + 1), np.array(historia)

    def crank_nicolson(r, t_final):
        dt = r * dx**2 / ALFA
        pasos = round(t_final / dt)
        n = NODOS - 2
        laplaciano = (
            np.diag(-2.0 * np.ones(n))
            + np.diag(np.ones(n - 1), 1)
            + np.diag(np.ones(n - 1), -1)
        )
        izquierda = np.eye(n) - 0.5 * r * laplaciano
        derecha = np.eye(n) + 0.5 * r * laplaciano
        T = condicion_inicial()
        historia = [T.copy()]
        for _ in range(pasos):
            b = derecha @ T[1:-1]
            b[-1] += r * T_BORDE  # Mantiene fijo el extremo derecho en ambos pasos.
            T[1:-1] = np.linalg.solve(izquierda, b)
            historia.append(T.copy())
        return dt * np.arange(pasos + 1), np.array(historia)

    return crank_nicolson, explicito


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Explícito con $r=0.4$, $0.5$ y $0.6$

        Fijamos $\alpha=10$, $L=10$, $T_b=100$ y $\Delta x=0.5$, y cambiamos
        $\Delta t$ para obtener cada $r$. Los tres casos avanzan hasta
        $t=1.5$, y la línea negra es la solución de Fourier.
        """
    )
    return


@app.cell
def _(explicito):
    T_FINAL = 1.5
    R_EXPLICITOS = (0.4, 0.5, 0.6)
    explicitos = {r: explicito(r, T_FINAL) for r in R_EXPLICITOS}
    return R_EXPLICITOS, T_FINAL, explicitos


@app.cell
def _(R_EXPLICITOS, ALFA, dx, explicitos, fourier, plt, x):
    _fig, _ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for _a, _r in zip(_ax, R_EXPLICITOS):
        _t, _T = explicitos[_r]
        for _instante in (0.1, 0.3, 0.6, 1.5):
            _k = int(round(_instante / (_t[1] - _t[0])))
            (_linea,) = _a.plot(x, _T[_k], "o-", ms=3, lw=1, label=f"t={_t[_k]:.2f}")
            _a.plot(x, fourier(_t[_k]), "k", lw=0.8)
        _a.set_ylim(-20, 120)
        _a.set_xlabel("x")
        _a.set_title(rf"$r={_r}$, $\Delta t={_r * dx**2 / ALFA:.4f}$")
        _a.grid(alpha=0.3)
        _a.text(
            0.03,
            0.05,
            rf"$\max|T|={abs(_T).max():.3g}$",
            transform=_a.transAxes,
            bbox={"facecolor": "white", "alpha": 0.8},
        )
    _ax[0].set_ylabel("T(x, t)")
    _ax[0].legend(loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Con $r=0.4$ el perfil sigue a la solución exacta. Con $r=0.5$ también,
        pero en el límite de estabilidad: el modo más oscilante no decae y
        conserva el ruido que el salto inicial en el borde le entrega. Con
        $r=0.6$ ese modo se multiplica por $-1.4$ en cada paso, y el perfil
        que se ve es sólo la oscilación de nodo a nodo, miles de millones de
        muchos órdenes de magnitud mayor que la temperatura física. La escala
        vertical está recortada; la anotación da el valor real.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Crank–Nicolson y el error en el tiempo

        Repetimos la integración con Crank–Nicolson y los mismos $\Delta t$,
        y medimos $\max_i|T_i^n-T(x_i,t_n)|$ en cada paso para los dos
        esquemas.
        """
    )
    return


@app.cell
def _(T_FINAL, crank_nicolson, fourier, np):
    cn = {r: crank_nicolson(r, T_FINAL) for r in (0.6, 5.0, 20.0)}

    def error_en_el_tiempo(t, T):
        return np.array([np.abs(T[k] - fourier(t[k])).max() for k in range(len(t))])

    return cn, error_en_el_tiempo


@app.cell
def _(R_EXPLICITOS, cn, error_en_el_tiempo, explicitos, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    for _r in R_EXPLICITOS:
        _t, _T = explicitos[_r]
        _ax.semilogy(_t[1:], error_en_el_tiempo(_t, _T)[1:], label=f"explícito, r={_r}")
    for _r in (0.6, 5.0):
        _t, _T = cn[_r]
        _ax.semilogy(
            _t[1:],
            error_en_el_tiempo(_t, _T)[1:],
            "--",
            label=f"Crank–Nicolson, r={_r}",
        )
    _ax.set_xlabel("t")
    _ax.set_ylabel(r"$\max_i |T_i^n - T(x_i,t_n)|$")
    _ax.set_title("Error frente a la solución de Fourier")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(R_EXPLICITOS, T_FINAL, ALFA, cn, dx, error_en_el_tiempo, explicitos, md):
    _filas = []
    for _nombre, _r, (_t, _T) in [
        *[("explícito", _r, explicitos[_r]) for _r in R_EXPLICITOS],
        *[("Crank–Nicolson", _r, cn[_r]) for _r in cn],
    ]:
        _error = error_en_el_tiempo(_t, _T)
        _filas.append(
            f"| {_nombre} | {_r} | {_r * dx**2 / ALFA:.4f} | {len(_t) - 1} "
            f"| {_T.min():.3g} | {_T.max():.3g} | {_error[-1]:.2g} | {_error[1:].max():.2g} |"
        )
    md(
        "\n".join(
            [
                f"Resultados hasta $t={T_FINAL}$ (la solución física está entre 0 y 100). "
                "El error máximo en el tiempo ocurre en los primeros pasos: la "
                "condición inicial es discontinua en la esquina $(x=L,\\,t=0)$ y "
                "ningún esquema la representa con pocos pasos.",
                "",
                r"| esquema | $r$ | $\Delta t$ | pasos | $\min T$ | $\max T$ | error en $t=1.5$ | error máximo en el tiempo |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell
def _(cn, explicitos, fourier, plt, x):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _instante = 0.75
    _ax.plot(x, fourier(_instante), "k", lw=1.5, label="Fourier")
    for _r in (0.6, 5.0):
        _t, _T = cn[_r]
        _k = int(round(_instante / (_t[1] - _t[0])))
        _ax.plot(x, _T[_k], "o-", ms=3, lw=1, label=f"Crank–Nicolson, r={_r}")
    _t, _T = explicitos[0.4]
    _k = int(round(_instante / (_t[1] - _t[0])))
    _ax.plot(x, _T[_k], "s--", ms=3, lw=1, label="explícito, r=0.4")
    _ax.set_xlabel("x")
    _ax.set_ylabel("T(x, t)")
    _ax.set_title(f"Perfiles en t = {_instante}")
    _ax.grid(alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Crank–Nicolson no se desestabiliza para ningún $r$: con $r=0.6$ es la
        opción de mayor precisión de la tabla, y con $r=5$ el paso es unas ocho
        veces mayor y el error no crece sin cota. Pero la condición inicial es
        discontinua en $x=L$, y los modos de alta frecuencia casi no se
        amortiguan cuando $r$ es grande: con $r=5$ y $r=20$ aparecen valores
        por encima de $T_b$ (columna $\max T$), que la solución física nunca
        alcanza. Para pasos grandes conviene un esquema completamente
        implícito, o arrancar Crank–Nicolson con unos pasos pequeños.
        """
    )
    return


if __name__ == "__main__":
    app.run()
