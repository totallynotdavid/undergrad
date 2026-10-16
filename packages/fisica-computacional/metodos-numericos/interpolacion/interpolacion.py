import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from math import factorial
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo renders figures at twice the requested density and displays them
    # at 100 dpi. Keep 72 dpi to limit the exported page size.
    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return factorial, md, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Interpolación polinomial

        Dados $n+1$ puntos $(x_i,y_i)$ con $x_i$ distintos, existe un único
        polinomio $p$ de grado $\le n$ con $p(x_i)=y_i$. Hay varias formas de
        escribirlo; todas dan el mismo polinomio (en aritmética exacta) y
        difieren en el costo y en cómo se actualizan cuando llega un punto
        nuevo.

        | forma | expresión | punto nuevo |
        |---|---|---|
        | Lagrange | $p(x)=\sum_i y_i L_i(x)$, $L_i(x)=\prod_{j\ne i}\dfrac{x-x_j}{x_i-x_j}$ | se recalcula todo |
        | Newton | $p(x)=\sum_k [x_0,\ldots,x_k]\prod_{j<k}(x-x_j)$ | se agrega un término |
        """
    )
    return


@app.cell
def _(np):
    def lagrange(x, y, t):
        """Valor en t del polinomio de Lagrange que pasa por (x, y)."""
        t = np.asarray(t, dtype=float)
        total = np.zeros_like(t)
        for i in range(len(x)):
            base = np.ones_like(t)
            for j in range(len(x)):
                if i != j:
                    base *= (t - x[j]) / (x[i] - x[j])
            total += y[i] * base
        return total

    def diferencias_divididas(x, y):
        """Columnas de la tabla de diferencias divididas.

        La columna j tiene n - j entradas: [x_i, ..., x_{i+j}] para i = 0...
        """
        x = np.asarray(x, dtype=float)
        columnas = [np.asarray(y, dtype=float)]
        for j in range(1, len(x)):
            previa = columnas[-1]
            columnas.append((previa[1:] - previa[:-1]) / (x[j:] - x[: len(x) - j]))
        return columnas

    def newton(x, y, t, terminos=None):
        """Polinomio de Newton en t con `terminos` coeficientes (todos si None)."""
        coef = [c[0] for c in diferencias_divididas(x, y)][:terminos]
        resultado = np.full_like(np.asarray(t, dtype=float), coef[-1])
        for k in range(len(coef) - 2, -1, -1):
            resultado = resultado * (t - x[k]) + coef[k]
        return resultado

    return diferencias_divididas, lagrange, newton


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Lagrange con cuatro puntos

        Datos $x=1,2,3,4$ con $y=0.671,\ 0.620,\ 0.567,\ 0.520$, interpolados
        en $x=2.8$. Cada $L_i$ vale 1 en $x_i$ y 0 en los demás nodos.
        """
    )
    return


@app.cell
def _(lagrange, md, np):
    X_L = np.array([1.0, 2.0, 3.0, 4.0])
    Y_L = np.array([0.671, 0.620, 0.567, 0.520])
    valor_lagrange = float(lagrange(X_L, Y_L, np.array(2.8)))
    _pesos = [float(lagrange(X_L, np.eye(4)[i], np.array(2.8))) for i in range(4)]
    md(
        "\n".join(
            [
                "| $i$ | $x_i$ | $y_i$ | $L_i(2.8)$ | $y_iL_i(2.8)$ |",
                "|---:|---:|---:|---:|---:|",
                *[
                    f"| {i} | {X_L[i]:.0f} | {Y_L[i]:.3f} | {_pesos[i]:+.4f} | {Y_L[i] * _pesos[i]:+.5f} |"
                    for i in range(4)
                ],
                "",
                f"$p(2.8)=\\sum_i y_iL_i(2.8)={valor_lagrange:.5f}$. "
                f"Los $L_i(2.8)$ suman {sum(_pesos):.1f}.",
            ]
        )
    )
    return X_L, Y_L


@app.cell
def _(X_L, Y_L, lagrange, np, plt):
    _t = np.linspace(0.8, 4.2, 300)
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))
    for _i in range(4):
        _ax[0].plot(_t, lagrange(X_L, np.eye(4)[_i], _t), label=rf"$L_{_i}$")
    _ax[0].set_title("Polinomios de la base de Lagrange")
    _ax[0].legend()
    _ax[1].plot(_t, lagrange(X_L, Y_L, _t), label="$p(x)$")
    _ax[1].plot(X_L, Y_L, "ko", label="datos")
    _ax[1].plot([2.8], lagrange(X_L, Y_L, np.array([2.8])), "rs", label="$p(2.8)$")
    _ax[1].set_title("Polinomio interpolante")
    _ax[1].legend()
    for _a in _ax:
        _a.set_xlabel("$x$")
        _a.grid(True, alpha=0.4)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Newton: diferencias divididas

        $[x_i]=y_i$ y
        $[x_i,\ldots,x_{i+j}]=\dfrac{[x_{i+1},\ldots,x_{i+j}]-[x_i,\ldots,x_{i+j-1}]}{x_{i+j}-x_i}$.
        El polinomio usa la primera fila de la tabla. Los datos son
        $J_0(x)$, la función de Bessel de primera especie y orden 0, con 5
        decimales, en siete nodos no equiespaciados. Se interpola en
        $x=0.28$ y el valor verdadero se calcula con la serie
        $J_0(x)=\sum_k\frac{(-1)^k}{(k!)^2}(x/2)^{2k}$.
        """
    )
    return


@app.cell
def _(diferencias_divididas, factorial, md, newton, np):
    X_N = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.2, 1.3])
    Y_N = np.array([0.99750, 0.99002, 0.96040, 0.88122, 0.76520, 0.67113, 0.62009])

    def bessel_j0(x):
        return sum(
            (-1) ** k * (x / 2) ** (2 * k) / factorial(k) ** 2 for k in range(20)
        )

    _tabla = diferencias_divididas(X_N, Y_N)
    _filas = []
    for _i in range(len(X_N)):
        _celdas = [f"{_tabla[_j][_i]:+.5f}" for _j in range(len(X_N) - _i)]
        _celdas += [""] * (len(X_N) - len(_celdas))
        _filas.append(f"| {X_N[_i]:.1f} | " + " | ".join(_celdas) + " |")
    _cabecera = "| $x_i$ | " + " | ".join(f"$j={_j}$" for _j in range(len(X_N))) + " |"
    _exacto = bessel_j0(0.28)
    _parciales = [
        float(newton(X_N, Y_N, np.array(0.28), terminos=_m)) for _m in range(1, 8)
    ]
    md(
        "\n".join(
            [
                _cabecera,
                "|---:|" + "---:|" * len(X_N),
                *_filas,
                "",
                f"$J_0(0.28)={_exacto:.7f}$. Sumando un término de Newton a la vez:",
                "",
                "| términos | grado | $p(0.28)$ | error |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {_m + 1} | {_m} | {_p:.7f} | {abs(_p - _exacto):.1e} |"
                    for _m, _p in enumerate(_parciales)
                ],
            ]
        )
    )
    return X_N, Y_N, bessel_j0


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Desde el grado 4 el error es del orden de $5\times10^{-6}$, el mismo
        que el redondeo de los datos a 5 decimales (hasta $5\times10^{-6}$ por
        dato): ya no mejora con más nodos. Cada fila agrega un término sin tocar los
        anteriores. En la forma de Lagrange habría que recalcular todos los
        $L_i$.

        Que Lagrange y Newton son el mismo polinomio se comprueba evaluando
        ambos en una malla:
        """
    )
    return


@app.cell
def _(X_N, Y_N, lagrange, md, newton, np):
    _t = np.linspace(0.1, 1.3, 500)
    _diferencia = np.max(np.abs(lagrange(X_N, Y_N, _t) - newton(X_N, Y_N, _t)))
    md(
        f"Máxima diferencia entre Lagrange y Newton sobre $[0.1,1.3]$: "
        f"{_diferencia:.1e}, es decir, redondeo de `float64`."
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Nodos equiespaciados: diferencias finitas

        Con paso $h$ constante, $[x_i,\ldots,x_{i+j}]=\Delta^jf_i/(j!\,h^j)$ y
        el polinomio queda en términos de las diferencias finitas. Con
        $s=(x-x_0)/h$, hacia adelante

        $$
        p(x)=\sum_{k=0}^n\binom sk\Delta^kf_0,\qquad
        \binom sk=\frac{s(s-1)\cdots(s-k+1)}{k!},
        $$

        y con $s=(x-x_n)/h$, hacia atrás (con $\nabla f_i=f_i-f_{i-1}$)

        $$
        p(x)=\sum_{k=0}^n\binom{s+k-1}{k}\nabla^kf_n .
        $$

        Dos tablas con paso constante. La primera, $x=150,160,\ldots,210$ con
        $y=35.5,37.8,43.6,45.7,47.3,50.1,51.2$, se interpola en $x=162$ con la
        fórmula hacia adelante. La segunda, $x=500,700,\ldots,1900$ con
        $y=365,361.6,370.74,379.68,384.46,395.5,395.95,397$, se interpola en
        $x=1000$ con la fórmula hacia atrás. Ambas se comparan con las
        diferencias divididas.
        """
    )
    return


@app.cell
def _(diferencias_divididas, md, newton, np):
    def binomial_generalizado(s, k):
        """s (s - 1) ... (s - k + 1) / k!"""
        resultado = 1.0
        for i in range(k):
            resultado *= (s - i) / (i + 1)
        return resultado

    def tabla_diferencias(y):
        columnas = [np.asarray(y, dtype=float)]
        while len(columnas[-1]) > 1:
            columnas.append(np.diff(columnas[-1]))
        return columnas

    def newton_adelante(x0, h, y, xi):
        s = (xi - x0) / h
        tabla = tabla_diferencias(y)
        return sum(binomial_generalizado(s, k) * tabla[k][0] for k in range(len(tabla)))

    def newton_atras(xn, h, y, xi):
        s = (xi - xn) / h
        tabla = tabla_diferencias(y)
        return sum(
            binomial_generalizado(s + k - 1, k) * tabla[k][-1]
            for k in range(len(tabla))
        )

    _x_a = 150.0 + 10.0 * np.arange(7)
    _y_a = np.array([35.5, 37.8, 43.6, 45.7, 47.3, 50.1, 51.2])
    _x_b = 500.0 + 200.0 * np.arange(8)
    _y_b = np.array([365.0, 361.6, 370.74, 379.68, 384.46, 395.5, 395.95, 397.0])
    _adelante = newton_adelante(_x_a[0], 10.0, _y_a, 162.0)
    _atras = newton_atras(_x_b[-1], 200.0, _y_b, 1000.0)
    _dd_a = float(newton(_x_a, _y_a, np.array(162.0)))
    _dd_b = float(newton(_x_b, _y_b, np.array(1000.0)))

    def _tabla_md(y):
        _t = tabla_diferencias(y)
        _n = len(y)
        _cab = "| $i$ | " + " | ".join(f"$\\Delta^{j}$" for j in range(_n)) + " |"
        _filas = []
        for _i in range(_n):
            _celdas = [f"{_t[_j][_i]:.2f}" if _i < _n - _j else "" for _j in range(_n)]
            _filas.append(f"| {_i} | " + " | ".join(_celdas) + " |")
        return [_cab, "|---:|" + "---:|" * _n, *_filas]

    md(
        "\n".join(
            [
                "Tabla de diferencias hacia adelante de la primera serie:",
                "",
                *_tabla_md(_y_a),
                "",
                "| serie | fórmula | $x$ | $p(x)$ | diferencias divididas |",
                "|---|---|---:|---:|---:|",
                f"| 1 | adelante | 162 | {_adelante:.6f} | {_dd_a:.6f} |",
                f"| 2 | atrás | 1000 | {_atras:.6f} | {_dd_b:.6f} |",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Las dos formas coinciden hasta el redondeo. La de adelante construye el
        polinomio desde el primer nodo y la de atrás desde el último, y sirven
        para interpolar cerca de ese extremo de la tabla. En la tabla de
        diferencias de la primera serie, $\Delta^k$ no decrece con $k$ (2.3,
        3.5, $-7.2$, 10.4, $-11.9$, 8.8): los datos tienen ruido, y el polinomio
        de grado 6 que pasa por todos ellos lo incorpora. Truncar la serie en
        un orden bajo suele dar un valor más razonable que usar todos los
        nodos.

        ## El fenómeno de Runge

        Más nodos no significa mejor aproximación. El error de interpolación es

        $$
        f(x)-p(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^n(x-x_i),
        $$

        y el producto depende solo de los nodos. Para nodos equiespaciados es
        pequeño en el centro y grande cerca de los extremos. Con
        $f(x)=\dfrac1{1+25x^2}$ en $[-1,1]$, las derivadas crecen tan rápido
        (hay polos en $x=\pm i/5$) que el error crece con el grado.
        Los nodos de Chebyshev, $x_i=\cos\frac{(2i+1)\pi}{2(n+1)}$, se agrupan
        hacia los extremos y hacen pequeño el producto.
        """
    )
    return


@app.cell
def _(lagrange, np):
    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    def nodos_equiespaciados(n):
        return np.linspace(-1.0, 1.0, n + 1)

    def nodos_chebyshev(n):
        return np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * (n + 1)))

    MALLA = np.linspace(-1.0, 1.0, 4001)
    GRADOS = np.arange(2, 41, 2)

    def error_maximo(nodos, n, malla):
        x = nodos(n)
        return np.max(np.abs(lagrange(x, runge(x), malla) - runge(malla)))

    errores_runge = {
        "equiespaciados": np.array(
            [error_maximo(nodos_equiespaciados, n, MALLA) for n in GRADOS]
        ),
        "Chebyshev": np.array(
            [error_maximo(nodos_chebyshev, n, MALLA) for n in GRADOS]
        ),
    }
    _centro = MALLA[np.abs(MALLA) <= 0.5]
    errores_centro = np.array(
        [error_maximo(nodos_equiespaciados, n, _centro) for n in GRADOS]
    )
    return (
        GRADOS,
        MALLA,
        errores_centro,
        errores_runge,
        nodos_chebyshev,
        nodos_equiespaciados,
        runge,
    )


@app.cell
def _(
    GRADOS,
    MALLA,
    errores_runge,
    lagrange,
    nodos_chebyshev,
    nodos_equiespaciados,
    np,
    plt,
    runge,
):
    _fig, _ax = plt.subplots(1, 2, figsize=(13, 4.5))
    _n = 14
    _ax[0].plot(MALLA, runge(MALLA), "k", lw=2, label=r"$1/(1+25x^2)$")
    for _nombre, _nodos in (
        ("equiespaciados", nodos_equiespaciados),
        ("Chebyshev", nodos_chebyshev),
    ):
        _x = _nodos(_n)
        _ax[0].plot(MALLA, lagrange(_x, runge(_x), MALLA), label=_nombre)
    _ax[0].plot(
        nodos_equiespaciados(_n),
        runge(nodos_equiespaciados(_n)),
        "ko",
        ms=4,
        label="nodos equiespaciados",
    )
    _ax[0].set_ylim(-0.6, 1.4)
    _ax[0].set_title(f"Polinomio de grado {_n}")
    _ax[0].set_xlabel("$x$")
    _ax[0].legend(loc="lower center")
    _rho = (1 + np.sqrt(26)) / 5
    for _nombre, _e in errores_runge.items():
        _ax[1].semilogy(GRADOS, _e, "o-", ms=4, label=_nombre)
    _ax[1].semilogy(
        GRADOS,
        errores_runge["Chebyshev"][-1] * _rho ** (GRADOS[-1] - GRADOS),
        ":",
        color="k",
        label=r"$\rho^{-n}$, $\rho=(1+\sqrt{26})/5$",
    )
    _ax[1].set_title("Error máximo en $[-1,1]$")
    _ax[1].set_xlabel("grado $n$")
    _ax[1].legend()
    for _a in _ax:
        _a.grid(True, which="both", alpha=0.4)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(GRADOS, errores_centro, errores_runge, md):
    _i = {int(n): k for k, n in enumerate(GRADOS)}
    _grados = (6, 10, 20, 30, 40)
    md(
        "\n".join(
            [
                "Error máximo sobre una malla de 4001 puntos:",
                "",
                "| grado $n$ | equiespaciados, $[-1,1]$ | equiespaciados, $[-0.5,0.5]$ | Chebyshev, $[-1,1]$ |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {n} | {errores_runge['equiespaciados'][_i[n]]:.1e} "
                    f"| {errores_centro[_i[n]]:.1e} "
                    f"| {errores_runge['Chebyshev'][_i[n]]:.1e} |"
                    for n in _grados
                ],
                "",
                "Con nodos equiespaciados el error en $[-1,1]$ crece con $n$ (las",
                "oscilaciones cerca de $x=\\pm1$ de la gráfica izquierda), pero en",
                "el centro, $|x|\\le0.5$, sigue bajando. Con nodos de Chebyshev el error baja",
                "geométricamente, con la razón $\\rho^{-1}$ que fija la distancia de los",
                "polos de $f$ al intervalo. La causa está en los nodos, no en la",
                "función ni en la aritmética: interpolar con polinomios de grado alto",
                "en nodos equiespaciados es una mala idea aunque no haya error de",
                "redondeo.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
