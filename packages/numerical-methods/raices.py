import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from decimal import Decimal, getcontext
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return Decimal, getcontext, md, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Solución de ecuaciones no lineales

        Se busca $x^*$ con $f(x^*)=0$. Tres métodos, con lo que exigen y lo que
        dan:

        | método | necesita | iteración | convergencia |
        |---|---|---|---|
        | bisección | $f(a)f(b)<0$ | $c=\frac{a+b}{2}$, se queda con el lado donde $f$ cambia de signo | lineal, razón $\tfrac12$ |
        | falsa posición | $f(a)f(b)<0$ | $c=\dfrac{af(b)-bf(a)}{f(b)-f(a)}$ (corte de la cuerda con el eje), mismo criterio de signo | lineal, razón depende de $f$ |
        | Newton | $f'$ y un punto inicial | $x_{k+1}=x_k-\dfrac{f(x_k)}{f'(x_k)}$ | cuadrática en una raíz simple |

        La convergencia de orden $p$ significa $e_{k+1}\approx C\,e_k^{\,p}$, con
        $e_k=|x_k-x^*|$. En una gráfica semilogarítmica de $e_k$ contra $k$, un
        método lineal es una recta y uno cuadrático cae cada vez más rápido: el
        número de cifras correctas se duplica en cada paso.
        """
    )
    return


@app.cell
def _(np):
    def biseccion(f, a, b, n):
        """n pasos de bisección. Devuelve los puntos medios y las semianchuras.

        Después de cada paso se toma el punto medio del intervalo que queda. Su
        semianchura acota el error, que en la práctica puede ser mucho menor.
        """
        fa = f(a)
        if fa * f(b) > 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos")
        medios, anchos = [], []
        for _ in range(n):
            c = (a + b) / 2
            fc = f(c)
            if fc == 0:
                medios.append(c)
                anchos.append(0.0)
                break
            if np.sign(fc) == np.sign(fa):
                a, fa = c, fc
            else:
                b = c
            medios.append((a + b) / 2)
            anchos.append((b - a) / 2)
        return np.array(medios), np.array(anchos)

    def falsa_posicion(f, a, b, n):
        """n pasos de falsa posición. Devuelve las abscisas del corte."""
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos")
        cortes = []
        for _ in range(n):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            cortes.append(c)
            if fc == 0:
                break
            if np.sign(fc) == np.sign(fa):
                a, fa = c, fc
            else:
                b, fb = c, fc
        return np.array(cortes)

    def newton(f, df, x0, n, multiplicidad=1):
        """n pasos de Newton. `multiplicidad` = 1 es el método clásico."""
        x = x0
        iterados = []
        for _ in range(n):
            derivada = df(x)
            if derivada == 0:
                break
            x = x - multiplicidad * f(x) / derivada
            iterados.append(x)
        return np.array(iterados)

    return biseccion, falsa_posicion, newton


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Bisección con $f(x)=x^3+1$

        Con el intervalo $[-3,2]$, $f(-3)=-26<0<f(2)=9$ y la raíz es $-1$. En
        cada paso el intervalo se reduce a la mitad, así que tras $n$ pasos el
        punto medio del intervalo que queda está a menos de $(b-a)/2^{n+1}$ de
        la raíz. Para llegar a
        una tolerancia $\tau$ hacen falta
        $n\ge\log_2\frac{b-a}{\tau}-1$ pasos, sin importar $f$.
        """
    )
    return


@app.cell
def _(biseccion, md, np):

    def _f(x):
        return x**3 + 1

    _filas = []
    for _tol in (1e-2, 1e-4, 1e-6, 1e-8):
        _n = int(np.ceil(np.log2(5 / _tol) - 1))
        _medios, _anchos = biseccion(_f, -3.0, 2.0, _n)
        _filas.append(
            f"| {_tol:.0e} | {_n} | {_medios[-1]:.9f} | {abs(_medios[-1] + 1):.1e} "
            f"| {_anchos[-1]:.1e} |"
        )
    md(
        "\n".join(
            [
                "| tolerancia $\\tau$ | pasos $n$ | $x_n$ | error real | semianchura |",
                "|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                "El error real queda por debajo de la semianchura, que es la cota",
                "garantizada, y por eso $n$ no depende de dónde esté la raíz dentro del",
                "intervalo.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Los tres métodos en $f(x)=x^3+5x^2+2$

        $f'(x)=3x^2+10x$. Esta función tiene una sola raíz real,
        $x^*\approx-5.0776$ (las otras dos son complejas), y está en
        $[-6,-4]$, donde $f(-6)=-34<0<f(-4)=18$. Para medir el error con más
        cifras que `float64` se calcula la raíz con 60 dígitos (`decimal`) y se
        compara cada iterado, convertido exactamente, contra ella.
        """
    )
    return


@app.cell
def _(Decimal, getcontext):
    getcontext().prec = 60

    def f_cubica(x):
        return x**3 + 5 * x**2 + 2

    def df_cubica(x):
        return 3 * x**2 + 10 * x

    def raiz_alta_precision(x0):
        x = Decimal(x0)
        for _ in range(100):
            x -= (x**3 + 5 * x**2 + 2) / (3 * x**2 + 10 * x)
        return x

    RAIZ = raiz_alta_precision(-5)

    def error_absoluto(iterados):
        """|x_k - x*| calculado con 60 dígitos."""
        return [abs(Decimal(float(x)) - RAIZ) for x in iterados]

    return RAIZ, df_cubica, error_absoluto, f_cubica


@app.cell
def _(
    biseccion,
    df_cubica,
    error_absoluto,
    f_cubica,
    falsa_posicion,
    newton,
    np,
):
    _medios, anchos_biseccion = biseccion(f_cubica, -6.0, -4.0, 50)
    historias = {
        "bisección": _medios,
        "falsa posición": falsa_posicion(f_cubica, -6.0, -4.0, 40),
        "Newton ($x_0=-6$)": newton(f_cubica, df_cubica, -6.0, 6),
    }
    errores = {
        nombre: np.array([float(e) for e in error_absoluto(h)])
        for nombre, h in historias.items()
    }
    return anchos_biseccion, errores


@app.cell
def _(anchos_biseccion, errores, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _nombre, _e in errores.items():
        _k = np.arange(1, len(_e) + 1)
        _ax.semilogy(_k[_e > 0], _e[_e > 0], "o-", ms=3, label=_nombre)
    _k_ancho = np.arange(1, len(anchos_biseccion) + 1)
    _ax.semilogy(
        _k_ancho[anchos_biseccion > 0],
        anchos_biseccion[anchos_biseccion > 0],
        ":",
        color="k",
        label=r"cota de la bisección, $(b-a)/2^{k+1}$",
    )
    _ax.axhline(np.finfo(float).eps, color="gray", lw=0.8, ls="--")
    _ax.text(1, 1.5 * np.finfo(float).eps, r"$\varepsilon$", color="gray")
    _ax.set_ylim(1e-17, 10)
    _ax.set_xlabel("iteración $k$")
    _ax.set_ylabel(r"$|x_k-x^*|$")
    _ax.set_title(r"Error de cada método, $f=x^3+5x^2+2$")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(anchos_biseccion, errores, md, np):
    def _primero(e, tol):
        _k = np.flatnonzero(e < tol)
        return str(_k[0] + 1) if len(_k) else "no llega"

    _tols = (1e-4, 1e-8, 1e-12)
    _filas = [
        f"| {n} | " + " | ".join(_primero(e, t) for t in _tols) + " |"
        for n, e in errores.items()
    ]
    _filas.append(
        "| bisección, cota garantizada | "
        + " | ".join(_primero(anchos_biseccion, t) for t in _tols)
        + " |"
    )
    md(
        "\n".join(
            [
                "Primera iteración con error real menor que la tolerancia:",
                "",
                "| método | $10^{-4}$ | $10^{-8}$ | $10^{-12}$ |",
                "|---|---:|---:|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El error real de la bisección no baja de forma monótona (la raíz puede
        quedar cerca de un punto medio por azar), pero la cota sí, y es lo que
        se garantiza. Falsa posición baja casi como una recta, más rápido que
        la bisección. Newton lo hace en 5 pasos.

        ## Orden de convergencia observado

        Si $e_{k+1}\approx C\,e_k^{\,p}$ entonces
        $p\approx\dfrac{\ln(e_{k+1}/e_k)}{\ln(e_k/e_{k-1})}$. En una gráfica
        de $e_{k+1}$ contra $e_k$ en escala logarítmica, $p$ es la pendiente.
        Solo se usan los errores mayores que $10^{-14}$, antes de que el
        redondeo de `float64` domine. Para la bisección se usa la cota, que
        es la que se reduce exactamente a la mitad.
        """
    )
    return


@app.cell
def _(anchos_biseccion, errores, np):
    _series = {
        "bisección (cota)": anchos_biseccion,
        "falsa posición": errores["falsa posición"],
        "Newton ($x_0=-6$)": errores["Newton ($x_0=-6$)"],
    }
    ordenes = {}
    razones = {}
    for _nombre, _e in _series.items():
        _e = _e[_e > 1e-14]
        ordenes[_nombre] = np.log(_e[2:] / _e[1:-1]) / np.log(_e[1:-1] / _e[:-2])
        razones[_nombre] = _e[1:] / _e[:-1]
    series_validas = {n: e[e > 1e-14] for n, e in _series.items()}
    return ordenes, razones, series_validas


@app.cell
def _(plt, series_validas):
    _fig, _ax = plt.subplots(figsize=(6.5, 5))
    for _nombre, _e in series_validas.items():
        _ax.loglog(_e[:-1], _e[1:], "o-", ms=4, label=_nombre)
    _ax.loglog([1e-14, 1], [1e-14, 1], ":", color="gray", label=r"$e_{k+1}=e_k$")
    _ax.set_xlabel(r"$e_k$")
    _ax.set_ylabel(r"$e_{k+1}$")
    _ax.set_title(r"Pendiente $=p$")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md, np, ordenes, razones):
    _filas = [
        f"| {n} | {np.median(ordenes[n]):.2f} | {razones[n][-1]:.3f} |" for n in ordenes
    ]
    md(
        "\n".join(
            [
                "| método | $p$ (mediana) | $e_{k+1}/e_k$ al final |",
                "|---|---:|---:|",
                *_filas,
                "",
                "La bisección y falsa posición dan $p=1$ (convergencia lineal) con",
                "razones $0.5$ y $\\approx0.28$: la segunda es una constante de la función",
                "y el intervalo, la primera no depende de ellos. Newton da $p=2$ y las",
                "razones $e_{k+1}/e_k$ no se estabilizan: tienden a cero.",
                "",
                "En falsa posición uno de los extremos queda fijo en todas las",
                "iteraciones (la función es cóncava en el intervalo y la cuerda siempre",
                "corta del mismo lado), así que el intervalo no se achica y el criterio",
                "$|b-a|<\\tau$ de la bisección no sirve; se mide $|f|$ o el cambio entre",
                "iterados.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Cuando Newton falla o pierde el orden

        **Punto inicial.** Con el mismo $f$ y $x_0=-1$ se tiene $f(-1)=6$ y
        $f'(-1)=-7$, así que $x_1=-1-6/(-7)\approx-0.143$: el primer paso
        cae cerca de $x=0$, donde $f'=0$ y las tangentes son casi horizontales,
        y los iterados siguientes saltan sin acercarse a la raíz. Con
        $x_0=-6$ converge en 5 pasos. Newton no exige un corchete y por eso no
        garantiza nada lejos de la raíz.

        **Raíz múltiple.** Si $f(x)=(x-1)^2(x+2)$, la raíz $x=1$ es doble:
        $f'(1)=0$ y la convergencia baja a lineal con razón $\tfrac12$. Para una
        raíz de multiplicidad $m$, $x_{k+1}=x_k-m\,f(x_k)/f'(x_k)$ recupera el
        orden 2.
        """
    )
    return


@app.cell
def _(df_cubica, f_cubica, newton, np):
    desde_uno = newton(f_cubica, df_cubica, -1.0, 40)
    _raiz = -5.077574223045071
    iterados_malos = desde_uno
    n_cerca = int(np.sum(np.abs(desde_uno - _raiz) < 1e-6))

    def f_doble(x):
        return (x - 1) ** 2 * (x + 2)

    def df_doble(x):
        return (x - 1) * (3 * x + 3)

    error_doble = {
        "Newton clásico": np.abs(newton(f_doble, df_doble, 2.0, 14) - 1),
        "Newton con $m=2$": np.abs(newton(f_doble, df_doble, 2.0, 14, 2) - 1),
    }
    return error_doble, iterados_malos, n_cerca


@app.cell
def _(iterados_malos, md, n_cerca, np):
    _min = np.min(np.abs(iterados_malos))
    md(
        f"Desde $x_0=-1$, de los 40 iterados de Newton, {n_cerca} quedan a menos de "
        f"$10^{{-6}}$ de la raíz. Primer iterado: $x_1={iterados_malos[0]:.4f}$. "
        f"El iterado más cercano a $x=0$ está a {_min:.3f}."
    )
    return


@app.cell
def _(error_doble, np, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _nombre, _e in error_doble.items():
        _k = np.arange(1, len(_e) + 1)
        _ax.semilogy(_k[_e > 0], _e[_e > 0], "o-", ms=4, label=_nombre)
    _ax.set_xlabel("iteración $k$")
    _ax.set_ylabel(r"$|x_k-1|$")
    _ax.set_title(r"Raíz doble de $f=(x-1)^2(x+2)$, $x_0=2$")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(error_doble, md):
    _e = error_doble["Newton clásico"]
    md(
        rf"""
        Con la raíz doble el error de Newton clásico se reduce a la mitad en
        cada paso (de {_e[0]:.2e} a {_e[-1]:.2e} en 14 pasos, razón
        {_e[-1] / _e[-2]:.3f}). Con $m=2$ el error cae a cero (se anula en
        `float64`) en 5 pasos. El precio es saber $m$ de antemano.
        """
    )
    return


if __name__ == "__main__":
    app.run()
