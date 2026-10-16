import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Marimo renders figures at twice the requested density and displays them
    # at 100 dpi. Keep 72 dpi to limit the exported page size.
    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return md, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Derivación numérica

        Se aproxima $f'(x)$ con valores de $f$ en puntos cercanos. Todas las
        fórmulas salen del desarrollo de Taylor
        $f(x+h)=f(x)+hf'(x)+\tfrac{h^2}{2}f''(x)+\tfrac{h^3}{6}f'''(x)+\cdots$

        | fórmula | expresión | error de truncamiento |
        |---|---|---|
        | adelante | $\dfrac{f(x+h)-f(x)}{h}$ | $+\tfrac h2 f''(x)+O(h^2)$ |
        | atrás | $\dfrac{f(x)-f(x-h)}{h}$ | $-\tfrac h2 f''(x)+O(h^2)$ |
        | centrada | $\dfrac{f(x+h)-f(x-h)}{2h}$ | $+\tfrac{h^2}{6} f'''(x)+O(h^4)$ |
        | 3 puntos adelante | $\dfrac{-3f(x)+4f(x+h)-f(x+2h)}{2h}$ | $-\tfrac{h^2}{3} f'''(x)+O(h^3)$ |
        | 3 puntos atrás | $\dfrac{3f(x)-4f(x-h)+f(x-2h)}{2h}$ | $-\tfrac{h^2}{3} f'''(x)+O(h^3)$ |

        (el error es $D-f'$; la tabla muestra el primer término). El ejemplo es $f(x)=\tan x$ en $x=1$, con
        $f'(x)=\sec^2 x$. Los números de este cuaderno están en doble
        precisión (`float64`).
        """
    )
    return


@app.cell
def _(np):
    X0 = 1.0
    f = np.tan

    def derivada_exacta(x):
        return 1.0 / np.cos(x) ** 2

    def adelante(f, x, h):
        return (f(x + h) - f(x)) / h

    def atras(f, x, h):
        return (f(x) - f(x - h)) / h

    def centrada(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    def adelante_3(f, x, h):
        return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)

    def atras_3(f, x, h):
        return (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)

    formulas = {
        "atrás": atras,
        "adelante": adelante,
        "centrada": centrada,
        "3 puntos atrás": atras_3,
        "3 puntos adelante": adelante_3,
    }
    return (
        X0,
        adelante,
        adelante_3,
        atras,
        centrada,
        derivada_exacta,
        f,
        formulas,
    )


@app.cell
def _(X0, derivada_exacta, f, formulas, md):
    _exacta = derivada_exacta(X0)
    _filas = []
    for _h in (0.1, 0.05, 0.02):
        for _nombre, _formula in formulas.items():
            _d = _formula(f, X0, _h)
            _filas.append(
                f"| {_h} | {_nombre} | {_d:.6f} | {100 * (_exacta - _d) / _exacta:.4f} |"
            )
    md(
        "\n".join(
            [
                "## Valores para $h=0.1,\\ 0.05,\\ 0.02$",
                "",
                f"$f'(1)=\\sec^2 1={_exacta:.6f}$. El error es $100\\,(f'-D)/f'$.",
                "",
                "| $h$ | fórmula | $D$ | error (%) |",
                "|---|---|---:|---:|",
                *_filas,
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Al pasar de $h=0.1$ a $0.05$ el error de las fórmulas de orden 1 se
        divide entre 2, y el de las de orden 2 entre 4. Esa es la definición
        práctica de orden: $e(h)\approx C\,h^p$.

        ## Error contra $h$: el orden y el piso del redondeo

        Reducir $h$ no mejora la derivada sin límite. El numerador
        $f(x+h)-f(x)$ resta dos números casi iguales y pierde cifras. Cada
        valor de $f$ tiene un error relativo de hasta $\varepsilon/2$ (con
        $\varepsilon=2^{-52}$ el épsilon de máquina de `float64`), así que el
        numerador tiene un error absoluto del orden de $\varepsilon\,|f|$,
        que al dividir entre $h$ da

        $$
        e(h)\;\approx\;\underbrace{C\,h^p}_{\text{truncamiento}}
        \;+\;\underbrace{\frac{\varepsilon\,|f(x)|}{h}}_{\text{redondeo}} .
        $$

        El truncamiento baja con $h$ y el redondeo sube. El mínimo está en
        $h\sim\varepsilon^{1/(p+1)}$: $\approx10^{-8}$ para $p=1$ y
        $\approx10^{-5}$ para $p=2$.
        """
    )
    return


@app.cell
def _(X0, adelante, adelante_3, centrada, derivada_exacta, f, np):
    H = np.logspace(-1, -16, 151)
    _exacta = derivada_exacta(X0)
    errores = {
        "adelante": np.abs(adelante(f, X0, H) - _exacta),
        "centrada": np.abs(centrada(f, X0, H) - _exacta),
        "3 puntos adelante": np.abs(adelante_3(f, X0, H) - _exacta),
    }

    def pendiente(nombre, h_max, h_min):
        """Pendiente de log e contra log h en h_min <= h <= h_max."""
        _m = (H <= h_max) & (H >= h_min)
        return np.polyfit(np.log10(H[_m]), np.log10(errores[nombre][_m]), 1)[0]

    pendientes = {
        "adelante": pendiente("adelante", 1e-3, 1e-6),
        "centrada": pendiente("centrada", 1e-2, 1e-4),
        "3 puntos adelante": pendiente("3 puntos adelante", 1e-2, 1e-4),
    }
    h_optimo = {nombre: H[e.argmin()] for nombre, e in errores.items()}
    e_minimo = {nombre: e.min() for nombre, e in errores.items()}

    # Modelo e(h) = C h^p + eps |f| / h con las derivadas de tan en x = 1:
    # f'' = 2t(1+t^2), f''' = 2(1+t^2)(1+3t^2), t = tan x.
    _t = np.tan(X0)
    _eps = np.finfo(float).eps
    _c1 = abs(2 * _t * (1 + _t**2)) / 2
    _c2 = abs(2 * (1 + _t**2) * (1 + 3 * _t**2)) / 6
    h_modelo = {
        "adelante": np.sqrt(_eps * abs(_t) / _c1),
        "centrada": (_eps * abs(_t) / (2 * _c2)) ** (1 / 3),
    }
    e_modelo = {
        "adelante": _c1 * h_modelo["adelante"] + _eps * abs(_t) / h_modelo["adelante"],
        "centrada": _c2 * h_modelo["centrada"] ** 2
        + _eps * abs(_t) / h_modelo["centrada"],
    }
    return H, e_minimo, e_modelo, errores, h_modelo, h_optimo, pendientes


@app.cell
def _(H, errores, f, np, plt, X0):
    _eps = np.finfo(float).eps
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _nombre, _e in errores.items():
        _ax.loglog(H, _e, ".-", ms=3, label=_nombre)
    _ax.loglog(H, _eps * abs(f(X0)) / H, "k:", label=r"$\varepsilon\,|f|/h$")
    _ax.loglog(H, 2 * H, "--", color="gray", lw=0.8, label=r"$\propto h$")
    _ax.loglog(H, 5 * H**2, "-.", color="gray", lw=0.8, label=r"$\propto h^2$")
    _ax.set_ylim(1e-13, 1e1)
    _ax.set_xlabel(r"$h$")
    _ax.set_ylabel(r"$|D-f'(1)|$")
    _ax.set_title(r"Error de la derivada de $\tan x$ en $x=1$")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(e_minimo, e_modelo, h_modelo, h_optimo, md, pendientes):
    _filas = [
        f"| {n} | {pendientes[n]:.2f} | {h_optimo[n]:.1e} | {e_minimo[n]:.1e} "
        + (f"| {h_modelo[n]:.1e} | {e_modelo[n]:.1e} |" if n in h_modelo else "| | |")
        for n in pendientes
    ]
    md(
        "\n".join(
            [
                "La pendiente se ajusta en el tramo donde domina el truncamiento.",
                "El modelo es $e(h)=C\\,h^p+\\varepsilon|f|/h$ con $C$ tomado de las",
                "derivadas de $\\tan$ en $x=1$, y su mínimo está en $h$ del modelo.",
                "",
                "| fórmula | pendiente | $h$ óptimo medido | error mínimo medido | $h$ del modelo | error del modelo |",
                "|---|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                r"El mínimo medido es una sola muestra de una curva ruidosa: puede caer",
                r"más abajo o más arriba que el del modelo (los redondeos a veces se",
                r"compensan), pero el modelo da el orden de magnitud. A la izquierda del",
                r"mínimo el error crece como $1/h$ (pendiente $-1$) y se",
                r"vuelve irregular: es ruido de redondeo. Las fórmulas de orden 2",
                r"alcanzan un error mucho menor que la de orden 1, y lo hacen con un $h$",
                r"mayor, donde el redondeo todavía es pequeño.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Extrapolación de Richardson

        Si el error de una fórmula $D(h)$ tiene la forma
        $D(h)=f'+a_1h^{p_1}+a_2h^{p_2}+\cdots$, se combinan dos evaluaciones con
        pasos $h$ y $h/2$ para eliminar el primer término:

        $$
        D_1(h)=D(h/2)+\frac{D(h/2)-D(h)}{2^{p_1}-1}
        $$

        tiene error $O(h^{p_2})$. Repetir con $D_1$ elimina el siguiente
        término. Para la fórmula centrada los exponentes son $2,4,6,\ldots$
        (solo potencias pares); para la de adelante son $1,2,3,\ldots$.
        Con $D_{k,0}=D(h/2^k)$ y exponentes $p_1<p_2<\cdots$,

        $$
        D_{k,m}=D_{k,m-1}+\frac{D_{k,m-1}-D_{k-1,m-1}}{2^{p_m}-1}.
        $$

        No se necesita derivar $f$ más de una vez: solo evaluar $f$ en más
        puntos.
        """
    )
    return


@app.cell
def _(np):
    def richardson(formula, f, x, h, niveles, exponentes):
        """Tabla de Richardson: fila k usa los pasos h, h/2, ..., h/2**k.

        La columna m tiene error O(h**exponentes[m]). Las entradas de la
        derecha de la diagonal quedan en NaN.
        """
        d = np.full((niveles + 1, niveles + 1), np.nan)
        for k in range(niveles + 1):
            d[k, 0] = formula(f, x, h / 2**k)
        for m in range(1, niveles + 1):
            for k in range(m, niveles + 1):
                d[k, m] = d[k, m - 1] + (d[k, m - 1] - d[k - 1, m - 1]) / (
                    2 ** exponentes[m - 1] - 1
                )
        return d

    EXPONENTES_CENTRADA = [2, 4, 6, 8, 10]
    EXPONENTES_ADELANTE = [1, 2, 3, 4, 5]
    return EXPONENTES_ADELANTE, EXPONENTES_CENTRADA, richardson


@app.cell
def _(
    EXPONENTES_ADELANTE,
    EXPONENTES_CENTRADA,
    X0,
    adelante,
    centrada,
    derivada_exacta,
    f,
    md,
    np,
    richardson,
):
    _exacta = derivada_exacta(X0)
    _H0 = 0.2
    _tablas = {
        "centrada": richardson(centrada, f, X0, _H0, 4, EXPONENTES_CENTRADA),
        "adelante": richardson(adelante, f, X0, _H0, 4, EXPONENTES_ADELANTE),
    }
    _texto = [
        f"Con $h={_H0}$ y 4 niveles (el paso más chico es $h/16=0.0125$). "
        "Error absoluto $|D_{k,m}-f'(1)|$:",
        "",
    ]
    for _nombre, _d in _tablas.items():
        _texto += [
            f"**{_nombre}**",
            "",
            "| $k$ | $D_{k,0}$ | $D_{k,1}$ | $D_{k,2}$ | $D_{k,3}$ | $D_{k,4}$ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _k in range(_d.shape[0]):
            _celdas = [
                "" if np.isnan(_v) else f"{abs(_v - _exacta):.2e}" for _v in _d[_k]
            ]
            _texto.append(f"| {_k} | " + " | ".join(_celdas) + " |")
        _texto.append("")
    md("\n".join(_texto))
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        En la fórmula centrada, con el mismo paso mínimo $h/16$, el error baja de
        $1.5\times10^{-3}$ en $D_{4,0}$ a $9.7\times10^{-11}$ en $D_{4,4}$. En
        la fórmula de adelante las primeras columnas ganan un orden cada una,
        pero las últimas mejoran cada vez menos: con $h=0.2$ el paso todavía
        es grande para $\tan$ (la singularidad $\pi/2$ está a $0.57$ de
        $x=1$) y los términos de orden alto del error pesan. A continuación, el
        error contra $h$ de la fórmula centrada con uno y con dos niveles de
        Richardson.
        """
    )
    return


@app.cell
def _(EXPONENTES_CENTRADA, H, X0, centrada, derivada_exacta, f, np, richardson):
    _exacta = derivada_exacta(X0)
    _filas = [
        richardson(centrada, f, X0, _h, 2, EXPONENTES_CENTRADA) for _h in H[H > 1e-9]
    ]
    h_rich = H[H > 1e-9]
    error_rich = {
        "centrada": np.array([abs(_t[0, 0] - _exacta) for _t in _filas]),
        "1 nivel (orden 4)": np.array([abs(_t[1, 1] - _exacta) for _t in _filas]),
        "2 niveles (orden 6)": np.array([abs(_t[2, 2] - _exacta) for _t in _filas]),
    }
    pendientes_rich = {
        nombre: np.polyfit(
            np.log10(h_rich[(h_rich <= 0.2) & (h_rich >= 0.02)]),
            np.log10(e[(h_rich <= 0.2) & (h_rich >= 0.02)]),
            1,
        )[0]
        for nombre, e in error_rich.items()
    }
    return error_rich, h_rich, pendientes_rich


@app.cell
def _(error_rich, h_rich, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _nombre, _e in error_rich.items():
        _ax.loglog(h_rich, _e, ".-", ms=3, label=_nombre)
    _ax.set_ylim(1e-16, 1e1)
    _ax.set_xlabel(r"$h$ (paso mayor)")
    _ax.set_ylabel(r"$|D-f'(1)|$")
    _ax.set_title("Richardson sobre la diferencia centrada")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(error_rich, h_rich, md, np, pendientes_rich):
    _i = np.argmin(np.abs(h_rich - 1e-2))
    _filas = [
        f"| {n} | {pendientes_rich[n]:.2f} | {error_rich[n][_i]:.1e} | {error_rich[n].min():.1e} |"
        for n in error_rich
    ]
    md(
        "\n".join(
            [
                "| método | pendiente | error en $h=10^{-2}$ | error mínimo |",
                "|---|---:|---:|---:|",
                *_filas,
                "",
                "Las pendientes (ajustadas en $0.02\\le h\\le0.2$) son 2, 4 y 6, como",
                "predice la teoría. Richardson no cambia el redondeo, $\\varepsilon|f|/h$,",
                "pero reduce el truncamiento, así que el mínimo se alcanza con un $h$",
                "mayor, donde el redondeo es menor. Por eso el error mínimo baja.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Derivada a partir de una tabla de diferencias

        Cuando $f$ está tabulada con paso constante $h$ se arma la tabla de
        diferencias $\Delta^i f_j$ ($\Delta f_j=f_{j+1}-f_j$,
        $\Delta^{i}f_j=\Delta^{i-1}f_{j+1}-\Delta^{i-1}f_j$). Como
        $hD=\ln(1+\Delta)$, la derivada en $x_j$ es la serie

        $$
        f'(x_j)\approx\frac1h\sum_{i=1}^{t}\frac{(-1)^{i+1}}{i}\,\Delta^i f_j ,
        $$

        truncada en el orden $t$. Con diferencias hacia atrás
        ($\nabla f_j=f_j-f_{j-1}$) los signos desaparecen:
        $f'(x_j)\approx\frac1h\sum_{i=1}^{t}\frac1i\nabla^i f_j$.

        Con $t=1$ es la diferencia adelante (o atrás); con $t=2$ es la fórmula
        de 3 puntos. Cada orden agrega un punto a la fórmula y una potencia de
        $h$ al error. Aquí $f=\tan x$ tabulada con $h=0.02$; la derivada
        adelante se calcula en $x=1$ con los puntos $1,1.02,\ldots$ y la
        derivada atrás en $x=1$ con los puntos $\ldots,0.98,1$.
        """
    )
    return


@app.cell
def _(np):
    def tabla_diferencias(y):
        """Columnas i = 0..n-1 de las diferencias finitas de y (largo n - i)."""
        columnas = [np.asarray(y, dtype=float)]
        while len(columnas[-1]) > 1:
            columnas.append(np.diff(columnas[-1]))
        return columnas

    def derivada_serie_adelante(y, h, t):
        """f' en el primer punto de y con la serie de diferencias hacia adelante."""
        tabla = tabla_diferencias(y)
        return sum((-1) ** (i + 1) * tabla[i][0] / i for i in range(1, t + 1)) / h

    def derivada_serie_atras(y, h, t):
        """f' en el último punto de y con la serie de diferencias hacia atrás."""
        tabla = tabla_diferencias(y)
        return sum(tabla[i][-1] / i for i in range(1, t + 1)) / h

    return derivada_serie_adelante, derivada_serie_atras


@app.cell
def _(
    X0,
    derivada_exacta,
    derivada_serie_adelante,
    derivada_serie_atras,
    f,
    md,
    np,
):
    _orden_max = 6
    _exacta = derivada_exacta(X0)
    _columnas = {}
    for _h in (0.02, 0.1):
        _y_adelante = f(X0 + _h * np.arange(_orden_max + 1))
        _y_atras = f(X0 - _h * np.arange(_orden_max, -1, -1))
        _columnas[_h] = [
            (
                derivada_serie_adelante(_y_adelante, _h, _t) - _exacta,
                derivada_serie_atras(_y_atras, _h, _t) - _exacta,
            )
            for _t in range(1, _orden_max + 1)
        ]
    _filas = [
        f"| {_t} | {_columnas[0.02][_t - 1][0]:+.2e} | {_columnas[0.02][_t - 1][1]:+.2e} "
        f"| {_columnas[0.1][_t - 1][0]:+.2e} | {_columnas[0.1][_t - 1][1]:+.2e} |"
        for _t in range(1, _orden_max + 1)
    ]
    md(
        "\n".join(
            [
                "Error $D-f'(1)$ según el orden de truncamiento $t$.",
                "",
                "| $t$ | adelante, $h=0.02$ | atrás, $h=0.02$ | adelante, $h=0.1$ | atrás, $h=0.1$ |",
                "|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                "Con $h=0.02$ el error decrece con cada término, cada vez más",
                r"despacio, y las diferencias hacia adelante alternan de signo. Con",
                r"$h=0.1$ la serie hacia adelante se aleja de la derivada: sus puntos",
                r"llegan hasta $x=1.6$, más allá de la singularidad de $\tan$ en",
                r"$\pi/2\approx1.571$, y la serie solo sirve cuando $h$ es pequeño",
                r"frente a la escala en que $f$ cambia. La serie hacia atrás, que usa",
                r"puntos con $x\le1$, sí converge, pero más despacio que con",
                r"$h=0.02$. Agregar términos no reemplaza a reducir $h$.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
