import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import math
    import struct
    from decimal import Decimal
    from fractions import Fraction
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["figure.dpi"] = 72

    def md(text):
        return mo.md(dedent(text))

    return Decimal, Fraction, math, md, np, plt, struct


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Error de punto flotante

        Un número real se guarda como $(-1)^s\,(1.m)_2\times2^{e}$: un signo, un
        exponente y una mantisa de longitud fija. Solo unos cuantos reales
        caben, y el resto se redondea al más cercano. De ahí salen todos los
        errores de este cuaderno.

        | tipo | NumPy | Fortran | signo | exponente | mantisa |
        |---|---|---|---:|---:|---:|
        | simple | `float32` | `real` | 1 | 8 (sesgo 127) | 23 |
        | doble | `float64` (el `float` de Python) | `double precision` | 1 | 11 (sesgo 1023) | 52 |

        ## Representación binaria

        Un entero se convierte a base 2 dividiendo entre 2 y juntando los
        restos, del último al primero. La conversión inversa suma $r_i2^i$.
        """
    )
    return


@app.cell
def _():
    def a_binario(n):
        if n == 0:
            return "0"
        restos = []
        while n:
            restos.append(str(n % 2))
            n //= 2
        return "".join(reversed(restos))

    def de_binario(cadena):
        return sum(int(r) * 2**i for i, r in enumerate(reversed(cadena)))

    return a_binario, de_binario


@app.cell
def _(a_binario, de_binario, md):
    _enteros = [0, 1, 5, 10, 255, 1000, 2**20 + 1]
    _filas = [
        f"| {n} | `{a_binario(n)}` | `{bin(n)[2:]}` | {de_binario(a_binario(n))} |"
        for n in _enteros
    ]
    assert all(a_binario(n) == bin(n)[2:] for n in range(2000))
    assert all(de_binario(a_binario(n)) == n for n in range(2000))
    md(
        "\n".join(
            [
                "| $n$ | división entre 2 | `bin(n)` | de vuelta a decimal |",
                "|---:|---|---|---:|",
                *_filas,
                "",
                "Los dos algoritmos coinciden con `bin` e `int(·, 2)` para los enteros de 0 a 1999.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Las fracciones son más delicadas: $\tfrac1{10}$ no tiene desarrollo
        binario finito ($0.0\overline{0011}_2$), igual que $\tfrac13$ no tiene
        uno decimal finito. `float.hex` muestra el número que realmente se
        guarda, exacto, como una mantisa hexadecimal y un exponente de 2, y
        `Decimal(x)` muestra su valor decimal exacto.
        """
    )
    return


@app.cell
def _(Decimal, md, struct):
    def campos64(x):
        """Signo, exponente y mantisa de un float64 como cadenas de bits."""
        (bits,) = struct.unpack(">Q", struct.pack(">d", x))
        texto = f"{bits:064b}"
        return texto[0], texto[1:12], texto[12:]

    def campos32(x):
        (bits,) = struct.unpack(">I", struct.pack(">f", x))
        texto = f"{bits:032b}"
        return texto[0], texto[1:9], texto[9:]

    _valores = [
        ("1", 1.0),
        ("0.5", 0.5),
        ("0.1", 0.1),
        ("0.1 + 0.2", 0.1 + 0.2),
        ("0.3", 0.3),
    ]
    _filas = []
    for _nombre, _x in _valores:
        _s, _e, _m = campos64(_x)
        _filas.append(
            f"| {_nombre} | `{float.hex(_x)}` | `{_s} {_e} {_m}` | {Decimal(_x)} |"
        )
    _s, _e, _m = campos32(0.1)
    md(
        "\n".join(
            [
                "| número | `float.hex` | signo, exponente, mantisa (`float64`) | valor exacto |",
                "|---|---|---|---|",
                *_filas,
                "",
                f"En `float32`, $0.1$ es `{_s} {_e} {_m}` (valor "
                f"{Decimal(struct.unpack('>f', struct.pack('>f', 0.1))[0])}).",
                "",
                "`0.1 + 0.2` y `0.3` difieren en el último dígito hexadecimal (`...34` y",
                "`...33`): son dos números de `float64` vecinos, y `0.1 + 0.2 == 0.3` es",
                "falso. Para comparar resultados de punto flotante se usa una tolerancia,",
                "no `==`.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Épsilon de máquina

        El épsilon de máquina $\varepsilon$ es la distancia entre $1$ y el
        siguiente número representable: $2^{-23}$ en `float32` y $2^{-52}$ en
        `float64`. Un número $x$ guardado con redondeo al más cercano tiene
        un error relativo de a lo sumo $\varepsilon/2$ (la *unidad de
        redondeo*).

        El algoritmo clásico divide $e$ entre 2 mientras $1+e>1$. Este es su
        resultado y el de `numpy.finfo`:
        """
    )
    return


@app.cell
def _(np):
    def epsilon_por_biseccion(tipo):
        """Divide e entre 2 mientras 1 + e > 1. Devuelve el último e y el previo."""
        uno = tipo(1)
        e = tipo(1)
        previo = e
        while uno + e > uno:
            previo = e
            e = e / tipo(2)
        return e, previo

    resultados_eps = {}
    for _tipo in (np.float32, np.float64):
        _final, _previo = epsilon_por_biseccion(_tipo)
        _info = np.finfo(_tipo)
        resultados_eps[_tipo.__name__] = (_final, _previo, _info)
    return epsilon_por_biseccion, resultados_eps


@app.cell
def _(md, np, resultados_eps):
    _filas = []
    for _nombre, (_final, _previo, _info) in resultados_eps.items():
        _filas.append(
            f"| `{_nombre}` | {float(_final):.4e} | {float(_previo):.4e} "
            f"| {float(_info.eps):.4e} | {float(_info.eps) / 2:.4e} "
            f"| {float(_info.tiny):.3e} | {float(_info.max):.3e} | {_info.precision} |"
        )
    _eps32 = np.finfo(np.float32).eps
    md(
        "\n".join(
            [
                "| tipo | último $e$ del bucle | $e$ previo | `finfo.eps` | `eps/2` | `tiny` | `max` | cifras decimales |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                "El bucle termina en el primer $e$ con $1+e=1$, que es $\\varepsilon/2$:",
                "$1+\\varepsilon/2$ está a medio camino entre dos números y el redondeo",
                "al par lo manda de vuelta a $1$. El valor anterior, $e$ previo, sí es",
                "$\\varepsilon$. Un programa que imprime el último $e$ del bucle",
                "informa $\\varepsilon/2$, la mitad del épsilon de máquina.",
                "",
                f"`numpy.finfo` da $\\varepsilon$ directamente ({float(_eps32):.4e} en `float32`).",
                "`tiny` es el menor normal positivo y `max` el mayor número finito.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El épsilon es la separación solo cerca de $1$. En general la separación
        entre vecinos crece con $|x|$, `numpy.spacing(x)`, de modo que el error
        *relativo* se mantiene y el *absoluto* no:
        """
    )
    return


@app.cell
def _(md, np):
    _xs = [1.0, 1e3, 1e8, 2.0**24, 2.0**53]
    _filas = []
    for _x in _xs:
        _s32 = np.spacing(np.float32(_x))
        _s64 = np.spacing(np.float64(_x))
        _filas.append(
            f"| {_x:.6g} | {float(_s32):.3e} | {float(_s32) / _x:.2e} "
            f"| {float(_s64):.3e} | {float(_s64) / _x:.2e} |"
        )
    md(
        "\n".join(
            [
                "| $x$ | `spacing`, `float32` | relativo | `spacing`, `float64` | relativo |",
                "|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                "En `float32`, $2^{24}=16\\,777\\,216$ es el primer entero que no tiene",
                "sucesor entero: $2^{24}+1$ se redondea a $2^{24}$. En `float64`",
                "ocurre en $2^{53}$.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Acumulación de errores de redondeo: sumar 0.00001 muchas veces

        Se suma $10^{-5}$ diez mil veces a $1$. El valor exacto de la suma de
        los números decimales $1+10^4\cdot10^{-5}$ es $1.1$. Cuatro
        variantes:

        1. acumulador `float64` y sumando `float64`;
        2. acumulador `float64` y sumando `float32` ($10^{-5}$ redondeado a
           simple, el efecto de escribir `0.00001` en un programa Fortran de
           doble precisión: una constante real es de precisión simple);
        3. todo en `float32`;
        4. todo en `float32`, sumando primero 100 grupos de 100 sumandos y
           después los grupos (suma por bloques).
        """
    )
    return


@app.cell
def _(np):
    def suma_naiva(valor, acumulador, n):
        total = acumulador
        for _ in range(n):
            total = total + valor
        return total

    def suma_por_bloques(valor, acumulador, bloques, tamano):
        total = acumulador
        for _ in range(bloques):
            parcial = type(valor)(0)
            for _ in range(tamano):
                parcial = parcial + valor
            total = total + parcial
        return total

    resultados_suma = {
        "1. `float64` + `float64`": float(
            suma_naiva(np.float64(1e-5), np.float64(1.0), 10_000)
        ),
        "2. `float64` + `float32`": float(
            suma_naiva(np.float64(np.float32(1e-5)), np.float64(1.0), 10_000)
        ),
        "3. todo `float32`": float(
            suma_naiva(np.float32(1e-5), np.float32(1.0), 10_000)
        ),
        "4. `float32` por bloques": float(
            suma_por_bloques(np.float32(1e-5), np.float32(1.0), 100, 100)
        ),
    }
    return resultados_suma, suma_naiva


@app.cell
def _(md, np, resultados_suma):
    _unidad = float(np.float32(1e-5))
    _filas = [
        f"| {n} | {v!r} | {abs(v - 1.1):.1e} |" for n, v in resultados_suma.items()
    ]
    md(
        "\n".join(
            [
                "| variante | resultado | $\\lvert\\mathrm{resultado}-1.1\\rvert$ |",
                "|---|---:|---:|",
                *_filas,
                "",
                f"$10^{{-5}}$ en `float32` es {_unidad!r}, que difiere de $10^{{-5}}$",
                f"en {abs(_unidad - 1e-5) / 1e-5:.1e} en relación. Esa diferencia",
                "se repite diez mil veces y da el error de la variante 2, que es un error de",
                "*representación*, no de acumulación: es del orden de $\\varepsilon_{32}/2$ por",
                "cada sumando. En la variante 3 el error crece mucho más porque cada suma",
                "redondea el resultado a 24 bits.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Para ver el crecimiento se suma $10^{-5}$ en `float32` hasta
        $N=2\times10^5$ de tres maneras y se compara con la suma exacta de los
        mismos sumandos: acumulando en serie, con la suma por pares de NumPy
        (`numpy.sum`) y acumulando en serie en `float64`.
        """
    )
    return


@app.cell
def _(Fraction, np):
    N_MAX = 200_000
    _unidad32 = np.float32(1e-5)
    _paso = np.full(N_MAX, _unidad32)

    def _exacta(n):
        return 1 + n * Fraction(float(_unidad32))

    def _error(suma, n):
        return float(abs(Fraction(float(suma)) - _exacta(n)))

    _ns = np.unique(np.logspace(1, np.log10(N_MAX), 40).astype(int))

    _serie32 = np.cumsum(np.concatenate([[np.float32(1.0)], _paso]), dtype=np.float32)
    _serie64 = np.cumsum(
        np.concatenate([[1.0], _paso.astype(np.float64)]), dtype=np.float64
    )
    ns_suma = _ns
    error_suma = {
        "serie, `float32`": np.array([_error(_serie32[n], n) for n in _ns]),
        "serie, `float64`": np.array([_error(_serie64[n], n) for n in _ns]),
        "por pares, `float32`": np.array(
            [
                _error(np.sum(np.concatenate([[np.float32(1.0)], _paso[:n]])), n)
                for n in _ns
            ]
        ),
    }
    return N_MAX, error_suma, ns_suma


@app.cell
def _(error_suma, np, ns_suma, plt):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    for _nombre, _e in error_suma.items():
        if _e.any():
            _ax.loglog(ns_suma[_e > 0], _e[_e > 0], "o-", ms=3, label=_nombre)
    _ax.loglog(
        ns_suma,
        np.finfo(np.float32).eps / 2 * ns_suma,
        ":",
        color="k",
        label=r"$N\,\varepsilon_{32}/2$",
    )
    _ax.set_xlabel("$N$ sumandos de $10^{-5}$")
    _ax.set_ylabel("error absoluto de la suma")
    _ax.set_title("Error de $1+N\\cdot10^{-5}$ (los sumandos son `float32`)")
    _ax.grid(True, which="both", alpha=0.4)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(error_suma, md, ns_suma):
    _i = len(ns_suma) - 1
    md(
        f"""
        En $N={ns_suma[_i]}$ el error de la serie en `float32` es
        {error_suma["serie, `float32`"][_i]:.1e}, el de la suma por pares
        {error_suma["por pares, `float32`"][_i]:.1e} y el de `float64`
        {error_suma["serie, `float64`"][_i]:.1e}. Cada suma en serie redondea un
        resultado de tamaño $\\approx1$, de modo que cada sumando de $10^{{-5}}$
        pierde cifras: ese es el error de *absorción*. La suma por pares suma
        primero números de tamaño parecido, y su error se queda en el redondeo
        del resultado final ($\\sim\\varepsilon_{{32}}$). En `float64`
        la suma sale exacta: cada sumando tiene 24 bits significativos y
        $N\\le2^{{18}}$ sumandos necesitan a lo sumo 42 bits, menos que los
        53 de la mantisa, así que su error es cero y no aparece en la gráfica.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Absorción: un término pequeño que desaparece

        Se evalúa $d(\theta)=\cos 1-\tfrac12\theta\sin 1$ para
        $\theta=10^{-1},10^{-2},\ldots,10^{-12}$ (el desarrollo de Taylor de
        $\cos(1+\theta/2)$ a primer orden en $\theta$). Cuando
        $\tfrac12\theta\sin 1$ es menor que la mitad de la separación entre
        números cercanos a $\cos1\approx0.54$, restarlo no cambia el resultado:
        $d(\theta)=\cos 1$.
        """
    )
    return


@app.cell
def _(md, np):
    _filas = []
    _primera32 = _primera64 = None
    _theta32 = np.float32(0.1)
    _theta64 = 0.1
    _cos32 = np.cos(np.float32(1))
    _sin32 = np.sin(np.float32(1))
    for _i in range(1, 13):
        _d32 = _cos32 - np.float32(0.5) * _theta32 * _sin32
        _d64 = np.cos(1.0) - 0.5 * _theta64 * np.sin(1.0)
        _efecto32 = float(_cos32 - _d32)
        _efecto64 = float(np.cos(1.0) - _d64)
        _termino = 0.5 * float(_theta32) * float(np.sin(1.0))
        if _primera32 is None and _efecto32 == 0:
            _primera32 = _i
        if _primera64 is None and _efecto64 == 0:
            _primera64 = _i
        _filas.append(
            f"| $10^{{-{_i}}}$ | {_termino:.3e} | {_efecto32:.3e} "
            f"| {abs(_efecto32 - _termino) / _termino:.1e} | {_efecto64:.3e} |"
        )
        _theta32 = _theta32 / np.float32(10)
        _theta64 = _theta64 / 10
    md(
        "\n".join(
            [
                "| $\\theta$ | $\\tfrac12\\theta\\sin1$ | $\\cos1-d$, `float32` | error relativo | $\\cos1-d$, `float64` |",
                "|---:|---:|---:|---:|---:|",
                *_filas,
                "",
                f"En `float32` el término se pierde por completo desde $\\theta=10^{{-{_primera32}}}$.",
                "En `float64` no se pierde en este rango ($10^{-12}\\gg\\varepsilon_{64}$)."
                if _primera64 is None
                else f"En `float64` se pierde desde $\\theta=10^{{-{_primera64}}}$.",
                "",
                "Antes de desaparecer, el término ya sale con error en `float32`: el error",
                "relativo del efecto de $\\theta$ crece al bajar $\\theta$, porque lo que se",
                "resta es del tamaño de la separación entre números cercanos a $\\cos1$.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Cancelación: restar números casi iguales

        La resta de dos números cercanos es exacta, pero deja visible el error
        que ya tenían. Para $x$ pequeño, $1-\cos x\approx x^2/2$ se calcula
        restando dos números cercanos a $1$, de modo que $\cos x$ guarda solo
        unas pocas cifras de lo que importa. La identidad
        $1-\cos x=2\sin^2(x/2)$ evita la resta. El cociente
        $(1-\cos x)/x^2$ debe tender a $\tfrac12$.
        """
    )
    return


@app.cell
def _(md, np):
    _filas = []
    for _k in range(1, 11):
        _x = 10.0**-_k
        _directa = (1 - np.cos(_x)) / _x**2
        _estable = 2 * np.sin(_x / 2) ** 2 / _x**2
        _x32 = np.float32(_x)
        _directa32 = (np.float32(1) - np.cos(_x32)) / _x32**2
        _filas.append(
            f"| $10^{{-{_k}}}$ | {_directa32:.6f} | {_directa:.12f} | {_estable:.12f} |"
        )
    md(
        "\n".join(
            [
                "| $x$ | directa, `float32` | directa, `float64` | $2\\sin^2(x/2)/x^2$ |",
                "|---:|---:|---:|---:|",
                *_filas,
                "",
                "La fórmula directa pierde cifras al reducir $x$ y acaba en $0$ cuando",
                "$\\cos x$ se redondea a $1$ (alrededor de $x\\lesssim\\sqrt{\\varepsilon}$: $10^{-4}$ en",
                "`float32`, $10^{-8}$ en `float64`). La forma con $\\sin^2$ da $0.5$",
                "con todas sus cifras. Es la misma causa que el piso de redondeo en la",
                "derivación numérica: restar valores casi iguales y dividir entre un",
                "número pequeño.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
