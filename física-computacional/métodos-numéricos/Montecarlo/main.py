import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    def md(text):
        return mo.md(dedent(text))

    return md, mo, np, plt


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Integración de Monte Carlo

        La versión interactiva de este notebook usa Python y NumPy (WASM). Las rutinas en Fortran se incluyen como referencia y solo se ejecutan localmente.
    
        GitHub Pages no puede compilar Fortran ni cargar extensiones nativas (.so) generadas con f2py. Para ejecutar el notebook localmente con Fortran y f2py:

        ```bash
        git clone https://github.com/totallynotdavid/undergrad
        cd undergrad
        ./install.sh
        uv run --package fisica-computacional marimo edit \
          'física-computacional/métodos-numéricos/Montecarlo/fortran_f2py.py'
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        En física estadística, muchas cantidades observables se escriben como
        valores esperados sobre un espacio de configuraciones. Si $R$ representa
        todos los grados de libertad de un sistema con Hamiltoniano $H(R)$,

        $$
        Z = \int e^{-\beta H(R)}\,dR,
        \qquad
        \langle A\rangle =
        \frac{1}{Z}\int e^{-\beta H(R)} A(R)\,dR.
        $$

        El problema numérico es que $R$ puede tener dimensión muy alta. Un
        sistema con $N$ partículas ya tiene posiciones y momentos para cada
        partícula. En modelos de red, el número de variables también crece con
        el número de sitios. Por eso el punto central no es el azar por sí
        mismo, sino cómo aproximar integrales de alta dimensión mediante el
        promedio de valores muestreados.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Una regla de cuadratura en una dimensión aproxima una integral de la
        forma

        $$
        \int_a^b f(x)\,dx
        $$

        evaluando $f$ en puntos del intervalo. Algunos métodos, como punto
        medio, trapecio o Simpson, usan nodos igualmente espaciados. Otros,
        como la cuadratura de Gauss, eligen nodos y pesos especiales.

        Para comparar con una malla regular, supongamos que la separación entre
        nodos es $h$. Si la regla tiene orden $k$, el error típico es
        proporcional a $h^k$ para funciones suficientemente suaves.

        En $d$ dimensiones, si se usan $n$ nodos por dirección, una malla
        tensorial requiere

        $$
        N = n^d.
        $$

        Entonces $n=N^{1/d}$ y el error se comporta como

        $$
        h^k \sim n^{-k} = N^{-k/d}.
        $$

        Esta degradación con $d$ es la razón por la que Monte Carlo puede ser
        más conveniente en integrales de alta dimensión, aunque sea ineficiente
        en muchos problemas de una sola variable.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Sea $\Omega$ una región de volumen $V_\Omega$ y sea $X$ una variable
        aleatoria uniforme en $\Omega$. La notación $\mathbb{E}$ representa el
        valor esperado respecto a esa distribución uniforme. Entonces

        $$
        I = \int_\Omega f(x)\,dx
        = \textcolor{teal}{V_\Omega}\,
        \mathbb{E}\!\left[\textcolor{purple}{f(X)}\right].
        $$

        Con muestras independientes $X_1,\ldots,X_N$, el estimador de Monte
        Carlo es

        $$
        \hat I_N =
        \textcolor{teal}{V_\Omega}\,
        \frac{1}{N}\sum_{i=1}^N \textcolor{purple}{f(X_i)}.
        $$

        La estructura siempre es la misma: muestrear el dominio, evaluar el
        integrando, promediar y multiplicar por el volumen.

        ```python
        volume = ...
        values = f(samples)
        estimate = volume * values.mean()
        ```

        `volume` representa $V_\Omega$, `values` contiene los valores
        $f(X_i)$, y `values.mean()` implementa
        $\frac{1}{N}\sum_i f(X_i)$.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Python/NumPy y Fortran implementan el mismo estimador con estilos
        distintos. En Python trabajaremos con arreglos completos:

        ```python
        u = rng.uniform(0.0, 1.0, size=n)
        x = a + (b - a) * u
        values = f(x)
        estimate = (b - a) * values.mean()
        ```

        En Fortran aparecerá el mismo cálculo como un bucle explícito:

        ```fortran
        acc = 0.0
        do i = 1, n
            call random_number(u)
            x = a + (b - a) * u
            acc = acc + f(x)
        end do
        integral = (b - a) * acc / real(n)
        ```

        La variable `acc` acumula $\sum_i f(X_i)$; dividir por `real(n)` forma
        el promedio muestral. En NumPy, `uniform(a, b)` genera valores en
        $[a,b)$. Para una variable continua, excluir el extremo derecho no
        cambia el valor de la integral.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Las simulaciones de Monte Carlo usan casi siempre números
        pseudoaleatorios: secuencias deterministas con propiedades estadísticas
        suficientemente parecidas a una muestra aleatoria. La semilla fija el
        estado inicial del generador.

        Usar una semilla no vuelve aleatorio el cálculo. Lo vuelve reproducible.
        Eso es necesario para depurar, comparar implementaciones y discutir
        resultados en clase.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Primero recordemos el caso general. Si un estimador tiene la forma

        $$
        \hat\theta_N = \textcolor{teal}{c}\,\bar Y,
        \qquad
        \bar Y = \frac{1}{N}\sum_{i=1}^N \textcolor{purple}{Y_i},
        $$

        donde $Y_i$ son muestras independientes de una variable aleatoria $Y$ y
        $c$ es una constante, entonces

        $$
        \operatorname{Var}(\hat\theta_N)
        =
        \textcolor{teal}{c^2}\,
        \frac{\operatorname{Var}(\textcolor{purple}{Y})}{N},
        \qquad
        \sigma_{\hat\theta}
        =
        \frac{|\textcolor{teal}{c}|}{\sqrt{N}}
        \sqrt{\operatorname{Var}(\textcolor{purple}{Y})}.
        $$

        Para la integral de Monte Carlo, la sustitución es

        $$
        \textcolor{teal}{c=V_\Omega},
        \qquad
        \textcolor{purple}{Y_i=f(X_i)}.
        $$

        Con esta sustitución,

        $$
        \hat I_N = \textcolor{teal}{V_\Omega}\,\bar f,
        \qquad
        \bar f =
        \frac{1}{N}\sum_{i=1}^N \textcolor{purple}{f(X_i)}.
        $$

        La desviación estándar del estimador es

        $$
        \sigma_{\hat I}
        =
        \frac{\textcolor{teal}{V_\Omega}}{\sqrt{N}}
        \sqrt{\operatorname{Var}\!\left[\textcolor{purple}{f(X)}\right]}.
        $$

        Como no conocemos $\operatorname{Var}[f(X)]$ de antemano, la estimamos
        con la varianza muestral:

        $$
        s_f^2 =
        \frac{1}{N-1}\sum_{i=1}^N
        \left(\textcolor{purple}{f(X_i)}-\bar f\right)^2,
        \qquad
        \widehat{\sigma}_{\hat I}
        =
        \frac{\textcolor{teal}{V_\Omega} s_f}{\sqrt{N}}.
        $$

        El factor $N-1$ evita subestimar la varianza a partir de la misma
        muestra usada para calcular la media.

        Esta cantidad es el error estándar estimado de $\hat I_N$.

        En los ejemplos se implementa como:

        ```python
        standard_error = volume * values.std(ddof=1) / np.sqrt(n)
        ```

        `ddof=1` indica que NumPy debe usar $N-1$ en el denominador de la
        varianza muestral.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Ejemplo 1: $\pi$ como integral de una indicadora

        El área del círculo unitario puede escribirse como una integral sobre
        el cuadrado $[-1,1]\times[-1,1]$:

        $$
        \pi =
        \int_{-1}^{1}\int_{-1}^{1}
        \mathbf{1}_{x^2+y^2\leq 1}\,dx\,dy.
        $$

        Aquí $V_\Omega=4$ y el integrando solo toma dos valores: $1$ si el
        punto cae dentro del círculo y $0$ si cae fuera. Por tanto,

        $$
        \hat\pi_N =
        \textcolor{teal}{4}\,\frac{1}{N}
        \sum_{i=1}^N
        \textcolor{purple}{\mathbf{1}_{x_i^2+y_i^2\leq 1}}.
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Para esta integral, `pi_volume = 4.0` porque el dominio es el cuadrado
        $[-1,1]\times[-1,1]$. La muestra se guarda en `pi_points`: cada fila es
        un punto $(x_i,y_i)$.

        La función que se promedia es la indicadora. En el código aparece como
        `pi_indicator`: vale `1.0` cuando el punto satisface
        $x_i^2+y_i^2\leq 1$ y `0.0` en caso contrario.

        Las líneas esenciales son:

        ```python
        pi_volume = 4.0
        pi_indicator = (np.sum(pi_points**2, axis=1) <= 1.0).astype(float)
        pi_python = pi_volume * pi_indicator.mean()
        pi_standard_error = (
            pi_volume * pi_indicator.std(ddof=1) / np.sqrt(pi_n_value)
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    pi_n = mo.ui.slider(start=100, stop=50000, step=100, value=5000, label="puntos")
    pi_seed = mo.ui.number(start=0, stop=1_000_000, step=1, value=42, label="semilla")
    return pi_n, pi_seed


@app.cell
def _(pi_n, pi_seed):
    pi_n_value = int(pi_n.value)
    pi_seed_value = int(pi_seed.value)
    return pi_n_value, pi_seed_value


@app.cell
def _(np, pi_n_value, pi_seed_value):
    pi_rng = np.random.default_rng(pi_seed_value)
    pi_points = pi_rng.uniform(-1.0, 1.0, size=(pi_n_value, 2))
    pi_volume = 4.0
    pi_indicator = (np.sum(pi_points**2, axis=1) <= 1.0).astype(float)
    pi_python = pi_volume * pi_indicator.mean()
    pi_count = int(pi_indicator.sum())
    pi_standard_error = pi_volume * pi_indicator.std(ddof=1) / np.sqrt(pi_n_value)
    return pi_count, pi_indicator, pi_points, pi_python, pi_standard_error


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pi_count,
    pi_indicator,
    pi_n,
    pi_n_value,
    pi_points,
    pi_python,
    pi_seed,
    pi_standard_error,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _inside = pi_indicator.astype(bool)
    _ax.scatter(
        pi_points[~_inside, 0],
        pi_points[~_inside, 1],
        alpha=0.25,
        s=10,
        label="fuera",
    )
    _ax.scatter(
        pi_points[_inside, 0],
        pi_points[_inside, 1],
        alpha=0.45,
        s=10,
        label="dentro",
    )
    _ax.set_aspect("equal")
    _ax.set_xlim(-1, 1)
    _ax.set_ylim(-1, 1)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(rf"$N={pi_n_value}$, $\hat\pi={pi_python:.4f}$, $k={pi_count}$")
    _ax.legend()
    _fig.tight_layout()

    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | estimación Python/NumPy | ${pi_python:.6f}$ |
    | error absoluto | ${abs(pi_python - np.pi):.6f}$ |
    | error estándar estimado de $\hat\pi_N$ | ${pi_standard_error:.6f}$ |
    """)

    _controls = mo.vstack([pi_n, pi_seed], gap=1)
    mo.hstack([_controls, mo.vstack([_fig, _results], gap=1)], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        La subrutina Fortran implementa el mismo estimador con un contador
        explícito. `k` acumula
        $\sum_i \mathbf{1}_{x_i^2+y_i^2\leq 1}$, `real(k) / real(n)` es el
        promedio muestral, y el factor `4.0` multiplica por el área del
        cuadrado.

        ```fortran
        subroutine compute_pi(n, pi_estimate)
            implicit none
            integer, intent(in) :: n
            real, intent(out) :: pi_estimate
            integer :: i, k
            real :: x, y

            k = 0
            do i = 1, n
                call random_number(x)
                call random_number(y)
                x = 2.0 * x - 1.0
                y = 2.0 * y - 1.0
                if (x * x + y * y <= 1.0) then
                    k = k + 1
                end if
            end do

            pi_estimate = 4.0 * real(k) / real(n)
        end subroutine
        ```

        Una ejecución local de `fortran_f2py.py` dio:

        | implementación | $N$ | estimación | error absoluto |
        |---|---:|---:|---:|
        | Fortran/f2py local | 5000 | 3.175200 | 0.033607 |
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Ejemplo 2: integral en una dimensión

        Consideremos

        $$
        I = \int_0^2 \sqrt{4-x^2}\,dx.
        $$

        Esta integral vale $\pi$, porque corresponde al área de un cuarto de
        círculo de radio $2$. Para Monte Carlo escribimos

        $$
        I =
        \textcolor{teal}{(2-0)}\,
        \mathbb{E}\!\left[\textcolor{purple}{\sqrt{4-X^2}}\right],
        \qquad X\sim U(0,2).
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Antes de muestrear, examinemos el integrando. El área bajo esta curva
        es la integral que el estimador aproximará usando valores de la función
        en puntos aleatorios.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    _x = np.linspace(0.0, 2.0, 200)
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.fill_between(_x, np.sqrt(4.0 - _x**2), alpha=0.2)
    _ax.plot(_x, np.sqrt(4.0 - _x**2))
    _ax.set_xlabel("x")
    _ax.set_ylabel(r"$\sqrt{4-x^2}$")
    _ax.set_title(r"Integrando en $[0,2]$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Aquí `int1d_volume = 2.0` representa la longitud del intervalo. La
        muestra `int1d_x` contiene puntos uniformes en $[0,2]$, y
        `int1d_values` contiene $\sqrt{4-x_i^2}$ para cada punto.

        Las líneas centrales del estimador son:

        ```python
        int1d_volume = 2.0
        int1d_x = int1d_rng.uniform(0.0, 2.0, size=int1d_n_value)
        int1d_values = np.sqrt(4.0 - int1d_x**2)
        int1d_python = int1d_volume * int1d_values.mean()
        int1d_standard_error = (
            int1d_volume
            * int1d_values.std(ddof=1)
            / np.sqrt(int1d_n_value)
        )
        ```

        La subrutina Fortran que veremos después hace el mismo promedio con un
        acumulador `acc`. Para estimar también el error estándar en Fortran se
        puede añadir otro acumulador para $f(x_i)^2$.
        """
    )
    return


@app.cell
def _(mo):
    int1d_n = mo.ui.slider(
        start=100, stop=100000, step=100, value=10000, label="puntos"
    )
    int1d_seed = mo.ui.number(start=0, stop=1_000_000, step=1, value=7, label="semilla")
    return int1d_n, int1d_seed


@app.cell
def _(int1d_n, int1d_seed):
    int1d_n_value = int(int1d_n.value)
    int1d_seed_value = int(int1d_seed.value)
    return int1d_n_value, int1d_seed_value


@app.cell
def _(int1d_n_value, int1d_seed_value, np):
    int1d_rng = np.random.default_rng(int1d_seed_value)
    int1d_volume = 2.0
    int1d_x = int1d_rng.uniform(0.0, 2.0, size=int1d_n_value)
    int1d_values = np.sqrt(4.0 - int1d_x**2)
    int1d_python = int1d_volume * int1d_values.mean()
    int1d_standard_error = (
        int1d_volume * int1d_values.std(ddof=1) / np.sqrt(int1d_n_value)
    )
    return int1d_python, int1d_standard_error


@app.cell(hide_code=True)
def _(int1d_n, int1d_python, int1d_seed, int1d_standard_error, mo, np):
    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | estimación Python/NumPy | ${int1d_python:.6f}$ |
    | error absoluto | ${abs(int1d_python - np.pi):.6f}$ |
    | error estándar estimado de $\hat I_N$ | ${int1d_standard_error:.6f}$ |
    """)
    mo.hstack([mo.vstack([int1d_n, int1d_seed], gap=1), _results], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        En Fortran, `random_number` genera primero un número uniforme en
        $[0,1)$. La línea `x = a + (b - a) * x` lo lleva al intervalo
        $[a,b]$. Luego `acc` acumula los valores
        $\sqrt{4-x_i^2}$ y la última línea multiplica el promedio por la
        longitud del intervalo.

        ```fortran
        subroutine compute_integral_1d(n, a, b, integral)
            implicit none
            integer, intent(in) :: n
            real, intent(in) :: a, b
            real, intent(out) :: integral
            integer :: i
            real :: x, acc

            acc = 0.0
            do i = 1, n
                call random_number(x)
                x = a + (b - a) * x
                acc = acc + sqrt(4.0 - x * x)
            end do

            integral = (b - a) * acc / real(n)
        end subroutine
        ```

        Una ejecución local de `fortran_f2py.py` dio:

        | implementación | $N$ | estimación | error absoluto |
        |---|---:|---:|---:|
        | Fortran/f2py local | 10000 | 3.146916 | 0.005323 |
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Ejemplo 3: integral en dos dimensiones

        Ahora estimamos

        $$
        I =
        \int_0^1\int_0^1
        \textcolor{purple}{9x^2y^2}\,dx\,dy.
        $$

        Como

        $$
        \int_0^1 x^2\,dx = \frac{1}{3},
        $$

        el valor exacto es $9(1/3)(1/3)=1$. En el estimador de Monte Carlo,
        $\textcolor{teal}{V_\Omega=1}$ porque el dominio es el cuadrado
        unitario.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        En dos dimensiones la integral corresponde al volumen bajo una
        superficie sobre el dominio. La muestra seguirá siendo uniforme en todo
        el cuadrado, aunque el integrando sea mayor cerca de $(1,1)$.
        """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    _x = np.linspace(0.0, 1.0, 60)
    _y = np.linspace(0.0, 1.0, 60)
    _X, _Y = np.meshgrid(_x, _y)
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.pcolormesh(_X, _Y, 9.0 * _X**2 * _Y**2, shading="auto")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(r"$f(x,y)=9x^2y^2$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El estimador tiene la misma forma. Ahora cada muestra tiene dos
        coordenadas. `int2d_x` e `int2d_y` son arreglos independientes de
        puntos uniformes en $[0,1]$. `int2d_values` contiene
        $9x_i^2y_i^2$.

        Como el dominio es el cuadrado unitario, el volumen es `1.0`:

        ```python
        int2d_volume = 1.0
        int2d_x = int2d_rng.uniform(0.0, 1.0, size=int2d_n_value)
        int2d_y = int2d_rng.uniform(0.0, 1.0, size=int2d_n_value)
        int2d_values = 9.0 * int2d_x**2 * int2d_y**2
        int2d_python = int2d_volume * int2d_values.mean()
        int2d_standard_error = (
            int2d_volume
            * int2d_values.std(ddof=1)
            / np.sqrt(int2d_n_value)
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    int2d_n = mo.ui.slider(
        start=100, stop=200000, step=100, value=20000, label="puntos"
    )
    int2d_seed = mo.ui.number(
        start=0, stop=1_000_000, step=1, value=11, label="semilla"
    )
    return int2d_n, int2d_seed


@app.cell
def _(int2d_n, int2d_seed):
    int2d_n_value = int(int2d_n.value)
    int2d_seed_value = int(int2d_seed.value)
    return int2d_n_value, int2d_seed_value


@app.cell
def _(int2d_n_value, int2d_seed_value, np):
    int2d_rng = np.random.default_rng(int2d_seed_value)
    int2d_volume = 1.0
    int2d_x = int2d_rng.uniform(0.0, 1.0, size=int2d_n_value)
    int2d_y = int2d_rng.uniform(0.0, 1.0, size=int2d_n_value)
    int2d_values = 9.0 * int2d_x**2 * int2d_y**2
    int2d_python = int2d_volume * int2d_values.mean()
    int2d_standard_error = (
        int2d_volume * int2d_values.std(ddof=1) / np.sqrt(int2d_n_value)
    )
    return int2d_python, int2d_standard_error


@app.cell(hide_code=True)
def _(int2d_n, int2d_python, int2d_seed, int2d_standard_error, mo):
    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | estimación Python/NumPy | ${int2d_python:.6f}$ |
    | error absoluto | ${abs(int2d_python - 1.0):.6f}$ |
    | error estándar estimado de $\hat I_N$ | ${int2d_standard_error:.6f}$ |
    """)
    mo.hstack([mo.vstack([int2d_n, int2d_seed], gap=1), _results], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        La subrutina Fortran usa dos llamadas a `random_number`, una para cada
        coordenada. Después aplica los mapas
        `x = a + (b - a) * x` y `y = c + (d - c) * y`. El acumulador `acc`
        suma $9x_i^2y_i^2$, y la última línea multiplica el promedio por el
        área del rectángulo.

        ```fortran
        subroutine compute_integral_2d(n, a, b, c, d, integral)
            implicit none
            integer, intent(in) :: n
            real, intent(in) :: a, b, c, d
            real, intent(out) :: integral
            integer :: i
            real :: x, y, acc

            acc = 0.0
            do i = 1, n
                call random_number(x)
                call random_number(y)
                x = a + (b - a) * x
                y = c + (d - c) * y
                acc = acc + 9.0 * x * x * y * y
            end do

            integral = (b - a) * (d - c) * acc / real(n)
        end subroutine
        ```

        Una ejecución local de `fortran_f2py.py` dio:

        | implementación | $N$ | estimación | error absoluto |
        |---|---:|---:|---:|
        | Fortran/f2py local | 20000 | 1.008351 | 0.008351 |
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Experimento de convergencia

        Para verificar el comportamiento $N^{-1/2}$ es necesario repetir el
        proceso para varios tamaños de muestra. En cada caso calculamos el error
        absoluto contra el valor exacto y lo comparamos con el error estándar
        estimado desde la muestra.

        La línea proporcional a $1/\sqrt{N}$ no predice cada fluctuación
        individual. Indica la escala típica de las fluctuaciones del promedio.
        """
    )
    return


@app.cell(hide_code=True)
def _(int1d_seed_value, int2d_seed_value, np, pi_seed_value):
    convergence_sizes = np.unique(np.logspace(2, 5, 7).astype(int))

    pi_convergence_errors = []
    pi_convergence_se = []
    int1d_convergence_errors = []
    int1d_convergence_se = []
    int2d_convergence_errors = []
    int2d_convergence_se = []

    for _i, _n in enumerate(convergence_sizes):
        _pi_rng = np.random.default_rng(pi_seed_value + _i)
        _pi_points = _pi_rng.uniform(-1.0, 1.0, size=(int(_n), 2))
        _pi_indicator = (np.sum(_pi_points**2, axis=1) <= 1.0).astype(float)
        _pi_estimate = 4.0 * _pi_indicator.mean()
        pi_convergence_errors.append(abs(_pi_estimate - np.pi))
        pi_convergence_se.append(4.0 * _pi_indicator.std(ddof=1) / np.sqrt(_n))

        _int1d_rng = np.random.default_rng(int1d_seed_value + _i)
        _int1d_x = _int1d_rng.uniform(0.0, 2.0, size=int(_n))
        _int1d_values = np.sqrt(4.0 - _int1d_x**2)
        _int1d_estimate = 2.0 * _int1d_values.mean()
        int1d_convergence_errors.append(abs(_int1d_estimate - np.pi))
        int1d_convergence_se.append(2.0 * _int1d_values.std(ddof=1) / np.sqrt(_n))

        _int2d_rng = np.random.default_rng(int2d_seed_value + _i)
        _int2d_x = _int2d_rng.uniform(0.0, 1.0, size=int(_n))
        _int2d_y = _int2d_rng.uniform(0.0, 1.0, size=int(_n))
        _int2d_values = 9.0 * _int2d_x**2 * _int2d_y**2
        _int2d_estimate = _int2d_values.mean()
        int2d_convergence_errors.append(abs(_int2d_estimate - 1.0))
        int2d_convergence_se.append(_int2d_values.std(ddof=1) / np.sqrt(_n))

    pi_convergence_errors = np.array(pi_convergence_errors)
    pi_convergence_se = np.array(pi_convergence_se)
    int1d_convergence_errors = np.array(int1d_convergence_errors)
    int1d_convergence_se = np.array(int1d_convergence_se)
    int2d_convergence_errors = np.array(int2d_convergence_errors)
    int2d_convergence_se = np.array(int2d_convergence_se)
    return (
        convergence_sizes,
        int1d_convergence_errors,
        int1d_convergence_se,
        int2d_convergence_errors,
        int2d_convergence_se,
        pi_convergence_errors,
        pi_convergence_se,
    )


@app.cell(hide_code=True)
def _(
    convergence_sizes,
    int1d_convergence_errors,
    int1d_convergence_se,
    int2d_convergence_errors,
    int2d_convergence_se,
    np,
    pi_convergence_errors,
    pi_convergence_se,
    plt,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    _series = [
        (r"$\pi$", pi_convergence_errors, pi_convergence_se),
        ("1D", int1d_convergence_errors, int1d_convergence_se),
        ("2D", int2d_convergence_errors, int2d_convergence_se),
    ]

    for _ax, (_title, _errors, _se) in zip(_axes, _series):
        _ax.loglog(convergence_sizes, _errors, "o-", label="error absoluto")
        _ax.loglog(convergence_sizes, _se, "s--", label="error estándar")
        _ax.loglog(
            convergence_sizes,
            _se[0] * np.sqrt(convergence_sizes[0] / convergence_sizes),
            ":",
            label=r"$N^{-1/2}$",
        )
        _ax.set_title(_title)
        _ax.set_xlabel(r"$N$")
        _ax.grid(True, which="both")

    _axes[0].set_ylabel("error")
    _axes[-1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Interpretación

        El método de Monte Carlo tiene error típico proporcional a
        $N^{-1/2}$, independientemente de la dimensión del dominio. Esto es
        lento en una dimensión, pero evita que el costo crezca como una malla
        tensorial cuando $d$ aumenta.

        La comparación conceptual es:

        $$
        \text{cuadratura en malla tensorial: } N^{-k/d},
        \qquad
        \text{Monte Carlo: } N^{-1/2}.
        $$

        Por eso el ejemplo de $\pi$ no muestra la principal ventaja de Monte
        Carlo. Su valor aparece cuando la integral representa un promedio sobre
        muchas variables.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Hacia muestreo por importancia

        En los tres ejemplos anteriores muestreamos uniformemente. En problemas
        físicos reales, el integrando suele estar concentrado en una región
        pequeña del espacio de configuraciones. En el promedio canónico, esa
        concentración aparece por el factor de Boltzmann
        $e^{-\beta H(R)}$.

        Si se muestrea uniformemente, muchas configuraciones contribuyen casi
        nada. El siguiente paso conceptual es elegir una distribución de
        muestreo más cercana a la región importante y compensar con pesos:

        $$
        \int p_{\rm real}(R)A(R)\,dR
        =
        \int p_{\rm sample}(R)
        \frac{p_{\rm real}(R)}{p_{\rm sample}(R)}
        A(R)\,dR.
        $$

        Cuando la distribución deseada no puede muestrearse directamente, los
        métodos de cadena de Markov, como Metropolis, construyen una secuencia
        de estados cuya distribución estacionaria es la distribución objetivo.
        En ese caso aparece un nuevo problema: las muestras están
        correlacionadas y la estimación del error debe tomar esa correlación en
        cuenta.
        """
    )
    return


if __name__ == "__main__":
    app.run()
