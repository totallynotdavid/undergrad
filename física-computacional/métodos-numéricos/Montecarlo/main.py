import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import importlib.util
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from textwrap import dedent

    notebook_dir = Path(__file__).resolve().parent
    build_dir = notebook_dir / "_build"


    def md(text):
        return mo.md(dedent(text))

    return (
        build_dir,
        dedent,
        importlib,
        md,
        mo,
        np,
        os,
        plt,
        shutil,
        subprocess,
        sys,
    )


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Método de Monte Carlo

        El método de Monte Carlo aproxima cantidades numéricas usando muestras
        aleatorias. Es especialmente útil cuando el problema no tiene solución
        analítica, por ejemplo, integrales de alta dimensión.

        En esta clase cubrimos tres ejemplos progresivos:

        1. Estimar $\pi$ con puntos aleatorios en el cuadrado unitario
        2. Integrales en una dimensión: $\displaystyle\int_a^b f(x)\,dx$
        3. Integrales en dos dimensiones: $\displaystyle\iint_R f(x,y)\,dx\,dy$

        En todos los casos la idea es la misma: muestrear puntos al azar,
        evaluar la cantidad de interés en cada punto, y promediar.

        El error del método escala como $1/\sqrt{N}$, donde $N$ es el número
        de muestras. Esto se deduce del teorema central del límite y es
        independiente de la dimensión del problema, a diferencia de los
        métodos de Newton–Cotes, cuyo error escala como $1/N^{k/d}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(build_dir, fortran_sources, importlib, os, shutil, subprocess, sys):
    def extension_path(name):
        candidates = sorted(build_dir.glob(f"{name}*.so"))
        return candidates[0] if candidates else None


    def build_extension(name, source_text):
        build_dir.mkdir(parents=True, exist_ok=True)
        source_path = build_dir / f"{name}.f90"
        source_path.write_text(source_text, encoding="utf-8")

        existing = extension_path(name)
        if existing and existing.stat().st_mtime >= source_path.stat().st_mtime:
            return existing

        if shutil.which("gfortran-13") is None:
            raise RuntimeError(
                "gfortran-13 is required; run ./install.sh from the repo root."
            )

        env = os.environ.copy()
        env["FC"] = "gfortran-13"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                source_path.name,
                "-m",
                name,
            ],
            cwd=build_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)

        built = extension_path(name)
        if built is None:
            raise RuntimeError(
                "f2py finished without producing a Python extension."
            )
        return built


    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


    extensions = {
        name: load_module(name, build_extension(name, source))
        for name, source in fortran_sources.items()
    }
    return (extensions,)


@app.cell(hide_code=True)
def _():
    pi_source = """\
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
    """

    integral_1d_source = """\
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
    """

    integral_2d_source = """\
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
    """

    fortran_sources = {
        "pi_module": pi_source,
        "integral_1d_module": integral_1d_source,
        "integral_2d_module": integral_2d_source,
    }
    return fortran_sources, integral_2d_source, pi_source


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## 1. Estimar $\pi$

        El área del círculo unitario es $\pi$, y el área del cuadrado
        $[-1, 1] \times [-1, 1]$ es $4$. Si elegimos puntos al azar uniformes
        dentro del cuadrado, la fracción que cae dentro del círculo aproxima
        el cociente de las áreas:

        $$
        \hat\pi = 4\,\frac{k}{N}
        $$

        donde $k$ es el número de puntos dentro del círculo y $N$ el total.
        """
    )
    return


@app.cell
def _(mo):
    pi_n = mo.ui.slider(
        start=100, stop=50000, step=100, value=5000, label="puntos"
    )
    pi_seed = mo.ui.number(
        start=0, stop=1_000_000, step=1, value=42, label="semilla"
    )
    return pi_n, pi_seed


@app.cell
def _(pi_n, pi_seed):
    pi_n_value = int(pi_n.value)
    pi_seed_value = int(pi_seed.value)
    return pi_n_value, pi_seed_value


@app.cell
def _(extensions, pi_n_value):
    pi_fortran = extensions["pi_module"].compute_pi(pi_n_value)
    return (pi_fortran,)


@app.cell
def _(np, pi_n_value, pi_seed_value):
    rng = np.random.default_rng(pi_seed_value)
    pi_points = rng.uniform(-1.0, 1.0, size=(pi_n_value, 2))
    pi_inside = np.sum(pi_points**2, axis=1) <= 1.0
    pi_python = 4.0 * pi_inside.mean()
    pi_count = int(pi_inside.sum())
    return pi_count, pi_inside, pi_points, pi_python


@app.cell(hide_code=True)
def _(
    mo,
    np,
    pi_count,
    pi_fortran,
    pi_inside,
    pi_n,
    pi_n_value,
    pi_points,
    pi_python,
    pi_seed,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(
        pi_points[~pi_inside, 0],
        pi_points[~pi_inside, 1],
        alpha=0.25,
        s=10,
        label="fuera",
    )
    _ax.scatter(
        pi_points[pi_inside, 0],
        pi_points[pi_inside, 1],
        alpha=0.45,
        s=10,
        label="dentro",
    )
    _ax.set_aspect("equal")
    _ax.set_xlim(-1, 1)
    _ax.set_ylim(-1, 1)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_title(
        rf"$N = {pi_n_value}$, $\hat\pi = {pi_python:.4f}$, $k = {pi_count}$"
    )
    _ax.legend()
    _fig.tight_layout()

    _results = mo.md(rf"""
    **Resultados:**

    | | estimación | error absoluto |
    |---|---|---|
    | Python/numpy | $\hat\pi = {pi_python:.6f}$ | ${abs(pi_python - np.pi):.6f}$ |
    | Fortran/f2py | $\hat\pi = {pi_fortran:.6f}$ | ${abs(pi_fortran - np.pi):.6f}$ |
    """)

    _controls = mo.vstack([pi_n, pi_seed], gap=1)
    _right = mo.vstack([_fig, _results], gap=1, align="center")
    mo.hstack([_controls, _right], gap=2, justify="start", align="center")
    return


@app.cell
def _(extensions, np, pi_n_value, pi_seed_value):
    _n_lo = 100
    _n_hi = max(_n_lo * 2, int(pi_n_value))
    pi_sample_sizes = np.unique(
        np.logspace(np.log10(_n_lo), np.log10(_n_hi), 6).astype(int)
    )
    pi_python_errors = []
    pi_fortran_estimates = []
    for _i, _n in enumerate(pi_sample_sizes):
        _rng = np.random.default_rng(pi_seed_value + _i)
        _points = _rng.uniform(-1.0, 1.0, size=(int(_n), 2))
        _inside = np.sum(_points**2, axis=1) <= 1.0
        pi_python_errors.append(abs(4.0 * _inside.mean() - np.pi))
        pi_fortran_estimates.append(extensions["pi_module"].compute_pi(int(_n)))
    pi_python_errors = np.array(pi_python_errors)
    pi_fortran_estimates = np.array(pi_fortran_estimates)
    pi_fortran_errors = np.abs(pi_fortran_estimates - np.pi)
    pi_sigma = np.sqrt(np.pi * (4.0 - np.pi) / pi_sample_sizes)
    return pi_fortran_errors, pi_python_errors, pi_sample_sizes, pi_sigma


@app.cell
def _(np, pi_fortran_errors, pi_python_errors, pi_sample_sizes, pi_sigma, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(pi_sample_sizes, pi_python_errors, "o-", label="Python")
    _ax.loglog(pi_sample_sizes, pi_fortran_errors, "s-", label="Fortran")
    _ax.loglog(pi_sample_sizes, pi_sigma, "--", label=r"$\sigma_{\hat\pi}$")
    _ax.loglog(
        pi_sample_sizes,
        1 / np.sqrt(pi_sample_sizes),
        ":",
        alpha=0.6,
        label=r"$1/\sqrt{N}$",
    )
    _ax.set_xlabel(r"$N$")
    _ax.set_ylabel(r"$|\pi - \hat\pi|$")
    _ax.set_title("Convergencia ($\\pi$)")
    _ax.legend()
    _ax.grid(True, which="both")
    _fig.tight_layout()
    pi_convergence = _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementación en Fortran

    El mismo algoritmo, compilado con `numpy.f2py` (usando `gfortran-13`):
    """)
    return


@app.cell(hide_code=True)
def _(dedent, md, pi_source):
    md(
        rf"""
        {dedent(pi_source)}
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## 2. Integral en una dimensión

        Para estimar $\displaystyle I = \int_a^b f(x)\,dx$ muestreamos
        $x_1, \ldots, x_N$ uniformes en $[a, b]$ y promediamos:

        $$
        I \;\approx\; (b - a)\,\frac{1}{N}\sum_{i=1}^{N} f(x_i)
        $$

        Probamos con $f(x) = \sqrt{4 - x^2}$ en $[0, 2]$, cuya integral
        vale $\pi$.
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
    _ax.set_ylabel(r"$\sqrt{4 - x^2}$")
    _ax.set_title(r"Función a integrar en $[0, 2]$")
    _fig.tight_layout()
    int1d_plot = _fig
    return


@app.cell
def _(mo):
    int1d_n = mo.ui.slider(
        start=100, stop=100000, step=100, value=10000, label="puntos"
    )
    int1d_seed = mo.ui.number(
        start=0, stop=1_000_000, step=1, value=7, label="semilla"
    )
    return int1d_n, int1d_seed


@app.cell
def _(int1d_n, int1d_seed):
    int1d_n_value = int(int1d_n.value)
    int1d_seed_value = int(int1d_seed.value)
    return int1d_n_value, int1d_seed_value


@app.cell
def _(extensions, int1d_n_value):
    int1d_fortran = extensions["integral_1d_module"].compute_integral_1d(
        int1d_n_value, 0.0, 2.0
    )
    return (int1d_fortran,)


@app.cell
def _(int1d_n_value, int1d_seed_value, np):
    _rng = np.random.default_rng(int1d_seed_value)
    _x = _rng.uniform(0.0, 2.0, size=int1d_n_value)
    int1d_python = 2.0 * np.sqrt(4.0 - _x**2).mean()
    return (int1d_python,)


@app.cell(hide_code=True)
def _(int1d_fortran, int1d_n, int1d_python, int1d_seed, mo, np):
    _results = mo.md(rf"""
    **Resultados para** $\int_0^2 \sqrt{{4 - x^2}}\,dx = \pi$:

    | | estimación | error absoluto |
    |---|---|---|
    | Python/numpy | ${int1d_python:.6f}$ | ${abs(int1d_python - np.pi):.6f}$ |
    | Fortran/f2py | ${int1d_fortran:.6f}$ | ${abs(int1d_fortran - np.pi):.6f}$ |
    """)

    _controls = mo.vstack([int1d_n, int1d_seed], gap=1)
    mo.hstack([_controls, _results], gap=2, justify="start", align="center")
    return


@app.cell
def _(extensions, int1d_n_value, int1d_seed_value, np):
    _n_lo = 100
    _n_hi = max(_n_lo * 2, int(int1d_n_value))
    int1d_sample_sizes = np.unique(
        np.logspace(np.log10(_n_lo), np.log10(_n_hi), 5).astype(int)
    )
    int1d_python_errors = []
    int1d_fortran_estimates = []
    for _i, _n in enumerate(int1d_sample_sizes):
        _rng = np.random.default_rng(int1d_seed_value + _i)
        _x = _rng.uniform(0.0, 2.0, size=int(_n))
        int1d_python_errors.append(abs(2.0 * np.sqrt(4.0 - _x**2).mean() - np.pi))
        int1d_fortran_estimates.append(
            extensions["integral_1d_module"].compute_integral_1d(int(_n), 0.0, 2.0)
        )
    int1d_python_errors = np.array(int1d_python_errors)
    int1d_fortran_estimates = np.array(int1d_fortran_estimates)
    int1d_fortran_errors = np.abs(int1d_fortran_estimates - np.pi)
    int1d_sigma = np.sqrt(np.pi * (4.0 - np.pi) / int1d_sample_sizes)
    return (
        int1d_fortran_errors,
        int1d_python_errors,
        int1d_sample_sizes,
        int1d_sigma,
    )


@app.cell
def _(
    int1d_fortran_errors,
    int1d_python_errors,
    int1d_sample_sizes,
    int1d_sigma,
    np,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(int1d_sample_sizes, int1d_python_errors, "o-", label="Python")
    _ax.loglog(int1d_sample_sizes, int1d_fortran_errors, "s-", label="Fortran")
    _ax.loglog(int1d_sample_sizes, int1d_sigma, "--", label=r"$\sigma_I$")
    _ax.loglog(
        int1d_sample_sizes,
        1 / np.sqrt(int1d_sample_sizes),
        ":",
        alpha=0.6,
        label=r"$1/\sqrt{N}$",
    )
    _ax.set_xlabel(r"$N$")
    _ax.set_ylabel(r"$|I - \hat I|$")
    _ax.set_title("Convergencia (integral 1D)")
    _ax.legend()
    _ax.grid(True, which="both")
    _fig.tight_layout()
    int1d_convergence = _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementación en Fortran
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    {dedent(integral_1d_source)}
    """)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## 3. Integral en dos dimensiones

        Generalizamos a un rectángulo $R = [a, b] \times [c, d]$:

        $$
        I \;=\; \iint_R f(x, y)\,dx\,dy
        \;\approx\; (b-a)(d-c)\,\frac{1}{N}\sum_{i=1}^{N} f(x_i, y_i)
        $$

        Probamos con $f(x, y) = 9 x^2 y^2$ en $[0, 1] \times [0, 1]$,
        donde la integral vale $1$.
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
    _ax.set_title(r"$f(x, y) = 9 x^2 y^2$")
    _fig.tight_layout()
    int2d_plot = _fig
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
def _(extensions, int2d_n_value):
    int2d_fortran = extensions["integral_2d_module"].compute_integral_2d(
        int2d_n_value, 0.0, 1.0, 0.0, 1.0
    )
    return (int2d_fortran,)


@app.cell
def _(int2d_n_value, int2d_seed_value, np):
    _rng = np.random.default_rng(int2d_seed_value)
    _x = _rng.uniform(0.0, 1.0, size=int2d_n_value)
    _y = _rng.uniform(0.0, 1.0, size=int2d_n_value)
    int2d_python = (9.0 * _x**2 * _y**2).mean()
    return (int2d_python,)


@app.cell(hide_code=True)
def _(int2d_fortran, int2d_n, int2d_python, int2d_seed, mo):
    _results = mo.md(rf"""
    **Resultados para** $\int_0^1\!\!\int_0^1 9x^2y^2\,dx\,dy = 1$:

    | | estimación | error absoluto |
    |---|---|---|
    | Python/numpy | ${int2d_python:.6f}$ | ${abs(int2d_python - 1.0):.6f}$ |
    | Fortran/f2py | ${int2d_fortran:.6f}$ | ${abs(int2d_fortran - 1.0):.6f}$ |
    """)

    _controls = mo.vstack([int2d_n, int2d_seed], gap=1)
    mo.hstack([_controls, _results], gap=2, justify="start", align="center")
    return


@app.cell
def _(extensions, int2d_n_value, int2d_seed_value, np):
    _n_lo = 100
    _n_hi = max(_n_lo * 2, int(int2d_n_value))
    int2d_sample_sizes = np.unique(
        np.logspace(np.log10(_n_lo), np.log10(_n_hi), 5).astype(int)
    )
    int2d_python_errors = []
    int2d_fortran_estimates = []
    for _i, _n in enumerate(int2d_sample_sizes):
        _rng = np.random.default_rng(int2d_seed_value + _i)
        _x = _rng.uniform(0.0, 1.0, size=int(_n))
        _y = _rng.uniform(0.0, 1.0, size=int(_n))
        int2d_python_errors.append(abs((9.0 * _x**2 * _y**2).mean() - 1.0))
        int2d_fortran_estimates.append(
            extensions["integral_2d_module"].compute_integral_2d(
                int(_n), 0.0, 1.0, 0.0, 1.0
            )
        )
    int2d_python_errors = np.array(int2d_python_errors)
    int2d_fortran_estimates = np.array(int2d_fortran_estimates)
    int2d_fortran_errors = np.abs(int2d_fortran_estimates - 1.0)
    int2d_sigma = np.sqrt(81.0 / 45.0 / int2d_sample_sizes)
    return (
        int2d_fortran_errors,
        int2d_python_errors,
        int2d_sample_sizes,
        int2d_sigma,
    )


@app.cell
def _(
    int2d_fortran_errors,
    int2d_python_errors,
    int2d_sample_sizes,
    int2d_sigma,
    np,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(int2d_sample_sizes, int2d_python_errors, "o-", label="Python")
    _ax.loglog(int2d_sample_sizes, int2d_fortran_errors, "s-", label="Fortran")
    _ax.loglog(int2d_sample_sizes, int2d_sigma, "--", label=r"$\sigma_I$")
    _ax.loglog(
        int2d_sample_sizes,
        1 / np.sqrt(int2d_sample_sizes),
        ":",
        alpha=0.6,
        label=r"$1/\sqrt{N}$",
    )
    _ax.set_xlabel(r"$N$")
    _ax.set_ylabel(r"$|I - \hat I|$")
    _ax.set_title("Convergencia (integral 2D)")
    _ax.legend()
    _ax.grid(True, which="both")
    _fig.tight_layout()
    int2d_convergence = _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementación en Fortran
    """)
    return


@app.cell(hide_code=True)
def _(dedent, integral_2d_source, md):
    md(
        rf"""
        {dedent(integral_2d_source)}
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## 4. Convergencia

        En los tres casos el error empírico cae como $1/\sqrt{N}$, como
        predice el teorema central del límite:

        $$
        \sigma_{\hat\theta} \;\sim\; \frac{1}{\sqrt{N}}
        $$

        Esta es la ventaja central del método de Monte Carlo: la precisión
        es independiente de la dimensión del problema. Métodos de
        Newton–Cotes o Gauss–Legendre de orden $k$ en $d$ dimensiones
        escalan como $1/N^{k/d}$, que se degrada rápidamente al subir la
        dimensión.
        """
    )
    return


if __name__ == "__main__":
    app.run()
