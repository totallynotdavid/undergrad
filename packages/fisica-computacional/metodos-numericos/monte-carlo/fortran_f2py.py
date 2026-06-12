import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from textwrap import dedent

    import marimo as mo
    import numpy as np

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
        notebook_dir,
        np,
        os,
        shutil,
        subprocess,
        sys,
    )


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        # Monte Carlo con Fortran y f2py

        Este notebook es el complemento local de `main.py`. El desarrollo del
        estimador, la interpretación estadística y la versión interactiva para
        GitHub Pages están en ese notebook. Aquí nos concentramos en la parte
        nativa: escribir las rutinas en Fortran, compilarlas con `f2py` y
        llamarlas desde Python.

        El navegador no puede compilar Fortran ni cargar extensiones nativas
        `.so`. Por eso este archivo debe ejecutarse localmente:

        ```bash
        ./install.sh
        uv run --package fisica-computacional marimo edit \
          packages/fisica-computacional/metodos-numericos/monte-carlo/fortran_f2py.py
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        `f2py` toma subrutinas de Fortran y genera una extensión de Python. En
        este notebook usamos una sola extensión, `montecarlo_fortran`, con todas
        las subrutinas.

        La convención principal es:

        ```fortran
        integer, intent(in) :: n
        real, intent(out) :: integral
        ```

        Los argumentos `intent(in)` se pasan desde Python. Un argumento
        `intent(out)` se devuelve como resultado de la llamada. Por ejemplo, una
        subrutina Fortran `compute_pi(n, pi_estimate)` queda disponible desde
        Python como `compute_pi(n)`.

        La extensión se compila en `_build/`. Para evitar recompilar en cada
        ejecución, el notebook solo reescribe el archivo `.f90` si el contenido
        cambió, y solo vuelve a llamar a `f2py` si la fuente es más reciente que
        el `.so`.
        """
    )
    return


@app.cell(hide_code=True)
def _(dedent):
    montecarlo_source = dedent(
        """\
        subroutine seed_random(seed)
            implicit none
            integer, intent(in) :: seed
            integer :: i, n
            integer, allocatable :: seed_values(:)

            call random_seed(size=n)
            allocate(seed_values(n))

            do i = 1, n
                seed_values(i) = modulo(seed + 37 * i, 2147483647)
                if (seed_values(i) <= 0) then
                    seed_values(i) = seed_values(i) + 2147483646
                end if
            end do

            call random_seed(put=seed_values)
            deallocate(seed_values)
        end subroutine

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
    )
    return (montecarlo_source,)


@app.cell(hide_code=True)
def _(build_dir, importlib, montecarlo_source, os, shutil, subprocess, sys):
    def extension_path(name):
        candidates = sorted(build_dir.glob(f"{name}*.so"))
        return candidates[0] if candidates else None

    def write_source_if_changed(source_path, source_text):
        if (
            source_path.exists()
            and source_path.read_text(encoding="utf-8") == source_text
        ):
            return
        source_path.write_text(source_text, encoding="utf-8")

    def build_extension(name, source_text):
        build_dir.mkdir(parents=True, exist_ok=True)
        source_path = build_dir / f"{name}.f90"
        write_source_if_changed(source_path, source_text)

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
            raise RuntimeError("f2py finished without producing a Python extension.")
        return built

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    extension_file = build_extension("montecarlo_fortran", montecarlo_source)
    montecarlo_fortran = load_module("montecarlo_fortran", extension_file)
    return extension_file, montecarlo_fortran


@app.cell(hide_code=True)
def _(extension_file, md, notebook_dir):
    _relative_extension = extension_file.relative_to(notebook_dir)
    md(
        rf"""
        La extensión nativa cargada en esta sesión es:

        `{_relative_extension}`
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        Para que los resultados locales sean reproducibles, primero fijamos la
        semilla del generador pseudoaleatorio de Fortran. `random_seed(size=n)`
        pregunta cuántos enteros necesita el compilador para representar el
        estado interno. Luego construimos un arreglo de ese tamaño y lo pasamos
        con `random_seed(put=seed_values)`.

        ```fortran
        subroutine seed_random(seed)
            implicit none
            integer, intent(in) :: seed
            integer :: i, n
            integer, allocatable :: seed_values(:)

            call random_seed(size=n)
            allocate(seed_values(n))

            do i = 1, n
                seed_values(i) = modulo(seed + 37 * i, 2147483647)
                if (seed_values(i) <= 0) then
                    seed_values(i) = seed_values(i) + 2147483646
                end if
            end do

            call random_seed(put=seed_values)
            deallocate(seed_values)
        end subroutine
        ```

        Esta subrutina no cambia el estimador de Monte Carlo. Solo fija el estado
        inicial del generador para poder repetir una ejecución local.
        """
    )
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Estimación de $\pi$

        En `main.py` se deriva la fórmula como una integral de una indicadora.
        En Fortran la implementación directa consiste en contar cuántos puntos
        caen dentro del círculo unitario.

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

        `random_number` genera números uniformes en $[0,1)$. Las asignaciones
        `x = 2.0 * x - 1.0` y `y = 2.0 * y - 1.0` llevan esos números al
        cuadrado $[-1,1]\times[-1,1]$. La variable `k` acumula el número de
        puntos aceptados, y `pi_estimate` recibe $4k/N$.
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
def _(montecarlo_fortran, pi_n_value, pi_seed_value):
    montecarlo_fortran.seed_random(pi_seed_value)
    pi_fortran = montecarlo_fortran.compute_pi(pi_n_value)
    return (pi_fortran,)


@app.cell(hide_code=True)
def _(mo, np, pi_fortran, pi_n, pi_n_value, pi_seed):
    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | muestras | ${pi_n_value}$ |
    | estimación Fortran/f2py | ${pi_fortran:.6f}$ |
    | error absoluto | ${abs(pi_fortran - np.pi):.6f}$ |
    """)

    mo.hstack([mo.vstack([pi_n, pi_seed], gap=1), _results], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Integral en una dimensión

        Consideremos la integral usada en `main.py`,

        $$
        I = \int_0^2 \sqrt{4-x^2}\,dx = \pi.
        $$

        La subrutina recibe los extremos `a` y `b`. Dentro del bucle, primero
        genera un número uniforme en $[0,1)$, luego lo transforma al intervalo
        $[a,b)$ con `x = a + (b - a) * x`. El acumulador `acc` suma las
        evaluaciones de la función.

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

        La última línea divide por `real(n)` para formar el promedio y multiplica
        por `(b - a)`, la longitud del intervalo.
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
def _(int1d_n_value, int1d_seed_value, montecarlo_fortran):
    montecarlo_fortran.seed_random(int1d_seed_value)
    int1d_fortran = montecarlo_fortran.compute_integral_1d(int1d_n_value, 0.0, 2.0)
    return (int1d_fortran,)


@app.cell(hide_code=True)
def _(int1d_fortran, int1d_n, int1d_n_value, int1d_seed, mo, np):
    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | muestras | ${int1d_n_value}$ |
    | estimación Fortran/f2py | ${int1d_fortran:.6f}$ |
    | error absoluto | ${abs(int1d_fortran - np.pi):.6f}$ |
    """)

    mo.hstack([mo.vstack([int1d_n, int1d_seed], gap=1), _results], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Integral en dos dimensiones

        Ahora usamos el ejemplo

        $$
        I = \int_0^1\int_0^1 9x^2y^2\,dx\,dy = 1.
        $$

        La subrutina recibe un rectángulo general
        $[a,b]\times[c,d]$. Por eso genera dos números uniformes, transforma cada
        coordenada a su intervalo y multiplica el promedio final por el área
        `(b - a) * (d - c)`.

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

        Aunque en este ejemplo el área vale $1$, mantener la fórmula general
        permite reutilizar la subrutina para otros rectángulos.
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
def _(int2d_n_value, int2d_seed_value, montecarlo_fortran):
    montecarlo_fortran.seed_random(int2d_seed_value)
    int2d_fortran = montecarlo_fortran.compute_integral_2d(
        int2d_n_value, 0.0, 1.0, 0.0, 1.0
    )
    return (int2d_fortran,)


@app.cell(hide_code=True)
def _(int2d_fortran, int2d_n, int2d_n_value, int2d_seed, mo):
    _results = mo.md(rf"""
    | cantidad | valor |
    |---|---:|
    | muestras | ${int2d_n_value}$ |
    | estimación Fortran/f2py | ${int2d_fortran:.6f}$ |
    | error absoluto | ${abs(int2d_fortran - 1.0):.6f}$ |
    """)

    mo.hstack([mo.vstack([int2d_n, int2d_seed], gap=1), _results], gap=2)
    return


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        El objetivo de esta versión es mostrar cómo se escribe el mismo cálculo
        cuando el promedio de Monte Carlo se expresa como un bucle explícito en
        Fortran y se expone a Python mediante `f2py`.

        Para discutir el error estándar, el comportamiento $N^{-1/2}$ y la
        comparación con cuadratura en malla tensorial, usa el notebook principal.
        """
    )
    return


if __name__ == "__main__":
    app.run()
