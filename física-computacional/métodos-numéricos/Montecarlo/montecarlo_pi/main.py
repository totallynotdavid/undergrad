import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
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
    fortran_source_path = notebook_dir / "src" / "pi_montecarlo.f90"
    fortran_build_dir = notebook_dir / "_build" / "f2py"
    fortran_module_name = "pi_fortran"

    def md(text):
        return mo.md(dedent(text))

    return (
        fortran_build_dir,
        fortran_module_name,
        fortran_source_path,
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
        # Estimación de $\pi$ mediante Monte Carlo y Fortran/f2py

        El área del círculo unitario es $\pi$. El área del cuadrado
        $[-1, 1] \times [-1, 1]$ es $4$. Si elegimos puntos aleatorios uniformes
        dentro del cuadrado, la fracción que cae dentro del círculo aproxima
        el cociente entre ambas áreas.

        $$
        \hat\pi = 4 \frac{k}{N}
        $$

        Aquí $k$ es el número de puntos dentro del círculo y $N$ el número
        total de puntos.
        """
    )
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(
        start=100,
        stop=20000,
        step=100,
        value=2000,
        label="puntos",
    )
    random_seed = mo.ui.number(
        start=0,
        stop=1_000_000,
        step=1,
        value=42,
        label="semilla",
    )
    mo.hstack([n_points, random_seed], gap=2)
    return n_points, random_seed


@app.cell
def _(n_points, random_seed):
    n_points_value = int(n_points.value)
    random_seed_value = int(random_seed.value)
    return n_points_value, random_seed_value


@app.cell(hide_code=True)
def _(md, n_points_value):
    reference_inside = 0.7854 * n_points_value
    md(
        rf"""
        ## Cálculo esperado

        Como el círculo ocupa aproximadamente el $78.54\%$ del cuadrado,
        para $N = {n_points_value}$ esperamos cerca de
        ${reference_inside:.0f}$ puntos dentro del círculo.
        """
    )
    return


@app.cell
def _(np):
    def sample_unit_square(n, seed):
        rng = np.random.default_rng(seed)
        points = rng.uniform(-1.0, 1.0, size=(n, 2))
        squared_radius = np.sum(points**2, axis=1)
        inside_circle = squared_radius <= 1.0
        return points, inside_circle

    def estimate_pi_from_mask(inside_circle):
        return 4.0 * inside_circle.mean()

    def estimate_pi_numpy(n, seed):
        _points, inside_circle = sample_unit_square(n, seed)
        return estimate_pi_from_mask(inside_circle)

    return estimate_pi_from_mask, estimate_pi_numpy, sample_unit_square


@app.cell
def _(
    estimate_pi_from_mask,
    n_points_value,
    random_seed_value,
    sample_unit_square,
):
    points, inside_circle = sample_unit_square(n_points_value, random_seed_value)
    inside_count = int(inside_circle.sum())
    pi_estimate = estimate_pi_from_mask(inside_circle)
    return inside_circle, inside_count, pi_estimate, points


@app.cell(hide_code=True)
def _(inside_count, md, n_points_value, np, pi_estimate):
    absolute_error = abs(pi_estimate - np.pi)
    md(
        rf"""
        ## Resultado

        Puntos dentro del círculo:

        $$
        k = {inside_count}, \qquad N = {n_points_value}
        $$

        Estimación:

        $$
        \hat{{\pi}} = {pi_estimate:.6f}
        $$

        Error absoluto frente a `numpy.pi`:

        $$
        |\pi - \hat{{\pi}}| = {absolute_error:.6f}
        $$
        """
    )
    return


@app.cell
def _(inside_circle, n_points_value, pi_estimate, plt, points):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(
        points[~inside_circle, 0],
        points[~inside_circle, 1],
        alpha=0.25,
        s=10,
        label="fuera",
    )
    _ax.scatter(
        points[inside_circle, 0],
        points[inside_circle, 1],
        alpha=0.45,
        s=10,
        label="dentro",
    )
    _ax.set_aspect("equal")
    _ax.set_xlim(-1, 1)
    _ax.set_ylim(-1, 1)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.legend()
    _ax.set_title(rf"$N = {n_points_value}, \hat{{\pi}} = {pi_estimate:.4f}$")
    _fig.tight_layout()
    return _fig


@app.cell(hide_code=True)
def _(fortran_source_path, md):
    fortran_source = fortran_source_path.read_text(encoding="utf-8")
    md(
        f"""
        ## Implementación en Fortran

        La subrutina original expresa el mismo algoritmo con un bucle explícito.
        En este notebook se compila con `numpy.f2py`, usando `gfortran-13`,
        para llamarla desde Python.

        ```fortran
        {fortran_source}
        ```
        """
    )
    return


@app.cell
def _(fortran_build_dir, fortran_module_name, fortran_source_path, os, shutil, subprocess, sys):
    def extension_candidates(build_dir, module_name):
        return sorted(build_dir.glob(f"{module_name}*.so"))

    def copy_source_if_needed(source_path, build_dir):
        build_dir.mkdir(parents=True, exist_ok=True)
        source_copy = build_dir / source_path.name
        source_text = source_path.read_text(encoding="utf-8")
        if source_copy.exists() and source_copy.read_text(encoding="utf-8") == source_text:
            return source_copy
        source_copy.write_text(source_text, encoding="utf-8")
        return source_copy

    def build_fortran_extension(source_path, build_dir, module_name):
        source_copy = copy_source_if_needed(source_path, build_dir)
        existing_extensions = extension_candidates(build_dir, module_name)
        if existing_extensions:
            newest_extension = max(existing_extensions, key=lambda path: path.stat().st_mtime)
            if newest_extension.stat().st_mtime >= source_copy.stat().st_mtime:
                return newest_extension

        if shutil.which("gfortran-13") is None:
            raise RuntimeError("gfortran-13 is required; run ./install.sh from the repo root.")

        environment = os.environ.copy()
        environment["FC"] = "gfortran-13"
        command = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            source_copy.name,
            "-m",
            module_name,
        ]
        completed = subprocess.run(
            command,
            cwd=build_dir,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)

        built_extensions = extension_candidates(build_dir, module_name)
        if not built_extensions:
            raise RuntimeError("f2py finished without producing a Python extension.")
        return max(built_extensions, key=lambda path: path.stat().st_mtime)

    fortran_extension_path = build_fortran_extension(
        fortran_source_path,
        fortran_build_dir,
        fortran_module_name,
    )

    return build_fortran_extension, fortran_extension_path


@app.cell
def _(fortran_extension_path, fortran_module_name, importlib):
    def load_fortran_module(module_name, extension_path):
        specification = importlib.util.spec_from_file_location(module_name, extension_path)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"Could not load {extension_path}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    pi_fortran_module = load_fortran_module(fortran_module_name, fortran_extension_path)
    compute_pi_fortran = pi_fortran_module.compute_pi
    return compute_pi_fortran, load_fortran_module, pi_fortran_module


@app.cell
def _(compute_pi_fortran, n_points_value):
    pi_fortran = compute_pi_fortran(n_points_value)
    return (pi_fortran,)


@app.cell(hide_code=True)
def _(md, np, pi_estimate, pi_fortran):
    md(
        rf"""
        ## Comparación Python y Fortran/f2py

        Python/numpy:

        $$
        \hat{{\pi}}_{{\mathrm{{Python}}}} = {pi_estimate:.6f}
        $$

        Fortran/f2py:

        $$
        \hat{{\pi}}_{{\mathrm{{Fortran}}}} = {pi_fortran:.6f}
        $$

        Errores absolutos:

        $$
        |\pi - \hat{{\pi}}_{{\mathrm{{Python}}}}|
        = {abs(pi_estimate - np.pi):.6f}
        $$

        $$
        |\pi - \hat{{\pi}}_{{\mathrm{{Fortran}}}}|
        = {abs(pi_fortran - np.pi):.6f}
        $$
        """
    )
    return


@app.cell
def _(compute_pi_fortran, estimate_pi_numpy, np, random_seed_value):
    sample_sizes = np.array([100, 300, 1000, 3000, 10000, 30000, 100000])
    pi_estimates = np.array(
        [
            estimate_pi_numpy(int(sample_size), random_seed_value + index)
            for index, sample_size in enumerate(sample_sizes)
        ]
    )
    pi_fortran_estimates = np.array(
        [compute_pi_fortran(int(sample_size)) for sample_size in sample_sizes]
    )
    absolute_errors_python = np.abs(pi_estimates - np.pi)
    absolute_errors_fortran = np.abs(pi_fortran_estimates - np.pi)
    expected_sigma = np.sqrt(np.pi * (4.0 - np.pi) / sample_sizes)
    return absolute_errors_fortran, absolute_errors_python, expected_sigma, sample_sizes


@app.cell
def _(absolute_errors_fortran, absolute_errors_python, expected_sigma, np, plt, sample_sizes):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(sample_sizes, absolute_errors_python, "o-", label="Python/numpy")
    _ax.loglog(sample_sizes, absolute_errors_fortran, "s-", label="Fortran/f2py")
    _ax.loglog(sample_sizes, expected_sigma, "--", label=r"$\sigma_{\hat\pi}$")
    _ax.loglog(
        sample_sizes,
        1 / np.sqrt(sample_sizes),
        ":",
        alpha=0.6,
        label=r"$1/\sqrt{N}$",
    )
    _ax.set_xlabel(r"$N$")
    _ax.set_ylabel(r"$|\pi - \hat\pi|$")
    _ax.legend()
    _ax.grid(True, which="both")
    _fig.tight_layout()
    return _fig


@app.cell(hide_code=True)
def _(md):
    md(
        r"""
        ## Convergencia

        El indicador "dentro del círculo" es una variable de Bernoulli con
        probabilidad $p = \pi/4$. Como el estimador multiplica la media por $4$,
        su desviación estándar es

        $$
        \sigma_{\hat\pi} =
        \sqrt{\frac{\pi(4 - \pi)}{N}} \approx \frac{1.64}{\sqrt{N}}.
        $$

        Una corrida individual puede quedar por encima o por debajo de esa
        escala, pero la tendencia mejora como $1/\sqrt{N}$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
