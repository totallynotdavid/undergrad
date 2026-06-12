import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import os
    import subprocess
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    FORTRAN_FILE = NOTEBOOK_DIR / "main.f95"
    EXECUTABLE = NOTEBOOK_DIR / "a.out"
    GFORTRAN_BIN = "gfortran-13"
    TEMPERATURES = [2000, 6000, 10000, 14000]
    CSV_BASENAME = "curvaDePlanck"
    return (
        CSV_BASENAME,
        EXECUTABLE,
        FORTRAN_FILE,
        GFORTRAN_BIN,
        NOTEBOOK_DIR,
        TEMPERATURES,
        mo,
        os,
        pd,
        plt,
        subprocess,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curva de Planck

    Este proyecto grafica la distribución de Planck para diferentes temperaturas.
    """)
    return


@app.cell
def _(EXECUTABLE, FORTRAN_FILE, GFORTRAN_BIN, NOTEBOOK_DIR, os, subprocess):
    try:
        print(f"Compilando el código Fortran con {GFORTRAN_BIN}...")
        subprocess.run(
            [GFORTRAN_BIN, str(FORTRAN_FILE), "-o", str(EXECUTABLE)],
            check=True,
            cwd=NOTEBOOK_DIR,
        )
        print("Compilación exitosa.")
    except subprocess.CalledProcessError as e:
        print("Error durante la compilación:")
        print(e)
        raise

    if os.name != "nt":
        os.chmod(EXECUTABLE, 0o755)

    if EXECUTABLE.exists():
        try:
            print("Ejecutando el programa Fortran...")
            subprocess.run([str(EXECUTABLE)], check=True, cwd=NOTEBOOK_DIR)
            print("Ejecutación exitosa. Archivos CSV generados.")
        except subprocess.CalledProcessError as e:
            print("Error durante la ejecución:")
            print(e)
            raise
    else:
        print(f"El ejecutable {EXECUTABLE} no se encontró.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cargamos ahora los datos:
    """)
    return


@app.cell
def _(CSV_BASENAME, NOTEBOOK_DIR, TEMPERATURES, pd):
    datos = {}

    # Cargar datos desde los archivos CSV
    for _temp in TEMPERATURES:
        archivo = NOTEBOOK_DIR / f"{CSV_BASENAME}_{_temp}.csv"
        if archivo.exists():
            try:
                datos[_temp] = pd.read_csv(archivo)
            except pd.errors.EmptyDataError:
                print(f"Archivo {archivo} está vacío.")
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")
        else:
            print(f"Archivo {archivo} no encontrado.")
    return (datos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finalmente, graficamos las curvas de Planck.
    """)
    return


@app.cell
def _(TEMPERATURES, datos, plt):
    plt.figure(figsize=(10, 6))
    for _temp in TEMPERATURES:
        df = datos.get(_temp)
        if df is not None:
            plt.plot(df["x"], df["y"], linewidth=2, label=f"{_temp} K")
    plt.title("Distribución de Planck", fontsize=15, pad=20)
    plt.xlabel("Longitud de onda λ (nm)", fontsize=12)
    plt.ylabel(
        "Radiación $B_\\lambda$ $(\\frac{W}{m^2 \\cdot sr \\cdot nm})$", fontsize=12
    )
    plt.xlim(0, 2000)
    plt.ylim(0, 2500000000000000.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
