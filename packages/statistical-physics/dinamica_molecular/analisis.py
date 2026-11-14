"""Observables de una trayectoria: g(r), desplazamiento cuadrático medio y D."""

from __future__ import annotations

import numba
import numpy as np


@numba.njit
def _histograma_pares(cuadros, lado, r_max, n_bins):
    n_cuadros, n = cuadros.shape[0], cuadros.shape[1]
    cuentas = np.zeros(n_bins)
    dr = r_max / n_bins
    for c in range(n_cuadros):
        for i in range(n - 1):
            for j in range(i + 1, n):
                r2 = 0.0
                for k in range(3):
                    d = cuadros[c, i, k] - cuadros[c, j, k]
                    d -= lado * np.round(d / lado)
                    r2 += d * d
                r = np.sqrt(r2)
                if r < r_max:
                    cuentas[int(r / dr)] += 2.0
    return cuentas


def g_de_r(
    cuadros: np.ndarray, lado: float, n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Función de distribución radial promediada sobre `cuadros`.

    Se calcula hasta lado / 2, donde la mínima imagen deja de ser única.
    Devuelve los centros de los bins y g(r).
    """
    n = cuadros.shape[1]
    r_max = lado / 2
    cuentas = _histograma_pares(
        np.ascontiguousarray(cuadros, dtype=np.float64), float(lado), r_max, n_bins
    )
    bordes = np.linspace(0.0, r_max, n_bins + 1)
    volumen_capa = 4.0 / 3.0 * np.pi * (bordes[1:] ** 3 - bordes[:-1] ** 3)
    ideal = n * (n / lado**3) * volumen_capa * cuadros.shape[0]
    return 0.5 * (bordes[1:] + bordes[:-1]), cuentas / ideal


def desplazamiento_cuadratico_medio(
    cuadros: np.ndarray, max_retardo: int | None = None
) -> np.ndarray:
    """MSD(k) promediado sobre partículas y sobre todos los orígenes de tiempo.

    `cuadros` debe tener las posiciones sin envolver en la caja.
    """
    n_cuadros = cuadros.shape[0]
    max_retardo = n_cuadros // 2 if max_retardo is None else max_retardo
    msd = np.empty(max_retardo + 1)
    msd[0] = 0.0
    for k in range(1, max_retardo + 1):
        desplazamiento = cuadros[k:] - cuadros[:-k]
        msd[k] = np.mean(np.sum(desplazamiento**2, axis=2))
    return msd


def coeficiente_difusion(
    tiempo: np.ndarray, msd: np.ndarray, desde: float, hasta: float
) -> tuple[float, float]:
    """D a partir de la relación de Einstein MSD = 6 D t.

    Ajusta una recta a MSD(t) para `desde <= t <= hasta`, tramo en que el
    movimiento ya es difusivo, y devuelve (D, ordenada en el origen).
    """
    ventana = (tiempo >= desde) & (tiempo <= hasta)
    pendiente, ordenada = np.polyfit(tiempo[ventana], msd[ventana], 1)
    return pendiente / 6.0, ordenada
