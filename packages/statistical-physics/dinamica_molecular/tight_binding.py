"""Potencial de enlace fuerte en segundo momento para cúmulos de potasio.

La energía de cohesión de un cúmulo de átomos aislados es

    E = sum_i [ A sum_{j != i} exp(-p (r_ij / r0 - 1))
                - sqrt( sum_{j != i} xi^2 exp(-2 q (r_ij / r0 - 1)) ) ]

El primer término es una repulsión de pares de tipo Born-Mayer. El segundo es
la energía de banda en la aproximación del segundo momento: la raíz cuadrada
hace que el enlace de cada átomo dependa de su coordinación, algo que un
potencial de pares como Lennard-Jones no puede describir.

Unidades: eV, angstrom y masa atómica. La unidad de tiempo resultante es
`TIEMPO` femtosegundos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

RYDBERG = 27.21138386 / 2.0  # eV
BOHR = 0.52917721  # angstrom
MASA_K = 39.0983  # u
# sqrt(u * angstrom^2 / eV) en femtosegundos.
TIEMPO = 10.1805


@dataclass(frozen=True)
class ParametrosTB:
    a: float  # eV, amplitud de la repulsión
    xi: float  # eV, integral de salto efectiva
    r0: float  # angstrom, distancia de referencia
    p: float
    q: float


POTASIO = ParametrosTB(
    a=1.51e-3 * RYDBERG,
    xi=19.3e-3 * RYDBERG,
    r0=8.253 * BOHR,
    p=10.58,
    q=1.34,
)


@numba.njit
def energia_y_fuerzas(pos, a, xi, r0, p, q):
    """Energía de cohesión total (eV) y fuerzas (eV/angstrom) de un cúmulo."""
    n = pos.shape[0]
    rho = np.zeros(n)
    repulsion = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
            x = r / r0 - 1.0
            repulsion += 2.0 * a * np.exp(-p * x)
            b = xi * xi * np.exp(-2.0 * q * x)
            rho[i] += b
            rho[j] += b
    energia = repulsion - np.sum(np.sqrt(rho))

    f = np.zeros_like(pos)
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r = np.sqrt(np.sum(d**2))
            x = r / r0 - 1.0
            d_rep = -2.0 * a * p / r0 * np.exp(-p * x)
            d_banda = (
                (0.5 / np.sqrt(rho[i]) + 0.5 / np.sqrt(rho[j]))
                * 2.0
                * q
                / r0
                * xi
                * xi
                * np.exp(-2.0 * q * x)
            )
            fuerza = -(d_rep + d_banda) * d / r
            f[i] += fuerza
            f[j] -= fuerza
    return energia, f


def energia_por_atomo(pos: np.ndarray, par: ParametrosTB = POTASIO) -> float:
    energia, _ = energia_y_fuerzas(
        np.ascontiguousarray(pos, dtype=np.float64), par.a, par.xi, par.r0, par.p, par.q
    )
    return energia / pos.shape[0]


def simplex_regular(n_atomos: int, distancia: float) -> np.ndarray:
    """Dímero, triángulo o tetraedro de lado `distancia` (n_atomos = 2, 3, 4)."""
    if n_atomos not in (2, 3, 4):
        raise ValueError("solo el dímero, el triángulo y el tetraedro son regulares")
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(2.0 / 3.0)],
        ]
    )
    return base[:n_atomos] * distancia


def integrar(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    n_pasos: int,
    par: ParametrosTB = POTASIO,
    masa: float = MASA_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity Verlet de un cúmulo. `dt` en unidades de `TIEMPO` fs.

    Devuelve las energías cinética y potencial (eV) en cada paso, con el paso
    0 como estado inicial.
    """
    pos = np.array(pos, dtype=np.float64)
    vel = np.array(vel, dtype=np.float64)
    ke = np.empty(n_pasos + 1)
    pe = np.empty(n_pasos + 1)
    args = (par.a, par.xi, par.r0, par.p, par.q)
    pe[0], f = energia_y_fuerzas(pos, *args)
    ke[0] = 0.5 * masa * np.sum(vel**2)
    for paso in range(1, n_pasos + 1):
        vel += 0.5 * dt * f / masa
        pos += dt * vel
        pe[paso], f = energia_y_fuerzas(pos, *args)
        vel += 0.5 * dt * f / masa
        ke[paso] = 0.5 * masa * np.sum(vel**2)
    return ke, pe
