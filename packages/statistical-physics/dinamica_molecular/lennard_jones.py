"""Dinámica molecular de un fluido de Lennard-Jones (argón) en unidades reducidas.

Las longitudes están en unidades de sigma, las energías en epsilon, las masas en
la masa atómica y la temperatura en epsilon / k_B. Para el argón
(Verlet 1967): sigma = 3.405 A, epsilon / k_B = 119.8 K.

La caja es cúbica y periódica. Las posiciones nunca se envuelven dentro de la
caja: la convención de mínima imagen de `fuerzas` ya acepta cualquier
coordenada, y las posiciones sin envolver dan el desplazamiento cuadrático
medio directamente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

CORTE = 2.5


@dataclass(frozen=True)
class Trayectoria:
    """Resultado de `integrar`.

    `energia_cinetica` y `energia_potencial` son totales (no por partícula) y
    tienen `n_pasos + 1` valores: el paso 0 es el estado inicial. `posiciones`
    guarda una instantánea cada `cada` pasos, sin envolver en la caja.
    """

    energia_cinetica: np.ndarray
    energia_potencial: np.ndarray
    posiciones: np.ndarray
    pos_final: np.ndarray
    vel_final: np.ndarray
    dt: float
    cada: int

    @property
    def energia_total(self) -> np.ndarray:
        return self.energia_cinetica + self.energia_potencial

    @property
    def tiempo(self) -> np.ndarray:
        return self.dt * np.arange(self.energia_cinetica.size)

    @property
    def tiempo_cuadros(self) -> np.ndarray:
        return self.dt * self.cada * np.arange(self.posiciones.shape[0])

    @property
    def temperatura(self) -> np.ndarray:
        return temperatura(self.energia_cinetica, self.pos_final.shape[0])


def temperatura(energia_cinetica, n_particulas: int):
    """Temperatura instantánea con 3(N-1) grados de libertad (momento total nulo)."""
    return 2.0 * energia_cinetica / (3.0 * (n_particulas - 1))


def red_fcc(n_celdas: int, densidad: float) -> tuple[np.ndarray, float]:
    """Red cúbica centrada en las caras de 4 n_celdas^3 átomos.

    Devuelve las posiciones y el lado de la caja que da la `densidad` pedida.
    """
    n = 4 * n_celdas**3
    lado = (n / densidad) ** (1.0 / 3.0)
    a = lado / n_celdas
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    celdas = np.stack(np.meshgrid(*[np.arange(n_celdas)] * 3, indexing="ij"), -1)
    celdas = celdas.reshape(-1, 1, 3)
    return ((celdas + base) * a).reshape(-1, 3), lado


def velocidades_maxwell(
    n_particulas: int, temperatura_objetivo: float, rng: np.random.Generator
) -> np.ndarray:
    """Velocidades gaussianas con momento total nulo y temperatura exacta."""
    vel = rng.normal(size=(n_particulas, 3))
    vel -= vel.mean(axis=0)
    energia = 0.5 * np.sum(vel**2)
    objetivo = 1.5 * (n_particulas - 1) * temperatura_objetivo
    return vel * np.sqrt(objetivo / energia)


@numba.njit
def fuerzas(pos, lado, corte):
    """Fuerzas y energía potencial total, con mínima imagen.

    Usa el potencial truncado y desplazado para que fuerza y energía se anulen
    en `corte`; así la energía total se conserva sin saltos al cruzar el corte.
    """
    n = pos.shape[0]
    f = np.zeros_like(pos)
    inv6_c = 1.0 / corte**6
    fuerza_c = 24.0 * inv6_c * (2.0 * inv6_c - 1.0) / corte
    pot_c = 4.0 * inv6_c * (inv6_c - 1.0)
    pe = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dx -= lado * np.round(dx / lado)
            dy -= lado * np.round(dy / lado)
            dz -= lado * np.round(dz / lado)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < corte * corte:
                r = np.sqrt(r2)
                inv6 = 1.0 / r2**3
                fuerza = 24.0 * inv6 * (2.0 * inv6 - 1.0) / r - fuerza_c
                pe += 4.0 * inv6 * (inv6 - 1.0) - pot_c + (r - corte) * fuerza_c
                fx = fuerza * dx / r
                fy = fuerza * dy / r
                fz = fuerza * dz / r
                f[i, 0] += fx
                f[i, 1] += fy
                f[i, 2] += fz
                f[j, 0] -= fx
                f[j, 1] -= fy
                f[j, 2] -= fz
    return f, pe


@numba.njit
def energia_truncada(pos, lado, corte):
    """Energía de Lennard-Jones truncada en `corte`, sin desplazar.

    Es la que usó Verlet (1967). Sirve para comparar con sus resultados, no
    para integrar: salta al cruzar el corte.
    """
    n = pos.shape[0]
    pe = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            r2 = 0.0
            for k in range(3):
                d = pos[i, k] - pos[j, k]
                d -= lado * np.round(d / lado)
                r2 += d * d
            if r2 < corte * corte:
                inv6 = 1.0 / r2**3
                pe += 4.0 * inv6 * (inv6 - 1.0)
    return pe


def correccion_cola(densidad: float, corte: float = CORTE) -> float:
    """Energía por partícula de las interacciones más allá de `corte`, con g = 1."""
    return 8.0 * np.pi * densidad / 3.0 * (1.0 / (3.0 * corte**9) - 1.0 / corte**3)


@numba.njit
def _integrar(pos, vel, lado, dt, n_pasos, corte, cada, t_objetivo, cada_reescala):
    n = pos.shape[0]
    pos = pos.copy()
    vel = vel.copy()
    ke = np.empty(n_pasos + 1)
    pe = np.empty(n_pasos + 1)
    cuadros = np.empty((n_pasos // cada + 1, n, 3))
    f, pe[0] = fuerzas(pos, lado, corte)
    ke[0] = 0.5 * np.sum(vel * vel)
    cuadros[0] = pos
    for paso in range(1, n_pasos + 1):
        vel += 0.5 * dt * f
        pos += dt * vel
        f, pe[paso] = fuerzas(pos, lado, corte)
        vel += 0.5 * dt * f
        if t_objetivo > 0.0 and paso % cada_reescala == 0:
            t_actual = 2.0 * 0.5 * np.sum(vel * vel) / (3.0 * (n - 1))
            vel *= np.sqrt(t_objetivo / t_actual)
        ke[paso] = 0.5 * np.sum(vel * vel)
        if paso % cada == 0:
            cuadros[paso // cada] = pos
    return ke, pe, cuadros, pos, vel


def integrar(
    pos: np.ndarray,
    vel: np.ndarray,
    lado: float,
    dt: float,
    n_pasos: int,
    *,
    corte: float = CORTE,
    cada: int = 1,
    t_objetivo: float = 0.0,
    cada_reescala: int = 50,
) -> Trayectoria:
    """Integra con velocity Verlet. No modifica `pos` ni `vel`.

    Con `t_objetivo > 0` reescala las velocidades cada `cada_reescala` pasos
    (equilibrado). Con `t_objetivo = 0` el sistema es microcanónico y la
    energía total debe conservarse.
    """
    if corte > lado / 2:
        raise ValueError("el corte debe ser menor que la mitad de la caja")
    ke, pe, cuadros, pos_f, vel_f = _integrar(
        np.ascontiguousarray(pos, dtype=np.float64),
        np.ascontiguousarray(vel, dtype=np.float64),
        float(lado),
        float(dt),
        int(n_pasos),
        float(corte),
        int(cada),
        float(t_objetivo),
        int(cada_reescala),
    )
    return Trayectoria(ke, pe, cuadros, pos_f, vel_f, dt, cada)
