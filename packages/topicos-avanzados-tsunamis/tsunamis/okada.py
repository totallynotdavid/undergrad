"""Vertical seafloor deformation of rectangular faults after Okada (1985).

Only the vertical component is computed. The evaluation is vectorised over the
points.

Geometry. A fault is a rectangle that dips towards azimuth ``strike + 90``.
Its origin is the surface projection of the corner of the upper edge at the
start of the strike direction. From there the upper edge runs ``length`` metres
along ``strike`` at depth ``depth``, and the plane continues ``width`` metres
down dip, which is ``width * cos(dip)`` metres horizontally.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

# P and S wave speeds (m/s) of the crust model.
VP = 4.82e3
VS = 2.78e3

# Distance below which the Okada kernels switch to their limit forms.
EPS = 1.0e-8


@dataclass(frozen=True)
class Subfault:
    """Rectangular dislocation.

    ``east`` and ``north`` locate the origin (see the module docstring) in
    metres in the frame of the grid. ``length`` and ``width`` are in metres,
    ``depth`` is the depth of the upper edge in metres, ``strike``, ``dip`` and
    ``rake`` are in degrees and ``slip`` is the dislocation in metres.
    """

    east: float
    north: float
    length: float
    width: float
    depth: float
    strike: float
    dip: float
    rake: float
    slip: float


def rigidity_ratio(vp: float = VP, vs: float = VS) -> float:
    """Return mu / (lambda + mu), the elastic constant of Okada's kernels."""
    return vs**2 / (vp**2 - vs**2)


def vertical_displacement(
    east: np.ndarray,
    north: np.ndarray,
    faults: Iterable[Subfault],
    *,
    ratio: float | None = None,
) -> np.ndarray:
    """Vertical displacement (m) at the points ``east``, ``north`` (m).

    ``east`` and ``north`` are arrays of the same shape, or broadcastable to
    one. The result is the sum of the contribution of every fault. Points are
    evaluated at the free surface. ``ratio`` is mu / (lambda + mu); it defaults
    to the value of the crust model (``VP``, ``VS``).
    """
    ratio = rigidity_ratio() if ratio is None else ratio
    east, north = np.broadcast_arrays(
        np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    )
    total = np.zeros(east.shape)
    for fault in faults:
        total += _single_fault(east - fault.east, north - fault.north, fault, ratio)
    return total


def _single_fault(
    east: np.ndarray, north: np.ndarray, fault: Subfault, ratio: float
) -> np.ndarray:
    strike = np.radians(fault.strike)
    dip = np.radians(fault.dip)
    rake = np.radians(fault.rake)
    cs, sn = np.cos(dip), np.sin(dip)

    # Coordinates along strike and across it. Okada's y axis points up dip.
    x = east * np.sin(strike) + north * np.cos(strike)
    down_dip = east * np.cos(strike) - north * np.sin(strike)
    y = fault.width * cs - down_dip

    # Depth of the lower edge, the reference depth of Okada's formulas.
    d = fault.depth + fault.width * sn
    p = y * cs + d * sn
    q = y * sn - d * cs

    def corners(kernel, slip):
        w, length = fault.width, fault.length
        return (
            kernel(x, p, q, cs, sn, ratio)
            - kernel(x, p - w, q, cs, sn, ratio)
            - kernel(x - length, p, q, cs, sn, ratio)
            + kernel(x - length, p - w, q, cs, sn, ratio)
        ) * (-slip / (2.0 * np.pi))

    return corners(_strike_slip_z, fault.slip * np.cos(rake)) + corners(
        _dip_slip_z, fault.slip * np.sin(rake)
    )


def _log_terms(xi, eta, q, cs, sn, r):
    """ln(R + eta) and ln(R + d~) with the limit forms of the singular lines."""
    dh = eta * sn - q * cs
    singular_eta = np.abs(r + eta) < EPS
    singular_d = np.abs(r + dh) < EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        log_eta = np.where(
            singular_eta, -np.log(np.where(singular_eta, r - eta, 1.0)), np.log(r + eta)
        )
        log_d = np.where(
            singular_d, -np.log(np.where(singular_d, r - dh, 1.0)), np.log(r + dh)
        )
    return log_eta, log_d, singular_d


def _vertical_i(xi, eta, q, cs, sn, ratio, r, log_eta, log_d, singular_d):
    """The I4 and I5 terms of Okada (1985, eq. 28) that enter Uz."""
    dh = eta * sn - q * cs
    xx = np.sqrt(xi**2 + q**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(cs) >= EPS:
            i4 = ratio * (log_d - sn * log_eta) / cs
            arg = (eta * (xx + q * cs) + xx * (r + xx) * sn) / (xi * (r + xx) * cs)
            i5 = np.where(np.abs(xi) >= EPS, 2.0 * ratio * np.arctan(arg) / cs, 0.0)
        else:
            i4 = np.where(singular_d, 0.0, -ratio * q / (r + dh))
            i5 = np.where(
                (np.abs(xi) >= EPS) & ~singular_d, -ratio * xi * sn / (r + dh), 0.0
            )
    return i4, i5


def _strike_slip_z(xi, eta, q, cs, sn, ratio):
    r = np.sqrt(xi**2 + eta**2 + q**2)
    dh = eta * sn - q * cs
    log_eta, log_d, singular_d = _log_terms(xi, eta, q, cs, sn, r)
    i4, _ = _vertical_i(xi, eta, q, cs, sn, ratio, r, log_eta, log_d, singular_d)
    regular = np.abs(r + eta) >= EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(regular, r + eta, 1.0)
        first = np.where(regular, dh * q / (r * safe), 0.0)
        second = np.where(regular, q * sn / safe, 0.0)
    return first + second + i4 * sn


def _dip_slip_z(xi, eta, q, cs, sn, ratio):
    r = np.sqrt(xi**2 + eta**2 + q**2)
    dh = eta * sn - q * cs
    log_eta, log_d, singular_d = _log_terms(xi, eta, q, cs, sn, r)
    _, i5 = _vertical_i(xi, eta, q, cs, sn, ratio, r, log_eta, log_d, singular_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        regular = np.abs(r + xi) >= EPS
        first = np.where(regular, dh * q / (r * np.where(regular, r + xi, 1.0)), 0.0)
        has_q = np.abs(q) >= EPS
        second = np.where(
            has_q,
            sn * np.arctan(xi * eta / (np.where(has_q, q, 1.0) * r)),
            0.0,
        )
    return first + second - i5 * sn * cs
