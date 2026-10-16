"""Vertical deformation after Mansinha and Smylie (1971), for cross-checking.

Port to vectorised NumPy of ``deform.f`` of the Topicos Avanzados en Tsunamis
course, based on the dislocation model of L. Mansinha and D. E. Smylie and
modified by Cesar Jimenez (2022). It evaluates the same rectangular fault as
:mod:`tsunamis.okada` with the closed form of Mansinha and Smylie, so the two
models can be compared. The model assumes a Poisson solid.
"""

from __future__ import annotations

import numpy as np

from tsunamis.okada import Subfault


def vertical_displacement(
    east: np.ndarray, north: np.ndarray, fault: Subfault
) -> np.ndarray:
    """Vertical displacement (m) at the surface for one fault.

    Same arguments and fault geometry as
    :func:`tsunamis.okada.vertical_displacement`. The two codes measure the
    rake from opposite sides of the slip vector; this function takes the rake of
    Okada's convention and converts it.
    """
    east, north = np.broadcast_arrays(
        np.asarray(east, dtype=float) - fault.east,
        np.asarray(north, dtype=float) - fault.north,
    )
    strike = np.radians(fault.strike)
    dip = np.radians(fault.dip)
    rake = np.radians(180.0 - fault.rake)

    x1 = east * np.sin(strike) + north * np.cos(strike)
    x2 = east * np.cos(strike) - north * np.sin(strike) + fault.depth / np.tan(dip)

    upper = fault.depth / np.sin(dip)
    lower = upper + fault.width
    strike_slip = fault.slip * np.cos(rake)
    dip_slip = fault.slip * np.sin(rake)

    def corners(kernel):
        return (
            kernel(x1, x2, fault.length, lower, dip)
            - kernel(x1, x2, fault.length, upper, dip)
            - kernel(x1, x2, 0.0, lower, dip)
            + kernel(x1, x2, 0.0, upper, dip)
        )

    scale = 1.0 / (12.0 * np.pi)
    return (corners(_strike) * strike_slip + corners(_dip) * dip_slip) * scale


def _atan(ax, ay):
    """Arctangent of ax / ay in (-pi/2, pi/2], zero where both vanish."""
    angle = np.arctan2(np.abs(ax), np.abs(ay))
    angle = np.where(ax * ay < 0, -angle, angle)
    return np.where((np.abs(ax) <= 1e-6) & (np.abs(ay) <= 1e-6), 0.0, angle)


def _surface_terms(x1, x2, c, cc, dp):
    """Quantities shared by both kernels for a point on the free surface.

    With x3 = 0 the distances to the fault edge and to its image coincide, so
    R = Q, and the primed coordinates of the original program reduce to x2 times
    a trigonometric factor.
    """
    sn, cs = np.sin(dp), np.cos(dp)
    c2, c3 = cc * cs, cc * sn
    r = np.sqrt((x1 - c) ** 2 + (x2 - c2) ** 2 + c3**2)
    return sn, cs, c3, r, x2 * sn, x2 * cs


def _strike(x1, x2, c, cc, dp):
    sn, cs, c3, r, r2, r3 = _surface_terms(x1, x2, c, cc, dp)
    q2, q3 = r2, -r3
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.log(r + r3 - cc)
        a2 = np.log(r + q3 + cc)
        a3 = np.log(r + c3)
        b1 = 1.0 + 3.0 * np.tan(dp) ** 2
        b2 = 3.0 * np.tan(dp) / cs
        b8 = 2.0 * (q2 + x2 * sn) * (q3 * sn)
        return (
            cs * (a1 + b1 * a2 - b2 * a3)
            + 2.0 * r2 * sn / r
            + 2.0 * sn * (q2 + x2 * sn) / r
            - 2.0 * r2**2 * cs / (r * (r + r3 - cc))
            - b8 / (r * (r + q3 + cc))
        )


def _dip(x1, x2, c, cc, dp):
    sn, cs, c3, r, r2, r3 = _surface_terms(x1, x2, c, cc, dp)
    q2, q3 = r2, -r3
    h = np.sqrt(q2**2 + (q3 + cc) ** 2)
    d1, d2, d3, d4 = x1 - c, x2 - cc * cs, -c3, c3
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = r * (r + d1)
        t1 = _atan(d1 * d2, (h + d4) * (r + h))
        t2 = _atan(d1 * (r3 - cc), r2 * r)
        t3 = _atan(d1 * (q3 + cc), q2 * r)
        # ln(R + d1) - ln(Q + d1) cancels because R = Q at the surface.
        return sn * (6.0 * d3 * d2 / b1 - 6.0 * t1 + 3.0 * t2 - 6.0 * t3) - cs * (
            6.0 * c3**2 / b1
        )
