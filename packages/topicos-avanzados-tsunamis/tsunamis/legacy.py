"""Reproduction of a ``def_oka.f`` run, to check the port against its output.

The Fortran program works on grid indices: it reads the origin of each fault as
a cell of the bathymetry grid and evaluates the displacement on a window of
that grid with a fixed cell size. This module repeats that convention,
including its one-cell offset of the origin, so a run can be compared cell by
cell with a ``deform_a.grd`` it wrote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsunamis.okada import Subfault, vertical_displacement

# Window and cell size set in the PARAMETER statements of def_oka.f.
IDS, IDE, JDS, JDE = 153, 637, 844, 1396
DX = 926.6244


def read_pfalla(path: str | Path) -> np.ndarray:
    """Read a ``pfalla.inp``: one row per subfault of nine numbers.

    The columns are I0, J0 (grid cell of the origin), slip (m), length (m),
    width (m), strike, dip, rake (degrees) and depth of the upper edge (m). The
    numbers may be spread over several lines.
    """
    values = np.loadtxt(path, ndmin=1).ravel()
    return values.reshape(-1, 9)


def run_def_oka(
    pfalla: np.ndarray,
    *,
    window: tuple[int, int, int, int] = (IDS, IDE, JDS, JDE),
    dx: float = DX,
) -> np.ndarray:
    """Vertical displacement (m) as ``def_oka.f`` computes it.

    ``window`` is ``(IDS, IDE, JDS, JDE)``, 1-based and inclusive. The result
    has shape ``(IDE - IDS + 1, JDE - JDS + 1)``, longitude first, like the
    rows and columns of ``deform_a.grd``.
    """
    ids, ide, jds, jde = window
    i = np.arange(ide - ids + 1)[:, None]
    j = np.arange(jde - jds + 1)[None, :]
    faults = []
    for i0, j0, slip, length, width, strike, dip, rake, depth in pfalla:
        # The program subtracts IDS - 1 from the origin cell and then measures
        # the zero-based cell index from it, which puts the origin one cell
        # east and north of the cell named in the input.
        faults.append(
            Subfault(
                (i0 - ids + 1) * dx,
                (j0 - jds + 1) * dx,
                length,
                width,
                depth,
                strike,
                dip,
                rake,
                slip,
            )
        )
    return vertical_displacement(i * dx + 0 * j, j * dx + 0 * i, faults)
