"""Kajiura filter: from seafloor uplift to the initial sea surface.

The sea surface does not copy a sudden seafloor uplift. Linear potential theory
for an incompressible ocean of depth ``h`` gives an initial elevation whose
Fourier transform is the transform of the uplift divided by ``cosh(k h)``,
with ``k`` the wavenumber. Wavelengths much longer than the
depth pass unchanged; short ones are suppressed. The course replaced this step
with a Laplacian filter in ``dib_fuente.m``.

Where the depth varies, :func:`kajiura_filter` filters the field once per depth
level and interpolates between levels with the local depth. Abbate et al.
(2024, Nat. Hazards Earth Syst. Sci. 24, 2773) evaluate the same filter cell by
cell with the local depth, which this interpolation only approximates.
"""

from __future__ import annotations

import numpy as np


def kajiura_filter(
    uplift: np.ndarray,
    depth: np.ndarray,
    dx: float,
    dy: float,
    *,
    step: float = 250.0,
) -> np.ndarray:
    """Initial sea surface (m) for a seafloor ``uplift`` (m).

    ``uplift`` and ``depth`` (m, zero or negative on land) are arrays of shape
    ``(ny, nx)``; ``dx`` and ``dy`` are the cell sizes in metres. The depth
    levels are ``step`` metres apart. The field is mirrored across its edges
    before the transform, so the periodic boundary of the FFT meets no jump
    even where the uplift is not zero at the border.
    """
    uplift = np.asarray(uplift, dtype=float)
    depth = np.maximum(np.asarray(depth, dtype=float), 0.0)
    if not depth.any():
        return uplift.copy()
    ny, nx = uplift.shape
    padded = np.pad(uplift, ((0, ny), (0, nx)), mode="symmetric")
    spectrum = np.fft.rfft2(padded)
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=dy)
    kx = 2.0 * np.pi * np.fft.rfftfreq(padded.shape[1], d=dx)
    k = np.hypot(ky[:, None], kx[None, :])

    levels = np.linspace(0.0, depth.max(), max(2, int(np.ceil(depth.max() / step)) + 1))
    fields = np.stack(
        [
            np.fft.irfft2(spectrum / np.cosh(k * h), s=padded.shape)[:ny, :nx]
            for h in levels
        ]
    )
    upper = np.clip(np.searchsorted(levels, depth), 1, len(levels) - 1)
    lower = upper - 1
    weight = (depth - levels[lower]) / (levels[upper] - levels[lower])
    rows, cols = np.indices(depth.shape)
    return (
        fields[lower, rows, cols] * (1.0 - weight) + fields[upper, rows, cols] * weight
    )
