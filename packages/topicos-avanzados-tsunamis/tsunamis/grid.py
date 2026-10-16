"""Bathymetry and topography grids with their geographic coordinates.

One reader replaces the MATLAB chain of the Topicos Avanzados en Tsunamis
course (``extraer_etopo5.m``, ``extraer_topo.m``, ``gebco2mat.m``,
``mat2vec.m`` and ``fix_latitud.m``). Those programs transposed the grid,
mirrored it, flipped the sign of the elevation and rebuilt the coordinates by
hand, and the ``xya.mat`` files they wrote did not match the data. Here the
coordinates come from the header of the source file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Elevation (m, positive up) on a regular longitude-latitude grid.

    ``lon`` and ``lat`` are the coordinates of the cell centres in degrees,
    both increasing. ``elevation`` has shape ``(len(lat), len(lon))``.
    """

    lon: np.ndarray
    lat: np.ndarray
    elevation: np.ndarray
    source: str = ""

    @property
    def depth(self) -> np.ndarray:
        """Water depth (m), zero on land."""
        return np.maximum(-self.elevation, 0.0)

    def crop(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> Grid:
        """Cells whose centre lies inside the rectangle."""
        i = np.flatnonzero((self.lon >= lon_min) & (self.lon <= lon_max))
        j = np.flatnonzero((self.lat >= lat_min) & (self.lat <= lat_max))
        return Grid(
            self.lon[i[0] : i[-1] + 1],
            self.lat[j[0] : j[-1] + 1],
            self.elevation[j[0] : j[-1] + 1, i[0] : i[-1] + 1],
            self.source,
        )

    def save(self, path: str | Path) -> None:
        """Write the grid to a compressed ``.npz`` with 16-bit elevations."""
        np.savez_compressed(
            path,
            lon=self.lon,
            lat=self.lat,
            elevation=np.rint(self.elevation).astype(np.int16),
            source=np.array(self.source),
        )


def read_grid(path: str | Path) -> Grid:
    """Read a grid from a GEBCO ASCII file (``.asc``) or from a ``.npz``.

    A GEBCO file lists the rows from the north-west corner eastwards, so the
    reader flips it to latitude increasing. Its header gives the extent and
    spacing of a pixel-registered grid, whose points are the cell centres.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return Grid(
                data["lon"],
                data["lat"],
                data["elevation"].astype(float),
                str(data["source"]),
            )
    return _read_gebco_ascii(path)


def _read_gebco_ascii(path: Path) -> Grid:
    header: list[str] = []
    with path.open() as f:
        for line in f:
            if not line.startswith("%"):
                break
            header.append(line[1:].strip())
    values = np.loadtxt(path, comments="%")

    def field(name: str) -> float:
        for line in header:
            match = re.match(rf"{name}\s*:\s*(-?[\d.]+)", line)
            if match:
                return float(match.group(1))
        raise ValueError(f"{path} has no '{name}' in its header")

    lon_min, lon_max = field("Minimum longitude"), field("Maximum longitude")
    lat_min, lat_max = field("Minimum latitude"), field("Maximum latitude")
    rows, cols = values.shape
    dlon = (lon_max - lon_min) / cols
    dlat = (lat_max - lat_min) / rows
    lon = lon_min + dlon * (np.arange(cols) + 0.5)
    lat = lat_min + dlat * (np.arange(rows) + 0.5)
    source = next((line for line in header if line.startswith("Data source")), "")
    return Grid(lon, lat, values[::-1], source.removeprefix("Data source").strip())
