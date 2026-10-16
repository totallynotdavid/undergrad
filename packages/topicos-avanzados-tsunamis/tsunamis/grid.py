"""Bathymetry and topography grids with their geographic coordinates.

The coordinates come from the header of the source file: its extent, its
spacing and its registration.
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
    reader flips it to latitude increasing. Its header gives the extent and the
    registration. In a grid-line registered file the extent is the position of
    the first and last points; in a pixel-registered one it is the outer edge of
    the cells, and the points are the cell centres.
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
    registration = None
    for line in header:
        match = re.search(r"is (grid line|pixel) registered", line)
        if match:
            registration = match.group(1)
    if registration is None:
        raise ValueError(f"{path} does not state its registration in its header")

    rows, cols = values.shape
    if registration == "grid line":
        lon = np.linspace(lon_min, lon_max, cols)
        lat = np.linspace(lat_min, lat_max, rows)
    else:
        lon = lon_min + (lon_max - lon_min) / cols * (np.arange(cols) + 0.5)
        lat = lat_min + (lat_max - lat_min) / rows * (np.arange(rows) + 0.5)
    source = next((line for line in header if line.startswith("Data source")), "")
    return Grid(lon, lat, values[::-1], source.removeprefix("Data source").strip())
