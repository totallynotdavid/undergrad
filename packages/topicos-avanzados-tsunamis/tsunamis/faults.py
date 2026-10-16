"""Fault geometry and source parameters for the Okada model.

* :func:`papazachos_scaling`: rupture size from the moment magnitude, with the
  empirical relations for dip-slip faults in subduction regions.
* :func:`fault_plane`: the subfaults of a multiple fault.
* :func:`subfaults_from_inversion`: conversion of the result of a slip
  inversion into subfaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tsunamis.okada import Subfault

# WGS84 ellipsoid.
EQUATORIAL_RADIUS = 6378137.0
FLATTENING = 1.0 / 298.257223563

# Average rigidity (N/m^2) of the crust.
RIGIDITY = 4.0e10

# Magnitude range of the relations for subduction regions: 6.7 <= M <= 9.2.
MIN_MAGNITUDE = 6.7
MAX_MAGNITUDE = 9.2


@dataclass(frozen=True)
class Scaling:
    """Rupture parameters of an earthquake of a given magnitude."""

    magnitude: float
    length_km: float
    width_km: float
    # Slip (m) from the moment, M0 = rigidity * length * width * slip.
    slip_m: float
    # Slip (m) from the empirical relation of the paper.
    empirical_slip_m: float
    moment_nm: float
    # Magnitudes that give this length within one standard deviation (0.18).
    magnitude_range: tuple[float, float]


def papazachos_scaling(magnitude: float) -> Scaling:
    """Length, width and slip of a rupture of moment magnitude ``magnitude``."""
    if not MIN_MAGNITUDE <= magnitude <= MAX_MAGNITUDE:
        raise ValueError(
            f"the scaling relations hold for {MIN_MAGNITUDE} <= Mw <= {MAX_MAGNITUDE}"
        )
    length = 10.0 ** (0.55 * magnitude - 2.19)
    width = 10.0 ** (0.31 * magnitude - 0.63)
    empirical_slip = 10.0 ** (0.64 * magnitude - 2.78) / 100.0
    moment = 10.0 ** (1.5 * magnitude + 9.1)
    slip = moment / (RIGIDITY * length * 1000.0 * width * 1000.0)
    spread = 0.18
    low = (np.log10(length) - spread + 2.19) / 0.55
    high = (np.log10(length) + spread + 2.19) / 0.55
    return Scaling(magnitude, length, width, slip, empirical_slip, moment, (low, high))


def metres_per_degree(lat: float) -> tuple[float, float]:
    """Metres per degree of longitude and of latitude at latitude ``lat``."""
    phi = np.radians(lat)
    e2 = 2.0 * FLATTENING - FLATTENING**2
    w = np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    per_lon = np.radians(1.0) * EQUATORIAL_RADIUS * np.cos(phi) / w
    per_lat = np.radians(1.0) * EQUATORIAL_RADIUS * (1.0 - e2) / w**3
    return float(per_lon), float(per_lat)


def wrap_longitude(lon):
    """Longitude in [-180, 180), element by element."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def local_xy(lon, lat, lon0: float, lat0: float):
    """East and north offsets (m) of ``lon``, ``lat`` from ``lon0``, ``lat0``.

    Linear in degrees, with the scale of the ellipsoid at ``lat0``. It is the
    projection :func:`fault_plane` uses to lay out the subfaults.
    """
    per_lon, per_lat = metres_per_degree(lat0)
    dlon = wrap_longitude(np.asarray(lon, dtype=float) - lon0)
    return dlon * per_lon, (np.asarray(lat, dtype=float) - lat0) * per_lat


@dataclass(frozen=True)
class GeoSubfault:
    """A :class:`~tsunamis.okada.Subfault` located by longitude and latitude."""

    lon: float
    lat: float
    length: float
    width: float
    depth: float
    strike: float
    dip: float
    rake: float
    slip: float


def to_local(subfaults: list[GeoSubfault], lon0: float, lat0: float) -> list[Subfault]:
    """Place geographic subfaults in the metre frame centred on ``lon0``, ``lat0``."""
    placed = []
    for s in subfaults:
        east, north = local_xy(s.lon, s.lat, lon0, lat0)
        placed.append(
            Subfault(
                float(east),
                float(north),
                s.length,
                s.width,
                s.depth,
                s.strike,
                s.dip,
                s.rake,
                s.slip,
            )
        )
    return placed


@dataclass(frozen=True)
class FaultPlane:
    """Outline of a fault and its subfaults."""

    # Closed outline, five rows of (lon, lat).
    outline: np.ndarray
    subfaults: list[GeoSubfault]

    @property
    def centroid(self) -> tuple[float, float]:
        """Centre of the surface projection of the fault, (lon, lat)."""
        lon, lat = self.outline[:4].mean(axis=0)
        return float(lon), float(lat)


def fault_plane(
    lon: float,
    lat: float,
    *,
    length: float,
    width: float,
    depth: float,
    strike: float,
    dip: float,
    rake: float,
    slip: float,
    along_dip: int = 1,
    along_strike: int = 1,
) -> FaultPlane:
    """Divide a rectangular fault into ``along_dip`` x ``along_strike`` subfaults.

    ``lon``, ``lat`` locate the origin (see :mod:`tsunamis.okada`) and
    ``depth`` is the depth of the upper edge. Lengths are in metres. Every
    subfault carries the same slip, so together they reproduce the whole fault.
    The subfaults are ordered along strike first, and within each strip from
    the shallowest to the deepest.
    """
    per_lon, per_lat = metres_per_degree(lat)
    st, dp = np.radians(strike), np.radians(dip)
    w_h = width * np.cos(dp)

    strike_step = np.array([np.sin(st) / per_lon, np.cos(st) / per_lat])
    dip_step = np.array([np.cos(st) / per_lon, -np.sin(st) / per_lat])
    origin = np.array([lon, lat], dtype=float)
    outline = np.array(
        [
            origin,
            origin + length * strike_step,
            origin + length * strike_step + w_h * dip_step,
            origin + w_h * dip_step,
            origin,
        ]
    )

    sub_length = length / along_strike
    sub_width = width / along_dip
    rise = sub_width * np.sin(dp)
    subfaults = []
    for i in range(along_strike):
        for j in range(along_dip):
            lon_k, lat_k = (
                origin
                + (i * sub_length) * strike_step
                + (j * sub_width * np.cos(dp)) * dip_step
            )
            subfaults.append(
                GeoSubfault(
                    float(lon_k),
                    float(lat_k),
                    sub_length,
                    sub_width,
                    depth + j * rise,
                    strike,
                    dip,
                    rake,
                    slip,
                )
            )
    return FaultPlane(outline, subfaults)


class InversionFormat(Enum):
    """Layouts of the result files of a slip inversion."""

    FUJI = 1
    CALTECH = 2
    PULIDO = 3
    KANAMORI = 4
    FAULT = 5


def subfaults_from_inversion(
    table: np.ndarray,
    fmt: InversionFormat,
    *,
    length_km: float | None = None,
    width_km: float | None = None,
) -> list[GeoSubfault]:
    """Convert the rows of an inversion result into subfaults.

    ``table`` has one row per subfault. The columns follow ``fmt``:

    * ``FUJI``: length (m), width (m), depth (m), strike, dip, rake, slip (m),
      lat, lon.
    * ``CALTECH``: lat, lon, depth (km), slip (cm), rake, strike, dip. The size
      of every subfault is ``length_km`` x ``width_km``.
    * ``PULIDO``: lon, lat, slip (cm), rake, strike, dip, depth (km), with
      subfaults of 20 x 20 km.
    * ``KANAMORI``: length (km), width (km), depth (km), strike, dip, rake,
      slip (m), lat, lon.
    * ``FAULT``: lon, lat, slip (m), length (m), width (m), strike, dip, rake,
      depth (m), the format written for the Okada program.
    """
    table = np.atleast_2d(np.asarray(table, dtype=float))
    c = table.T
    fmt = InversionFormat(fmt)
    n = table.shape[0]
    if fmt is InversionFormat.FUJI:
        length, width, depth, strike, dip, rake, slip, lat, lon = c[:9]
    elif fmt is InversionFormat.KANAMORI:
        length, width, depth = c[0] * 1e3, c[1] * 1e3, c[2] * 1e3
        strike, dip, rake, slip, lat, lon = c[3:9]
    elif fmt is InversionFormat.CALTECH:
        if length_km is None or width_km is None:
            raise ValueError("the Caltech format needs length_km and width_km")
        lat, lon, depth = c[0], c[1], c[2] * 1e3
        slip, rake, strike, dip = c[3] / 100.0, c[4], c[5], c[6]
        length = np.full(n, length_km * 1e3)
        width = np.full(n, width_km * 1e3)
    elif fmt is InversionFormat.PULIDO:
        lon, lat, slip, rake, strike, dip, depth = c[:7]
        slip, depth = slip / 100.0, depth * 1e3
        length = np.full(n, 20e3)
        width = np.full(n, 20e3)
    else:
        lon, lat, slip, length, width, strike, dip, rake, depth = c[:9]
    lon = wrap_longitude(lon)
    return [
        GeoSubfault(
            float(lon[k]),
            float(lat[k]),
            float(length[k]),
            float(width[k]),
            float(depth[k]),
            float(strike[k]),
            float(dip[k]),
            float(rake[k]),
            float(slip[k]),
        )
        for k in range(n)
    ]
