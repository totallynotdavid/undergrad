import unittest

import numpy as np
from tsunamis.faults import (
    InversionFormat,
    fault_plane,
    local_xy,
    metres_per_degree,
    papazachos_scaling,
    subfaults_from_inversion,
    wrap_longitude,
)


class Scaling(unittest.TestCase):
    def test_magnitude_8(self):
        s = papazachos_scaling(8.0)
        self.assertAlmostEqual(s.length_km, 162.18, places=2)
        self.assertAlmostEqual(s.width_km, 70.79, places=2)
        # M0 = 10**(1.5 * 8 + 9.1) N m shared over the rupture area at 40 GPa.
        self.assertAlmostEqual(s.moment_nm, 1.2589e21, delta=1e17)
        self.assertAlmostEqual(s.slip_m, 2.74, places=2)

    def test_rejects_magnitudes_outside_the_fitted_range(self):
        for magnitude in (6.4, 6.6, 9.3):
            with self.assertRaises(ValueError):
                papazachos_scaling(magnitude)
        papazachos_scaling(6.7)
        papazachos_scaling(9.2)


class Geometry(unittest.TestCase):
    def test_metres_per_degree(self):
        per_lon, per_lat = metres_per_degree(0.0)
        self.assertAlmostEqual(per_lon, 111319.49, delta=0.1)
        self.assertAlmostEqual(per_lat, 110574.28, delta=0.1)
        per_lon, per_lat = metres_per_degree(-14.0)
        self.assertAlmostEqual(per_lat, 110.6e3, delta=100.0)
        self.assertAlmostEqual(
            per_lon / 111319.49, np.cos(np.radians(14.0)), delta=2e-3
        )

    def test_local_xy_crosses_the_antimeridian(self):
        east, north = local_xy(-179.9, 0.0, 179.9, 0.0)
        self.assertAlmostEqual(east, 0.2 * 111319.49, delta=1.0)
        self.assertAlmostEqual(north, 0.0)

    def test_fault_plane_reproduces_the_pisco_2007_table(self):
        # lab3.pdf, Table I: subfault 1 and the subfaults that follow it down
        # dip (2) and along strike (3). Squares of 45 km, strike 325, dip 18.
        plane = fault_plane(
            -76.300,
            -14.750,
            length=45e3,
            width=90e3,
            depth=14e3,
            strike=325.0,
            dip=18.0,
            rake=63.0,
            slip=2.3,
            along_dip=2,
        )
        deeper = plane.subfaults[1]
        self.assertAlmostEqual(deeper.lon, -75.983, delta=0.01)
        self.assertAlmostEqual(deeper.lat, -14.533, delta=0.01)
        self.assertAlmostEqual(deeper.depth, 27.91e3, delta=100.0)

        plane = fault_plane(
            -76.300,
            -14.750,
            length=90e3,
            width=45e3,
            depth=14e3,
            strike=325.0,
            dip=18.0,
            rake=63.0,
            slip=2.3,
            along_strike=2,
        )
        along = plane.subfaults[1]
        self.assertAlmostEqual(along.lon, -76.533, delta=0.01)
        self.assertAlmostEqual(along.lat, -14.417, delta=0.01)
        self.assertEqual(along.depth, 14e3)

    def test_subfaults_are_ordered_along_strike_then_down_dip(self):
        plane = fault_plane(
            0.0,
            0.0,
            length=40e3,
            width=20e3,
            depth=1e3,
            strike=0.0,
            dip=30.0,
            rake=90.0,
            slip=1.0,
            along_dip=2,
            along_strike=2,
        )
        depths = [s.depth for s in plane.subfaults]
        self.assertEqual(depths[0], depths[2])
        self.assertLess(depths[0], depths[1])
        self.assertLess(plane.subfaults[0].lat, plane.subfaults[2].lat)

    def test_outline_and_centroid(self):
        plane = fault_plane(
            -77.0,
            -14.0,
            length=160e3,
            width=70e3,
            depth=5e3,
            strike=310.0,
            dip=18.0,
            rake=63.0,
            slip=3.0,
        )
        np.testing.assert_allclose(plane.outline[0], plane.outline[4])
        east, north = local_xy(*plane.outline[:4].T, -77.0, -14.0)
        # The outline is a rectangle: adjacent sides are perpendicular.
        side = np.array([east[1] - east[0], north[1] - north[0]])
        other = np.array([east[3] - east[0], north[3] - north[0]])
        self.assertAlmostEqual(
            float(side @ other),
            0.0,
            delta=1e-3 * np.linalg.norm(side) * np.linalg.norm(other),
        )
        self.assertAlmostEqual(np.hypot(*side), 160e3, delta=1.0)
        self.assertAlmostEqual(
            np.hypot(*other), 70e3 * np.cos(np.radians(18.0)), delta=1.0
        )
        lon, lat = plane.centroid
        self.assertAlmostEqual(lon, plane.outline[:4, 0].mean())
        self.assertAlmostEqual(lat, plane.outline[:4, 1].mean())


class Inversion(unittest.TestCase):
    def test_each_longitude_is_wrapped_on_its_own(self):
        # Each row may use either longitude convention, so wrapping must be
        # applied independently to preserve every subfault's location.
        table = np.array(
            [
                [-76.3, -14.75, 2.3, 45e3, 45e3, 325, 18, 63, 14e3],
                [283.7, -14.75, 2.3, 45e3, 45e3, 325, 18, 63, 14e3],
                [-76.0, -14.75, 2.3, 45e3, 45e3, 325, 18, 63, 14e3],
            ]
        )
        subfaults = subfaults_from_inversion(table, InversionFormat.FAULT)
        self.assertEqual([round(s.lon, 6) for s in subfaults], [-76.3, -76.3, -76.0])

    def test_wrap_longitude(self):
        np.testing.assert_allclose(
            wrap_longitude([0.0, 180.0, 359.0, -181.0]), [0.0, -180.0, -1.0, 179.0]
        )

    def test_fault_format_keeps_the_columns(self):
        row = [-76.3, -14.75, 2.3, 45e3, 40e3, 325, 18, 63, 14e3]
        (s,) = subfaults_from_inversion(np.array(row), InversionFormat.FAULT)
        self.assertEqual(
            (s.lon, s.lat, s.slip, s.length, s.width, s.strike, s.dip, s.rake, s.depth),
            tuple(row),
        )

    def test_kanamori_and_fuji_differ_in_units(self):
        fuji = [45e3, 45e3, 14e3, 325, 18, 63, 2.3, -14.75, -76.3]
        kanamori = [45, 45, 14, 325, 18, 63, 2.3, -14.75, -76.3]
        (a,) = subfaults_from_inversion(np.array(fuji), InversionFormat.FUJI)
        (b,) = subfaults_from_inversion(np.array(kanamori), InversionFormat.KANAMORI)
        self.assertEqual(a, b)

    def test_pulido_and_caltech(self):
        (p,) = subfaults_from_inversion(
            np.array([-76.3, -14.75, 230.0, 63, 325, 18, 14.0]), InversionFormat.PULIDO
        )
        self.assertEqual((p.length, p.width, p.slip, p.depth), (20e3, 20e3, 2.3, 14e3))
        (c,) = subfaults_from_inversion(
            np.array([-14.75, -76.3, 14.0, 230.0, 63, 325, 18]),
            InversionFormat.CALTECH,
            length_km=45,
            width_km=30,
        )
        self.assertEqual(
            (c.lon, c.lat, c.length, c.width, c.slip, c.depth),
            (-76.3, -14.75, 45e3, 30e3, 2.3, 14e3),
        )
        with self.assertRaises(ValueError):
            subfaults_from_inversion(np.zeros(7), InversionFormat.CALTECH)


if __name__ == "__main__":
    unittest.main()
