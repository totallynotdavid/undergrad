import unittest
from pathlib import Path

import numpy as np
from tsunamis import legacy, mansinha, okada
from tsunamis.faults import fault_plane, local_xy, to_local

PACKAGE = Path(__file__).resolve().parents[1]


class OkadaAgainstFortran(unittest.TestCase):
    def test_reproduces_committed_deform_a(self):
        # deform_a.grd is written with F7.3, so it carries a rounding error of
        # 5e-4 m; the Fortran also runs in single precision.
        reference = np.loadtxt(PACKAGE / "lab2" / "deform_a.grd")
        pfalla = legacy.read_pfalla(PACKAGE / "lab2" / "pfalla.inp")
        ours = legacy.run_def_oka(pfalla)
        self.assertEqual(ours.shape, reference.shape)
        self.assertLess(np.abs(ours - reference).max(), 1.0e-3)
        self.assertAlmostEqual(ours.max(), 1.28, places=2)
        self.assertAlmostEqual(ours.min(), -0.41, places=2)


class OkadaAgainstMansinhaSmylie(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-250e3, 150e3, 161) + 13.7
        y = np.linspace(-100e3, 250e3, 141) + 7.3
        self.east, self.north = np.meshgrid(x, y)

    def test_same_field_for_a_poisson_solid(self):
        # Mansinha and Smylie assume mu = lambda, that is mu / (lambda + mu) = 1/2.
        for dip in (10.0, 45.0, 80.0):
            for rake in (0.0, 90.0, -90.0, 135.0, 200.0):
                for strike in (0.0, 310.0):
                    fault = okada.Subfault(
                        0, 0, 100e3, 40e3, 3e3, strike, dip, rake, 2.0
                    )
                    a = okada.vertical_displacement(
                        self.east, self.north, [fault], ratio=0.5
                    )
                    b = mansinha.vertical_displacement(self.east, self.north, fault)
                    self.assertLess(np.abs(a - b).max(), 1.0e-6, (dip, rake, strike))

    def test_crustal_model_changes_the_field_little(self):
        fault = okada.Subfault(0, 0, 160e3, 70e3, 5e3, 310.0, 18.0, 63.0, 3.0)
        a = okada.vertical_displacement(self.east, self.north, [fault])
        b = mansinha.vertical_displacement(self.east, self.north, fault)
        self.assertLess(np.abs(a - b).max(), 0.01 * np.abs(a).max())


class OkadaBehaviour(unittest.TestCase):
    def test_thrust_lifts_the_seafloor_above_the_fault_and_drops_it_behind(self):
        fault = okada.Subfault(0, 0, 100e3, 50e3, 5e3, 0.0, 20.0, 90.0, 2.0)
        x = np.linspace(-50e3, 200e3, 251)
        # Strike is north, so the fault dips east: the free edge is on the west.
        uz = okada.vertical_displacement(x, np.full_like(x, 50e3), [fault])
        self.assertGreater(uz.max(), 0.5)
        self.assertLess(uz.min(), -0.05)
        self.assertGreater(x[uz.argmax()], 0.0)
        self.assertLess(x[uz.argmax()], x[uz.argmin()])

    def test_displacement_is_linear_in_slip(self):
        east, north = np.meshgrid(
            np.linspace(-1e5, 1e5, 21), np.linspace(-1e5, 1e5, 21)
        )
        base = okada.Subfault(0, 0, 80e3, 40e3, 4e3, 45.0, 30.0, 70.0, 1.0)
        double = okada.Subfault(0, 0, 80e3, 40e3, 4e3, 45.0, 30.0, 70.0, 2.0)
        one = okada.vertical_displacement(east, north, [base])
        two = okada.vertical_displacement(east, north, [double])
        np.testing.assert_allclose(two, 2.0 * one, atol=1e-12)

    def test_no_singularity_at_strike_zero_and_360(self):
        # def_oka.f nudges these two strikes by 0.001 degrees to avoid a tan(90).
        east, north = np.meshgrid(
            np.linspace(-1e5, 1e5, 21), np.linspace(-1e5, 1e5, 21)
        )
        fields = []
        for strike in (0.0, 360.0, 0.001):
            fault = okada.Subfault(0, 0, 80e3, 40e3, 4e3, strike, 30.0, 70.0, 1.0)
            fields.append(okada.vertical_displacement(east, north, [fault]))
        self.assertTrue(np.isfinite(fields[0]).all())
        np.testing.assert_allclose(fields[0], fields[1], atol=1e-9)
        np.testing.assert_allclose(fields[0], fields[2], atol=1e-3)

    def test_points_on_the_fault_edges_are_finite(self):
        # xi = 0 and q = 0 lines are where the kernels switch to limit forms.
        fault = okada.Subfault(0, 0, 100e3, 50e3, 2e3, 90.0, 30.0, 90.0, 1.0)
        east = np.array([0.0, 50e3, 100e3, 0.0])
        north = np.array([0.0, 0.0, 0.0, -50e3])
        self.assertTrue(
            np.isfinite(okada.vertical_displacement(east, north, [fault])).all()
        )

    def test_eight_subfaults_add_up_to_the_whole_fault(self):
        whole = fault_plane(
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
        tiled = fault_plane(
            -77.0,
            -14.0,
            length=160e3,
            width=70e3,
            depth=5e3,
            strike=310.0,
            dip=18.0,
            rake=63.0,
            slip=3.0,
            along_dip=2,
            along_strike=4,
        )
        self.assertEqual(len(tiled.subfaults), 8)
        lon, lat = np.meshgrid(
            np.linspace(-79.5, -75.5, 120), np.linspace(-15.5, -11.5, 120)
        )
        east, north = local_xy(lon, lat, -77.0, -14.0)
        one = okada.vertical_displacement(
            east, north, to_local(whole.subfaults, -77.0, -14.0)
        )
        eight = okada.vertical_displacement(
            east, north, to_local(tiled.subfaults, -77.0, -14.0)
        )
        self.assertLess(np.abs(one - eight).max(), 2.0e-3 * np.abs(one).max())


if __name__ == "__main__":
    unittest.main()
