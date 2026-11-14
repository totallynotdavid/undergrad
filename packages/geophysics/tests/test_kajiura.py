import unittest

import numpy as np
from tsunamis.kajiura import kajiura_filter


class Kajiura(unittest.TestCase):
    def test_a_sinusoid_is_damped_by_one_over_cosh_kh(self):
        n, dx = 256, 1000.0
        x = np.arange(n) * dx
        centre = slice(n // 2 - 16, n // 2 + 16)
        for wavelength, depth in ((16e3, 2000.0), (64e3, 2000.0), (32e3, 1234.0)):
            k = 2 * np.pi / wavelength
            uplift = np.tile(np.cos(k * x), (n, 1))
            out = kajiura_filter(uplift, np.full((n, n), depth), dx, dx, step=100.0)
            ratio = (
                np.abs(out[n // 2, centre]).max() / np.abs(uplift[n // 2, centre]).max()
            )
            self.assertAlmostEqual(ratio, 1 / np.cosh(k * depth), places=5)

    def test_land_and_zero_depth_are_unfiltered(self):
        rng = np.random.default_rng(1)
        uplift = rng.normal(size=(40, 50))
        out = kajiura_filter(uplift, np.full((40, 50), -100.0), 500.0, 500.0)
        np.testing.assert_allclose(out, uplift, atol=1e-12)

    def test_a_localised_uplift_keeps_its_volume_and_loses_its_peak(self):
        n, dx = 160, 1000.0
        x = (np.arange(n) - n / 2) * dx
        xx, yy = np.meshgrid(x, x)
        uplift = np.exp(-(xx**2 + yy**2) / (2 * 6e3**2))
        out = kajiura_filter(uplift, np.full((n, n), 4000.0), dx, dx)
        self.assertLess(out.max(), 0.95 * uplift.max())
        self.assertAlmostEqual(out.sum() / uplift.sum(), 1.0, places=3)

    def test_filter_follows_the_local_depth(self):
        n, dx = 128, 1000.0
        x = np.arange(n) * dx
        uplift = np.tile(np.cos(2 * np.pi * x / 16e3), (n, 1))
        depth = np.zeros((n, n))
        depth[:, n // 2 :] = 3000.0
        out = kajiura_filter(uplift, depth, dx, dx, step=100.0)
        shallow = np.abs(out[n // 2, 24:40]).max()
        deep = np.abs(out[n // 2, 88:104]).max()
        self.assertGreater(shallow, 0.95)
        self.assertAlmostEqual(deep, 1 / np.cosh(2 * np.pi / 16e3 * 3000.0), delta=0.05)


if __name__ == "__main__":
    unittest.main()
