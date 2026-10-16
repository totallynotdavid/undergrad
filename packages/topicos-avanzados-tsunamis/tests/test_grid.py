import tempfile
import unittest
from pathlib import Path

import numpy as np
from tsunamis.grid import Grid, read_grid

PACKAGE = Path(__file__).resolve().parents[1]

GEBCO = """%Gridded data provided through the GEBCO Grid Demonstrator software interface
%Data source GEBCO_08 Grid version 20090202
%Minimum longitude : -80 degrees (+ve East)
%Maximum longitude : -78 degrees (+ve East)
%Minimum latitude : -15 degrees (+ve North)
%Maximum latitude : -14 degrees (+ve North)
%Data spacing : 30 seconds
%Rows*Columns : 3x4
%Units : Depths in Metres (terrestrial values are positive)
%Points start at the North-West corner working eastwards across to the North-East before restarting in the West
%Depth
 1 2 3 4
 -5 -6 -7 -8
 -10 -20 -30 -40
"""


class ReadGrid(unittest.TestCase):
    def test_gebco_ascii_has_cell_centre_coordinates_and_north_up_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.asc"
            path.write_text(GEBCO)
            grid = read_grid(path)
        np.testing.assert_allclose(grid.lon, [-79.75, -79.25, -78.75, -78.25])
        np.testing.assert_allclose(grid.lat, [-14.8333333, -14.5, -14.1666667])
        self.assertEqual(grid.elevation.shape, (3, 4))
        # The file starts in the north-west corner: its first row is the
        # northernmost, and latitude must increase along the rows of the grid.
        np.testing.assert_array_equal(grid.elevation[-1], [1, 2, 3, 4])
        np.testing.assert_array_equal(grid.elevation[0], [-10, -20, -30, -40])
        self.assertEqual(grid.source, "GEBCO_08 Grid version 20090202")

    def test_missing_header_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.asc"
            path.write_text("%Depth\n1 2\n3 4\n")
            with self.assertRaises(ValueError):
                read_grid(path)

    def test_npz_round_trip_and_crop(self):
        grid = Grid(
            np.array([-80.0, -79.0, -78.0]),
            np.array([-15.0, -14.0]),
            np.array([[-5.0, -4.0, 3.0], [-9.0, 6.0, 7.0]]),
            "test",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.npz"
            grid.save(path)
            back = read_grid(path)
        np.testing.assert_array_equal(back.elevation, grid.elevation)
        np.testing.assert_array_equal(back.lon, grid.lon)
        self.assertEqual(back.source, "test")
        cropped = back.crop(-79.5, -78.0, -14.5, -14.0)
        np.testing.assert_array_equal(cropped.lon, [-79.0, -78.0])
        np.testing.assert_array_equal(cropped.elevation, [[6.0, 7.0]])
        np.testing.assert_array_equal(back.depth, [[5.0, 4.0, 0.0], [9.0, 0.0, 0.0]])


class StudyArea(unittest.TestCase):
    def setUp(self):
        self.grid = read_grid(PACKAGE / "data" / "estudio.npz")

    def elevation_at(self, lon, lat):
        i = np.abs(self.grid.lon - lon).argmin()
        j = np.abs(self.grid.lat - lat).argmin()
        return self.grid.elevation[j, i]

    def test_study_grid_is_small(self):
        self.assertLess((PACKAGE / "data" / "estudio.npz").stat().st_size, 1_000_000)

    def test_coordinates_place_the_peruvian_coast_correctly(self):
        # Callao is on the coast, Lima inland, the Peru trench is far offshore,
        # and the Andes rise east of Lima.
        self.assertLess(abs(self.elevation_at(-77.15, -12.06)), 50.0)
        self.assertGreater(self.elevation_at(-77.0, -12.0), 100.0)
        self.assertLess(self.elevation_at(-78.5, -12.5), -3000.0)
        self.assertGreater(self.elevation_at(-75.5, -12.0), 2000.0)

    def test_spacing_is_thirty_arc_seconds(self):
        np.testing.assert_allclose(np.diff(self.grid.lon), 1 / 120, atol=1e-9)
        np.testing.assert_allclose(np.diff(self.grid.lat), 1 / 120, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
