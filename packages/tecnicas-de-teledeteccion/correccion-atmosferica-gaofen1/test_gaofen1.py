import unittest
from datetime import UTC, datetime

import numpy as np
from correction import load_clip, load_inputs, load_metadata, run_correction


class GaofenCorrectionTest(unittest.TestCase):
    def test_clip_correction_matches_prepared_reference(self):
        sensor, _ = load_clip()
        inputs = load_inputs()
        corrected = run_correction(sensor, load_metadata(), inputs)
        reference = inputs["reference"]

        np.testing.assert_equal(corrected.shape, tuple(reference["shape"]))
        np.testing.assert_allclose(
            np.nanmean(corrected, axis=(1, 2)), reference["mean"], rtol=1e-6
        )
        np.testing.assert_allclose(
            np.nanmax(corrected, axis=(1, 2)), reference["max"], rtol=1e-6
        )
        np.testing.assert_allclose(
            corrected[:, 200, 200], reference["sample"], rtol=1e-6
        )

    def test_clip_reflectances_are_bounded(self):
        sensor, _ = load_clip()
        corrected = run_correction(sensor, load_metadata(), load_inputs())
        finite = corrected[np.isfinite(corrected)]

        self.assertTrue(np.all((finite >= 0) & (finite <= 1.1)))

    def test_acquisition_time_is_utc(self):
        self.assertEqual(
            load_metadata()["date"],
            datetime(2020, 9, 14, 15, 36, 52, tzinfo=UTC),
        )


if __name__ == "__main__":
    unittest.main()
