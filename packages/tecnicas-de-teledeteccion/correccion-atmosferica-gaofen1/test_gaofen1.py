import unittest

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


if __name__ == "__main__":
    unittest.main()
