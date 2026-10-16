"""Run each notebook as marimo runs it and check the numbers it computes.

Run from the repository root:
uv run --package mecanica-estadistica python -m unittest discover -s packages/mecanica-estadistica
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run_notebook(name: str) -> dict:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defs = module.app.run()
    return defs


class HypersphereTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("calcular_volumen_esfera_n_dimensiones")

    def test_monte_carlo_agrees_with_the_exact_formula(self):
        d = self.defs
        measured = d["aciertos"] > 0
        z = np.abs(d["estimado"] - d["exacto"])[measured] / d["sigma"][measured]
        self.assertLess(z.max(), 5.0)

    def test_exact_volume_peaks_at_five_dimensions(self):
        self.assertEqual(self.defs["n_pico"], 5)
        self.assertAlmostEqual(self.defs["exacto"][4], 8 * np.pi**2 / 15)

    def test_acceptance_rate_collapses_and_runs_out_of_hits(self):
        d = self.defs
        self.assertTrue(np.all(np.diff(d["tasa"][4:]) < 0))
        self.assertEqual(d["aciertos"][-1], 0)
        self.assertEqual(d["estimado"][-1], 0.0)


class IsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("modelo_de_ising")

    def test_running_energy_matches_the_final_lattice(self):
        e, m, spins = self.defs["simular"](6, 1 / 4.0, 50, 3)
        s = spins.astype(np.int64)
        energy = -sum((s * np.roll(s, 1, axis=a)).sum() for a in range(3))
        self.assertAlmostEqual(e[-1], energy / 6**3)
        self.assertAlmostEqual(m[-1], s.sum() / 6**3)

    def test_binder_cumulant_goes_from_ordered_to_disordered_limit(self):
        temperatures = self.defs["TEMPERATURAS"]
        for u4 in (o["U4"].mean(axis=1) for o in self.defs["observables"].values()):
            self.assertAlmostEqual(u4[temperatures.argmin()], 2 / 3, delta=0.01)
            self.assertLess(u4[temperatures.argmax()], 0.3)

    def test_binder_crossings_bracket_the_critical_temperature_from_below(self):
        tc = 1 / self.defs["BETA_HASENBUSCH"]
        for crossing, _ in self.defs["cruces"].values():
            self.assertGreater(crossing, tc - 0.3)
            self.assertLess(crossing, tc)


if __name__ == "__main__":
    unittest.main()
