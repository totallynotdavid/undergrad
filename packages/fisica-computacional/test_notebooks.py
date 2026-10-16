"""Run the numerical-methods notebooks as marimo runs them and check their numbers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

METODOS = Path(__file__).parent / "metodos-numericos"


def run_notebook(relative: str) -> dict:
    path = METODOS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defs = module.app.run()
    return defs


class DerivacionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("derivacion-numerica/derivacion_numerica.py")

    def test_truncation_orders(self):
        p = self.defs["pendientes"]
        self.assertAlmostEqual(p["adelante"], 1.0, delta=0.05)
        self.assertAlmostEqual(p["centrada"], 2.0, delta=0.05)
        self.assertAlmostEqual(p["3 puntos adelante"], 2.0, delta=0.05)

    def test_measured_minimum_error_matches_the_roundoff_model(self):
        d = self.defs
        for nombre in ("adelante", "centrada"):
            razon = d["e_minimo"][nombre] / d["e_modelo"][nombre]
            self.assertTrue(0.01 < razon < 10, f"{nombre}: {razon}")
            desvio = np.log10(d["h_optimo"][nombre] / d["h_modelo"][nombre])
            self.assertLess(abs(desvio), 2)

    def test_richardson_raises_the_order(self):
        p = self.defs["pendientes_rich"]
        self.assertAlmostEqual(p["centrada"], 2.0, delta=0.1)
        self.assertAlmostEqual(p["1 nivel (orden 4)"], 4.0, delta=0.1)
        self.assertAlmostEqual(p["2 niveles (orden 6)"], 6.0, delta=0.15)

    def test_series_of_differences_with_t_two_is_the_three_point_formula(self):
        d = self.defs
        y = d["f"](1.0 + 0.02 * np.arange(3))
        serie = d["derivada_serie_adelante"](y, 0.02, 2)
        self.assertAlmostEqual(serie, d["adelante_3"](d["f"], 1.0, 0.02), places=12)


class RaicesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("solucion-ecuaciones-no-lineales/raices.py")

    def test_observed_orders(self):
        ordenes = self.defs["ordenes"]
        self.assertAlmostEqual(np.median(ordenes["bisección (cota)"]), 1.0, delta=0.01)
        self.assertAlmostEqual(np.median(ordenes["falsa posición"]), 1.0, delta=0.05)
        self.assertAlmostEqual(np.median(ordenes["Newton ($x_0=-6$)"]), 2.0, delta=0.1)

    def test_bisection_bound_halves_every_step(self):
        ancho = self.defs["anchos_biseccion"][:30]
        np.testing.assert_allclose(ancho[1:] / ancho[:-1], 0.5)

    def test_bisection_handles_a_decreasing_function(self):
        medios, _ = self.defs["biseccion"](lambda x: 2.0 - x, 0.0, 5.0, 40)
        self.assertAlmostEqual(medios[-1], 2.0, places=10)

    def test_newton_at_a_double_root_is_linear(self):
        e = self.defs["error_doble"]["Newton clásico"]
        np.testing.assert_allclose(e[1:] / e[:-1], 0.5, atol=0.05)
        self.assertAlmostEqual(e[-1] / e[-2], 0.5, delta=0.002)

    def test_modified_newton_recovers_quadratic_convergence(self):
        e = self.defs["error_doble"]["Newton con $m=2$"]
        self.assertLess(e[3], 1e-12)


class InterpolacionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("interpolacion/interpolacion.py")

    def test_lagrange_reproduces_a_polynomial_exactly(self):
        d = self.defs
        x = np.array([-1.0, 0.0, 2.0, 3.0])

        def cubica(t):
            return 2 * t**3 - t + 5

        t = np.linspace(-2, 4, 7)
        np.testing.assert_allclose(
            d["lagrange"](x, cubica(x), t), cubica(t), atol=1e-10
        )
        np.testing.assert_allclose(d["newton"](x, cubica(x), t), cubica(t), atol=1e-10)

    def test_lagrange_and_newton_are_the_same_polynomial(self):
        d = self.defs
        t = np.linspace(0.1, 1.3, 50)
        np.testing.assert_allclose(
            d["lagrange"](d["X_N"], d["Y_N"], t),
            d["newton"](d["X_N"], d["Y_N"], t),
            atol=1e-12,
        )

    def test_equispaced_forms_match_divided_differences(self):
        d = self.defs
        x = 150.0 + 10.0 * np.arange(7)
        y = np.array([35.5, 37.8, 43.6, 45.7, 47.3, 50.1, 51.2])
        self.assertAlmostEqual(
            d["newton_adelante"](x[0], 10.0, y, 162.0),
            float(d["newton"](x, y, np.array(162.0))),
            places=9,
        )
        self.assertAlmostEqual(
            d["newton_atras"](x[-1], 10.0, y, 162.0),
            float(d["newton"](x, y, np.array(162.0))),
            places=9,
        )

    def test_runge_phenomenon(self):
        e = self.defs["errores_runge"]
        self.assertGreater(e["equiespaciados"][-1], 100 * e["equiespaciados"][2])
        self.assertLess(e["Chebyshev"][-1], 0.01 * e["Chebyshev"][0])


class PuntoFlotanteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defs = run_notebook("errores-numericos/punto_flotante.py")

    def test_epsilon_loop_finds_the_machine_epsilon(self):
        for tipo in (np.float32, np.float64):
            ultimo, previo = self.defs["epsilon_por_biseccion"](tipo)
            self.assertEqual(previo, np.finfo(tipo).eps)
            self.assertEqual(ultimo, np.finfo(tipo).eps / 2)

    def test_binary_conversions_round_trip(self):
        d = self.defs
        for n in (0, 1, 2, 255, 12345, 2**40 + 7):
            self.assertEqual(d["a_binario"](n), bin(n)[2:])
            self.assertEqual(d["de_binario"](d["a_binario"](n)), n)

    def test_single_precision_literal_accumulates_in_a_double(self):
        r = self.defs["resultados_suma"]
        self.assertAlmostEqual(r["2. `float64` + `float32`"], 1.0999999974737875)
        self.assertLess(abs(r["1. `float64` + `float64`"] - 1.1), 1e-11)

    def test_blocked_single_precision_sum_beats_the_serial_one(self):
        r = self.defs["resultados_suma"]
        serie = abs(r["3. todo `float32`"] - 1.1)
        bloques = abs(r["4. `float32` por bloques"] - 1.1)
        self.assertLess(bloques, serie / 10)


if __name__ == "__main__":
    unittest.main()
