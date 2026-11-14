import unittest

import numpy as np
from dinamica_molecular import tight_binding as tb

P = tb.POTASIO
ARGS = (P.a, P.xi, P.r0, P.p, P.q)


def cumulo_irregular():
    rng = np.random.default_rng(2)
    return tb.simplex_regular(4, 4.4) + rng.normal(scale=0.15, size=(4, 3))


class Fuerzas(unittest.TestCase):
    def test_fuerza_es_menos_el_gradiente_de_la_energia(self):
        pos = cumulo_irregular()
        _, f = tb.energia_y_fuerzas(pos, *ARGS)
        h = 1e-5
        for i in range(4):
            for k in range(3):
                mas, menos = pos.copy(), pos.copy()
                mas[i, k] += h
                menos[i, k] -= h
                e_mas, _ = tb.energia_y_fuerzas(mas, *ARGS)
                e_menos, _ = tb.energia_y_fuerzas(menos, *ARGS)
                self.assertAlmostEqual(f[i, k], -(e_mas - e_menos) / (2 * h), places=7)

    def test_fuerza_total_nula(self):
        _, f = tb.energia_y_fuerzas(cumulo_irregular(), *ARGS)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)


class Energia(unittest.TestCase):
    def test_dimero_tiene_su_minimo_en_la_distancia_analitica(self):
        # Para N = 2 el mínimo de A e^{-p x} - xi e^{-q x} cumple
        # e^{-(p-q) x} = q xi / (p A), con x = r / r0 - 1.
        x = -np.log(P.q * P.xi / (P.p * P.a)) / (P.p - P.q)
        esperado = P.r0 * (1 + x)
        d = np.linspace(3.5, 5.0, 3001)
        e = [tb.energia_por_atomo(tb.simplex_regular(2, r)) for r in d]
        self.assertAlmostEqual(d[np.argmin(e)], esperado, delta=1e-3)

    def test_mas_vecinos_ligan_mas_por_atomo(self):
        # La raíz del segundo momento hace que el enlace crezca con la coordinación.
        minimos = []
        for n in (2, 3, 4):
            d = np.linspace(3.5, 5.5, 400)
            minimos.append(
                min(tb.energia_por_atomo(tb.simplex_regular(n, r)) for r in d)
            )
        self.assertGreater(minimos[0], minimos[1])
        self.assertGreater(minimos[1], minimos[2])


class Dinamica(unittest.TestCase):
    def test_energia_total_se_conserva_en_vibracion(self):
        rng = np.random.default_rng(4)
        pos = tb.simplex_regular(4, 4.3) + rng.normal(scale=0.1, size=(4, 3))
        vel = rng.normal(scale=0.01, size=(4, 3))
        vel -= vel.mean(axis=0)
        ke, pe = tb.integrar(pos, vel, dt=0.5, n_pasos=2000)
        total = ke + pe
        self.assertLess(np.max(np.abs(total - total[0])), 1e-4)  # eV
        self.assertGreater(np.ptp(pe), 10 * 1e-4)  # el cúmulo sí vibra


if __name__ == "__main__":
    unittest.main()
