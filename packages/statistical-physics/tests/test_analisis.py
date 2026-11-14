import unittest

import numpy as np
from dinamica_molecular import analisis, animacion


class Difusion(unittest.TestCase):
    def test_msd_de_movimiento_balistico(self):
        rng = np.random.default_rng(0)
        vel = rng.normal(size=(30, 3))
        tiempo = np.arange(21) * 0.1
        cuadros = tiempo[:, None, None] * vel[None]
        msd = analisis.desplazamiento_cuadratico_medio(cuadros)
        np.testing.assert_allclose(
            msd, tiempo[:11] ** 2 * np.sum(vel**2, axis=1).mean()
        )

    def test_difusion_recupera_la_pendiente_de_einstein(self):
        tiempo = np.linspace(0, 10, 101)
        d, ordenada = analisis.coeficiente_difusion(
            tiempo, 6 * 0.037 * tiempo + 0.2, 2, 10
        )
        self.assertAlmostEqual(d, 0.037, places=12)
        self.assertAlmostEqual(ordenada, 0.2, places=10)

    def test_caminata_aleatoria_da_el_d_esperado(self):
        # Pasos gaussianos de varianza 2 D dt por eje dan MSD = 6 D t.
        rng = np.random.default_rng(3)
        d_real, dt = 0.05, 0.1
        pasos = rng.normal(scale=np.sqrt(2 * d_real * dt), size=(400, 200, 3))
        cuadros = np.cumsum(pasos, axis=0)
        msd = analisis.desplazamiento_cuadratico_medio(cuadros)
        tiempo = dt * np.arange(msd.size)
        d, _ = analisis.coeficiente_difusion(tiempo, msd, 1, tiempo[-1])
        self.assertAlmostEqual(d, d_real, delta=0.005)


class DistribucionRadial(unittest.TestCase):
    def test_gas_ideal_da_g_igual_a_uno(self):
        rng = np.random.default_rng(5)
        lado = 8.0
        cuadros = rng.uniform(0, lado, size=(60, 200, 3))
        r, g = analisis.g_de_r(cuadros, lado, n_bins=20)
        self.assertEqual(r.size, 20)
        np.testing.assert_allclose(g[5:], 1.0, atol=0.05)

    def test_red_cubica_simple_solo_tiene_pares_a_sus_distancias(self):
        lado = 8.0
        malla = np.arange(4) * 2.0
        pos = np.stack(np.meshgrid(malla, malla, malla, indexing="ij"), -1).reshape(
            -1, 3
        )
        r, g = analisis.g_de_r(pos[None], lado, n_bins=80)
        # Los primeros vecinos están a 2 y los segundos a 2 sqrt(2).
        # No hay pares a una distancia menor que 2.
        self.assertEqual(g[r < 1.9].sum(), 0.0)
        self.assertGreater(g[np.abs(r - 2.0) < 0.05].max(), 5.0)
        self.assertGreater(g[np.abs(r - 2 * np.sqrt(2)) < 0.05].max(), 5.0)
        self.assertEqual(g[(r > 2.1) & (r < 2.7)].sum(), 0.0)


class Animacion(unittest.TestCase):
    def test_gif_valido_y_pequeno(self):
        rng = np.random.default_rng(0)
        cuadros = rng.uniform(0, 5, size=(6, 40, 3))
        gif = animacion.gif_trayectoria(cuadros, 5.0, pixeles=120)
        self.assertEqual(gif[:6], b"GIF89a")
        self.assertLess(len(gif), 1_000_000)


if __name__ == "__main__":
    unittest.main()
