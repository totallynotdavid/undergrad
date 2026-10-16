import unittest

import numpy as np
from dinamica_molecular import lennard_jones as lj

# Cota de la deriva de energía durante la corrida corta:
# |E(t) - E(0)| / |E(0)| < 5e-4.
# La corrida esperada da cerca de 2e-4. Una integración defectuosa, como una
# con paredes reflectantes o un paso grande, supera la cota.
COTA_DERIVA = 5e-4


def liquido(n_celdas=3, densidad=0.85, temperatura=0.78, semilla=1):
    pos, lado = lj.red_fcc(n_celdas, densidad)
    vel = lj.velocidades_maxwell(len(pos), temperatura, np.random.default_rng(semilla))
    return pos, vel, lado


class EnergiaTotal(unittest.TestCase):
    def test_deriva_acotada_en_corrida_corta(self):
        pos, vel, lado = liquido()
        eq = lj.integrar(pos, vel, lado, 0.004, 400, t_objetivo=0.78)
        corrida = lj.integrar(eq.pos_final, eq.vel_final, lado, 0.004, 600)
        total = corrida.energia_total
        deriva = np.max(np.abs(total - total[0])) / abs(total[0])
        self.assertLess(deriva, COTA_DERIVA)

    def test_paso_grande_rompe_la_cota(self):
        # Garantiza que la cota de arriba distingue una integración mala.
        pos, vel, lado = liquido()
        eq = lj.integrar(pos, vel, lado, 0.004, 400, t_objetivo=0.78)
        corrida = lj.integrar(eq.pos_final, eq.vel_final, lado, 0.03, 100)
        total = corrida.energia_total
        deriva = np.max(np.abs(total - total[0])) / abs(total[0])
        self.assertGreater(deriva, COTA_DERIVA)

    def test_momento_total_se_conserva(self):
        pos, vel, lado = liquido()
        corrida = lj.integrar(pos, vel, lado, 0.004, 200)
        self.assertLess(np.abs(corrida.vel_final.sum(axis=0)).max(), 1e-10)

    def test_la_temperatura_inicial_es_la_pedida(self):
        pos, vel, lado = liquido(temperatura=1.3)
        corrida = lj.integrar(pos, vel, lado, 0.004, 1)
        self.assertAlmostEqual(corrida.temperatura[0], 1.3, places=10)

    def test_equilibrado_lleva_la_temperatura_al_objetivo(self):
        pos, vel, lado = liquido(temperatura=2.0)
        eq = lj.integrar(pos, vel, lado, 0.004, 200, t_objetivo=0.9, cada_reescala=10)
        self.assertAlmostEqual(eq.temperatura[-1], 0.9, delta=0.02)


class FuerzasPeriodicas(unittest.TestCase):
    def test_energia_de_un_par_es_la_total_sin_dividir_por_n(self):
        r = 2 ** (1 / 6)
        pos = np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]])
        _, pe = lj.fuerzas(pos, 10.0, lj.CORTE)
        rc = lj.CORTE
        fuerza_c = 24 * (2 * rc**-12 - rc**-6) / rc
        esperado = 4 * (r**-12 - r**-6) - 4 * (rc**-12 - rc**-6) + (r - rc) * fuerza_c
        self.assertAlmostEqual(pe, esperado, places=12)

    def test_un_par_interactua_a_traves_de_la_frontera(self):
        lado = 8.0
        pos = np.array([[0.4, 4.0, 4.0], [lado - 0.7, 4.0, 4.0]])  # separados 1.1
        f, _ = lj.fuerzas(pos, lado, lj.CORTE)
        # La partícula 0 es repelida hacia +x, lejos de su vecina a través de la frontera.
        self.assertGreater(f[0, 0], 0)
        self.assertLess(f[1, 0], 0)
        np.testing.assert_allclose(f[0], -f[1])

    def test_trasladar_una_particula_un_lado_no_cambia_nada(self):
        pos, _, lado = liquido()
        f0, pe0 = lj.fuerzas(pos, lado, lj.CORTE)
        movido = pos.copy()
        movido[5] += np.array([lado, -2 * lado, 0.0])
        f1, pe1 = lj.fuerzas(movido, lado, lj.CORTE)
        self.assertAlmostEqual(pe0, pe1, places=9)
        np.testing.assert_allclose(f0, f1, atol=1e-9)

    def test_energia_cinetica_y_potencial_son_totales(self):
        pos, vel, lado = liquido()
        corrida = lj.integrar(pos, vel, lado, 0.004, 1)
        self.assertAlmostEqual(
            corrida.energia_cinetica[0], 0.5 * np.sum(vel**2), places=10
        )

    def test_corte_mayor_que_media_caja_se_rechaza(self):
        pos, vel, lado = liquido(n_celdas=2)
        with self.assertRaises(ValueError):
            lj.integrar(pos, vel, lado, 0.004, 1, corte=lado)


class ComparacionConVerlet(unittest.TestCase):
    def test_energia_potencial_de_la_tabla_i(self):
        # Verlet (1967), tabla I: rho = 0.85, T = 0.782 da U = -6.04 por partícula,
        # con corte 2.5 y la corrección de cola de largo alcance.
        pos, vel, lado = liquido(n_celdas=4, temperatura=0.78)
        eq = lj.integrar(pos, vel, lado, 0.004, 1000, t_objetivo=0.78, cada=100)
        corrida = lj.integrar(eq.pos_final, eq.vel_final, lado, 0.004, 1000, cada=100)
        n = pos.shape[0]
        u = (
            np.mean(
                [lj.energia_truncada(p, lado, lj.CORTE) for p in corrida.posiciones]
            )
            / n
        )
        u += lj.correccion_cola(0.85)
        self.assertAlmostEqual(u, -6.04, delta=0.1)


if __name__ == "__main__":
    unittest.main()
