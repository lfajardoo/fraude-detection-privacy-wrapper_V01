import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from attacks.api_client import AttackClient
from attacks.inversion_attack import denoise, invert_feature


class MockClient(AttackClient):
    def __init__(self, a_true=1.2, b_true=-0.4, sigma_in=0.0, sigma_out=0.01, seed=7):
        self.a_true = a_true
        self.b_true = b_true
        self.sigma_in = sigma_in
        self.rng = np.random.default_rng(seed)
        self.query_count = 0
        self.sigma_out = sigma_out

    def query(self, features: dict) -> float:
        self.query_count += 1
        value = features["secret"] + self.rng.normal(0.0, self.sigma_in)
        probability = 1.0 / (1.0 + math.exp(-(self.a_true * value + self.b_true)))
        return float(np.clip(probability + self.rng.normal(0.0, self.sigma_out), 0.0, 1.0))


class TestInversionAttack(unittest.TestCase):
    def test_recuperacion_exacta_sin_ruido(self):
        client = MockClient(sigma_in=0.0, sigma_out=0.001)
        target_probability = 1.0 / (1.0 + math.exp(-client.b_true))
        estimate = invert_feature(client, {}, "secret", -2.0, 2.0, 2000, target_probability)
        self.assertIsNotNone(estimate)
        self.assertLess(abs(estimate - 0.0), 0.03)

    def test_piso_crece_con_ruido_entrada(self):
        def run(sigma):
            errors = []
            for seed, real in enumerate(np.linspace(-1.5, 1.5, 12)):
                client = MockClient(sigma_in=sigma, sigma_out=0.001, seed=seed)
                target = {"secret": float(real)}
                estimate = invert_feature(client, {}, "secret", -2.0, 2.0, 150, denoise(client, target, 150))
                errors.append(estimate - real)
            return sum(abs(error) for error in errors) / len(errors)
        self.assertGreater(run(1.0), run(0.05))

    def test_denoise_cancela_salida(self):
        variances = []
        for K in (10, 100, 1000):
            means = [denoise(MockClient(sigma_in=0.0, sigma_out=0.2, seed=seed), {"secret": 0.3}, K) for seed in range(30)]
            variances.append(float(np.var(means)))
        self.assertLess(variances[1], variances[0])
        self.assertLess(variances[2], variances[1])

    def test_feature_sin_influencia(self):
        client = MockClient(a_true=0.0, sigma_in=0.0, sigma_out=0.0)
        self.assertIsNone(invert_feature(client, {}, "secret", -1.0, 1.0, 5, 0.5))


if __name__ == "__main__":
    unittest.main()