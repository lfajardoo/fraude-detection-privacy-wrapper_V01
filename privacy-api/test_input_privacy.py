"""
Tests unitarios para InputPrivacyLayer.

Cubre los cinco requisitos contractuales:
  1. Con epsilon_in=0, la salida es idéntica a la entrada.
  2. Con epsilon_in pequeño (0.1), la desviación media en V1 es mayor que con
     epsilon_in grande (10.0) sobre 1000 aplicaciones con seed fijo.
  3. Time nunca cambia, sin importar epsilon_in.
  4. Todos los valores perturbados quedan dentro de FEATURE_BOUNDS.
  5. El dict original no se muta.
  6. (nuevo) El reparto de presupuesto es correcto: epsilon_per_feature = epsilon_in / n.
  7. (nuevo) apply_with_detail devuelve una entrada por feature perturbada, nunca Time.
"""

import sys
import os
import unittest

# Permite ejecutar desde la raíz del proyecto: python -m pytest privacy-api/
sys.path.insert(0, os.path.dirname(__file__))

from input_privacy_layer import InputPrivacyLayer, FEATURE_BOUNDS


# ---------------------------------------------------------------------------
# Fixture: transacción con todas las features relevantes + Time
# ---------------------------------------------------------------------------
_SAMPLE_TRANSACTION = {
    "Time": 12345.0,
    "V1": 1.0,  "V2": -2.0, "V3": 0.5,  "V4": 3.0,
    "V5": -1.0, "V6": 0.0,  "V7": 2.0,  "V8": -0.5,
    "V9": 1.5,  "V10": -3.0,"V11": 0.3, "V12": -1.2,
    "V13": 2.5, "V14": -0.8,
    "Amount": 149.62,
}


class TestInputPrivacyLayerContract(unittest.TestCase):

    # ------------------------------------------------------------------
    # Test 1: epsilon_in = 0 → sin ruido, output idéntico al input
    # ------------------------------------------------------------------
    def test_zero_epsilon_returns_identical_features(self):
        layer = InputPrivacyLayer(seed=42)
        result = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=0.0)
        self.assertEqual(result, _SAMPLE_TRANSACTION)

    def test_negative_epsilon_returns_identical_features(self):
        layer = InputPrivacyLayer(seed=42)
        result = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=-1.0)
        self.assertEqual(result, _SAMPLE_TRANSACTION)

    # ------------------------------------------------------------------
    # Test 2: mayor ruido con epsilon pequeño que con epsilon grande
    # ------------------------------------------------------------------
    def test_smaller_epsilon_produces_larger_deviation_in_v1(self):
        n = 1000
        original_v1 = _SAMPLE_TRANSACTION["V1"]

        layer_small = InputPrivacyLayer(mechanism="laplace", seed=42)
        layer_large = InputPrivacyLayer(mechanism="laplace", seed=42)

        dev_small = sum(
            abs(layer_small.apply(_SAMPLE_TRANSACTION, epsilon_in=0.1)["V1"] - original_v1)
            for _ in range(n)
        ) / n

        dev_large = sum(
            abs(layer_large.apply(_SAMPLE_TRANSACTION, epsilon_in=10.0)["V1"] - original_v1)
            for _ in range(n)
        ) / n

        self.assertGreater(
            dev_small, dev_large,
            msg=(
                f"Se esperaba mayor desviación con epsilon=0.1 ({dev_small:.4f}) "
                f"que con epsilon=10.0 ({dev_large:.4f})"
            ),
        )

    # ------------------------------------------------------------------
    # Test 3: Time nunca cambia
    # ------------------------------------------------------------------
    def test_time_is_never_modified(self):
        original_time = _SAMPLE_TRANSACTION["Time"]
        layer = InputPrivacyLayer(seed=0)
        for epsilon in (0.0, 0.1, 1.0, 10.0, 100.0):
            result = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=epsilon)
            self.assertEqual(
                result["Time"], original_time,
                msg=f"Time cambió con epsilon_in={epsilon}",
            )

    def test_time_absent_key_stays_absent(self):
        """Si Time no está en el input, tampoco debe aparecer en el output."""
        no_time = {k: v for k, v in _SAMPLE_TRANSACTION.items() if k != "Time"}
        layer = InputPrivacyLayer(seed=1)
        result = layer.apply(no_time, epsilon_in=1.0)
        self.assertNotIn("Time", result)

    # ------------------------------------------------------------------
    # Test 4: todos los valores perturbados dentro de FEATURE_BOUNDS
    # ------------------------------------------------------------------
    def test_perturbed_values_within_feature_bounds(self):
        layer = InputPrivacyLayer(seed=7)
        for _ in range(200):
            result = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=0.01)
            for feature, (low, high) in FEATURE_BOUNDS.items():
                if feature in result:
                    self.assertGreaterEqual(
                        result[feature], low,
                        msg=f"{feature}={result[feature]} está por debajo de {low}",
                    )
                    self.assertLessEqual(
                        result[feature], high,
                        msg=f"{feature}={result[feature]} supera {high}",
                    )

    def test_bounds_hold_for_gaussian_mechanism(self):
        layer = InputPrivacyLayer(mechanism="gaussian", seed=99)
        for _ in range(200):
            result = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=0.5)
            for feature, (low, high) in FEATURE_BOUNDS.items():
                if feature in result:
                    self.assertGreaterEqual(result[feature], low)
                    self.assertLessEqual(result[feature], high)

    # ------------------------------------------------------------------
    # Test 5: el dict original no se muta
    # ------------------------------------------------------------------
    def test_original_dict_is_not_mutated(self):
        original_copy = dict(_SAMPLE_TRANSACTION)
        layer = InputPrivacyLayer(seed=3)
        _ = layer.apply(_SAMPLE_TRANSACTION, epsilon_in=1.0)
        self.assertEqual(
            _SAMPLE_TRANSACTION, original_copy,
            msg="apply() mutó el dict original",
        )

    # ------------------------------------------------------------------
    # Extras: constructor y get_info
    # ------------------------------------------------------------------
    def test_invalid_mechanism_raises(self):
        with self.assertRaises(ValueError):
            InputPrivacyLayer(mechanism="exponential")

    def test_get_info_fields(self):
        layer = InputPrivacyLayer(seed=0)
        info = layer.get_info()
        self.assertEqual(info["mechanism"], "laplace")
        self.assertEqual(info["n_features_protected"], len(FEATURE_BOUNDS))
        self.assertIn("V1", info["feature_bounds"])
        self.assertNotIn("Time", info["feature_bounds"])

    def test_extra_keys_are_preserved_unchanged(self):
        """Claves no listadas en FEATURE_BOUNDS pasan intactas."""
        features = dict(_SAMPLE_TRANSACTION)
        features["extra_field"] = 999.9
        layer = InputPrivacyLayer(seed=5)
        result = layer.apply(features, epsilon_in=1.0)
        self.assertEqual(result["extra_field"], 999.9)

    # ------------------------------------------------------------------
    # Test 6: reparto de presupuesto secuencial correcto
    # ------------------------------------------------------------------
    def test_budget_split_epsilon_per_feature(self):
        """Con epsilon_in = n_features, epsilon_used por feature == 1.0."""
        n_perturbable = len([f for f in FEATURE_BOUNDS if f in _SAMPLE_TRANSACTION])
        epsilon_in = float(n_perturbable)  # 15.0 → epsilon_per_feature = 1.0

        layer = InputPrivacyLayer(seed=42)
        _, detail = layer.apply_with_detail(_SAMPLE_TRANSACTION, epsilon_in=epsilon_in)

        self.assertEqual(len(detail), n_perturbable)
        for entry in detail:
            self.assertAlmostEqual(entry["epsilon_used"], 1.0, places=10)

        total_epsilon = sum(e["epsilon_used"] for e in detail)
        self.assertAlmostEqual(total_epsilon, epsilon_in, places=10)

    def test_budget_split_sum_equals_epsilon_in(self):
        """La suma de epsilon_used siempre iguala epsilon_in."""
        layer = InputPrivacyLayer(seed=7)
        for eps in (0.5, 1.0, 3.0, 15.0):
            _, detail = layer.apply_with_detail(_SAMPLE_TRANSACTION, epsilon_in=eps)
            if detail:
                self.assertAlmostEqual(
                    sum(e["epsilon_used"] for e in detail), eps, places=10,
                    msg=f"Suma de epsilon_used no iguala epsilon_in={eps}",
                )

    def test_budget_split_zero_epsilon_returns_empty_detail(self):
        layer = InputPrivacyLayer(seed=0)
        _, detail = layer.apply_with_detail(_SAMPLE_TRANSACTION, epsilon_in=0.0)
        self.assertEqual(detail, [])

    # ------------------------------------------------------------------
    # Test 7: apply_with_detail estructura y cobertura de features
    # ------------------------------------------------------------------
    def test_apply_with_detail_returns_entry_per_perturbable_feature(self):
        """Una entrada por cada feature en FEATURE_BOUNDS presente en input."""
        layer = InputPrivacyLayer(seed=0)
        _, detail = layer.apply_with_detail(_SAMPLE_TRANSACTION, epsilon_in=1.0)

        detail_features = [d["feature"] for d in detail]
        self.assertNotIn("Time", detail_features)
        for f in FEATURE_BOUNDS:
            if f in _SAMPLE_TRANSACTION:
                self.assertIn(f, detail_features,
                              msg=f"Feature {f} ausente en el detalle")

    def test_apply_with_detail_entry_keys(self):
        """Cada entrada de detalle tiene los cinco campos requeridos."""
        layer = InputPrivacyLayer(seed=1)
        _, detail = layer.apply_with_detail(_SAMPLE_TRANSACTION, epsilon_in=1.0)
        required_keys = {"feature", "original", "clipped", "perturbed", "epsilon_used"}
        for entry in detail:
            self.assertEqual(set(entry.keys()), required_keys)


if __name__ == "__main__":
    unittest.main()
