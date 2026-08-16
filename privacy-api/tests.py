"""
Tests unitarios para privacy-api
Valida que privacy_layer y client funcionan correctamente
"""

import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import ConnectionError as RequestsConnectionError
from privacy_layer import DifferentialPrivacyLayer
from client import ModelAPIClient
from config import EPSILON_FRAUD_PROBABILITY, EPSILON_CONFIDENCE_SCORE
from governor import (
    GovernanceDecision, governor,
    validate_monotone_schedule, epsilon_from_risk, MONOTONE_EPSILON_SCHEDULE,
)
from privacy_governor_simple import gobernar_politica
from metrics.noise_metrics import mae, rmse, noise_summary
from metrics.classification_metrics import (
    confusion_matrix_counts, accuracy, fnr, f1,
    informedness, markedness, mcc, classification_report,
)
from metrics.privacy_metrics import (
    utility_retention, risk_noise_correlation, privacy_utility_summary,
)


class TestSimplePrivacyGovernor(unittest.TestCase):
    """Pruebas de regresión del árbol hardcodeado del Privacy Governor."""

    def test_selects_all_policies(self):
        self.assertEqual(gobernar_politica(0.51, 0.9, 0)["politica"], "P1")
        self.assertEqual(gobernar_politica(0.2, 0.5, 0)["politica"], "P2")
        self.assertEqual(gobernar_politica(0.2, 0.9, 0)["politica"], "P3")
        self.assertEqual(gobernar_politica(0.2, 0.9, 3)["politica"], "P4")

    def test_p4_does_not_perturb_values(self):
        response = {
            "is_fraud": 1,
            "fraud_probability": 0.8,
            "confidence_score": 0.9,
        }
        result = DifferentialPrivacyLayer().apply_dp_to_response(
            response.copy(),
            GovernanceDecision("laplace", 0.0, 1e-6, False, 0.0, 0.0),
        )
        self.assertEqual(result["fraud_probability"], 0.8)
        self.assertEqual(result["confidence_score"], 0.9)


class TestPrivacyLayer(unittest.TestCase):
    """Tests para DifferentialPrivacyLayer"""

    def setUp(self):
        """Inicializar antes de cada test"""
        self.dp_laplace = DifferentialPrivacyLayer(mechanism="laplace")
        self.dp_gaussian = DifferentialPrivacyLayer(mechanism="gaussian")

    def test_laplace_initialization(self):
        """Test: Inicialización con Laplace"""
        self.assertEqual(self.dp_laplace.mechanism, "laplace")

    def test_gaussian_initialization(self):
        """Test: Inicialización con Gaussian"""
        self.assertEqual(self.dp_gaussian.mechanism, "gaussian")

    def test_invalid_mechanism(self):
        """Test: Rechazo de mecanismo inválido"""
        with self.assertRaises(ValueError):
            DifferentialPrivacyLayer(mechanism="invalid")

    def test_probability_clipping(self):
        """Test: Valores de probabilidad están en [0, 1]"""
        for _ in range(10):  # Múltiples iteraciones por aleatoriedad
            result = self.dp_laplace.apply_dp_to_probability(0.5)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_confidence_clipping(self):
        """Test: Valores de confianza están en [0, 1]"""
        for _ in range(10):
            result = self.dp_laplace.apply_dp_to_confidence(0.7)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_response_transformation(self):
        """Test: Transformación de respuesta completa"""
        response = {
            "is_fraud": 0,
            "fraud_probability": 0.2,
            "confidence_score": 0.95,
            "message": "Test"
        }
        result = self.dp_laplace.apply_dp_to_response(response.copy())

        # Verificar que fue modificado
        self.assertIn("fraud_probability", result)
        self.assertIn("confidence_score", result)

        # Verificar bounds
        self.assertGreaterEqual(result["fraud_probability"], 0.0)
        self.assertLessEqual(result["fraud_probability"], 1.0)
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)

    def test_get_info(self):
        """Test: Obtener información de configuración"""
        info = self.dp_laplace.get_info()

        self.assertIn("mechanism", info)
        self.assertIn("epsilon_fraud_probability", info)
        self.assertIn("epsilon_confidence_score", info)
        self.assertIn("delta", info)
        self.assertEqual(info["mechanism"], "laplace")


class TestModelAPIClient(unittest.TestCase):
    """Tests para ModelAPIClient"""

    def setUp(self):
        """Inicializar antes de cada test"""
        self.client = ModelAPIClient()

    @patch('client.requests.post')
    def test_predict_success(self, mock_post):
        """Test: Predicción exitosa"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "is_fraud": 0,
            "fraud_probability": 0.1,
            "confidence_score": 0.95,
            "message": "Legítimo"
        }
        mock_post.return_value = mock_response

        transaction = {"Time": 0, "V1": -1.35, "V2": -0.07}
        result = self.client.predict(transaction)

        self.assertEqual(result["is_fraud"], 0)
        self.assertEqual(result["fraud_probability"], 0.1)
        mock_post.assert_called_once()

    @patch('client.requests.post')
    def test_predict_connection_error(self, mock_post):
        """Test: Manejo de error de conexión"""
        mock_post.side_effect = RequestsConnectionError("Connection refused")

        transaction = {"Time": 0, "V1": -1.35}
        with self.assertRaises(ConnectionError):
            self.client.predict(transaction)

    @patch('client.requests.get')
    def test_health_check_success(self, mock_get):
        """Test: Health check exitoso"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "model_loaded": True
        }
        mock_get.return_value = mock_response

        result = self.client.health()
        self.assertTrue(result)

    @patch('client.requests.get')
    def test_health_check_failure(self, mock_get):
        """Test: Health check fallido"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "unhealthy",
            "model_loaded": False
        }
        mock_get.return_value = mock_response

        result = self.client.health()
        self.assertFalse(result)

    @patch('client.requests.get')
    def test_get_features(self, mock_get):
        """Test: Obtener features"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "features": ["Time", "V1", "V2", "Amount"],
            "total": 4
        }
        mock_get.return_value = mock_response

        result = self.client.get_features()
        self.assertEqual(result["total"], 4)


class TestIntegration(unittest.TestCase):
    """Tests de integración (sin mocks)"""

    def test_privacy_layer_deterministic_bounds(self):
        """Test: DP siempre respeta bounds"""
        dp = DifferentialPrivacyLayer()

        for value in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = dp.apply_dp_to_probability(value)
            self.assertGreaterEqual(result, 0.0, f"Fallo para valor {value}")
            self.assertLessEqual(result, 1.0, f"Fallo para valor {value}")

    def test_mechanism_switching(self):
        """Test: Cambiar mecanismo dinámicamente"""
        dp = DifferentialPrivacyLayer(mechanism="laplace")
        self.assertEqual(dp.mechanism, "laplace")

        # Cambiar a gaussian
        dp.mechanism = "gaussian"
        dp._validate_mechanism()
        self.assertEqual(dp.mechanism, "gaussian")


# ===========================================================================
# Tests de métricas de ruido [Chai & Draxler, 2014]
# ===========================================================================

class TestNoiseMetrics(unittest.TestCase):
    """Tests para metrics/noise_metrics.py"""

    def test_mae_perfect(self):
        pairs = [(0.5, 0.5), (0.8, 0.8)]
        self.assertAlmostEqual(mae(pairs), 0.0)

    def test_rmse_perfect(self):
        pairs = [(0.5, 0.5), (0.8, 0.8)]
        self.assertAlmostEqual(rmse(pairs), 0.0)

    def test_mae_known_value(self):
        # |0.9-0.7| + |0.3-0.1| = 0.2 + 0.2 → mean = 0.2
        pairs = [(0.9, 0.7), (0.3, 0.1)]
        self.assertAlmostEqual(mae(pairs), 0.2)

    def test_rmse_known_value(self):
        # sqrt((0.2^2 + 0.2^2) / 2) = 0.2
        pairs = [(0.9, 0.7), (0.3, 0.1)]
        self.assertAlmostEqual(rmse(pairs), 0.2)

    def test_rmse_ge_mae(self):
        """RMSE ≥ MAE always [Chai & Draxler §2]."""
        pairs = [(0.9, 0.5), (0.4, 0.45), (0.1, 0.6)]
        self.assertGreaterEqual(rmse(pairs), mae(pairs))

    def test_empty_pairs(self):
        self.assertEqual(mae([]), 0.0)
        self.assertEqual(rmse([]), 0.0)

    def test_noise_summary_keys(self):
        report = noise_summary([(0.8, 0.6), (0.3, 0.2)])
        for key in ("mae", "rmse", "rmse_mae_ratio", "noise_std", "n_samples"):
            self.assertIn(key, report)

    def test_noise_summary_ratio_none_when_mae_zero(self):
        report = noise_summary([(0.5, 0.5)])
        self.assertIsNone(report["rmse_mae_ratio"])

    def test_noise_summary_ratio_none_with_single_sample(self):
        """ratio must be None with n<2 — equals 1.0 trivially with 1 sample."""
        report = noise_summary([(0.8, 0.6)])
        self.assertIsNone(report["rmse_mae_ratio"])

    def test_noise_std_none_with_single_sample(self):
        report = noise_summary([(0.8, 0.6)])
        self.assertIsNone(report["noise_std"])

    def test_noise_std_computed_with_multiple_samples(self):
        pairs = [(0.8, 0.6), (0.5, 0.45), (0.9, 0.7)]
        report = noise_summary(pairs)
        self.assertIsNotNone(report["noise_std"])
        self.assertGreaterEqual(report["noise_std"], 0.0)

    def test_noise_std_zero_for_equal_noise(self):
        """Constant noise magnitude across repetitions → std = 0."""
        pairs = [(0.8, 0.6), (0.5, 0.3), (0.9, 0.7)]  # |noise| = 0.2 always
        report = noise_summary(pairs)
        self.assertAlmostEqual(report["noise_std"], 0.0)


# ===========================================================================
# Tests de flip_rate (clasificación) derivada de N repeticiones
# ===========================================================================

class TestFlipRate(unittest.TestCase):
    """Tests para la lógica de flip_rate computada en el loop de /predict."""

    def _count_flips(self, orig_is_fraud, noisy_probs):
        """Replicar la lógica del loop de app.py."""
        return sum(
            1 for p in noisy_probs
            if (1 if p > 0.5 else 0) != orig_is_fraud
        )

    def test_no_flips_when_noise_is_minimal(self):
        noisy = [0.9, 0.85, 0.78, 0.82, 0.91]  # all fraud, orig=1
        self.assertEqual(self._count_flips(1, noisy), 0)

    def test_all_flips_when_noise_inverts(self):
        noisy = [0.1, 0.2, 0.3, 0.4]  # all below 0.5, orig=1 → all flip
        self.assertEqual(self._count_flips(1, noisy), 4)

    def test_partial_flips(self):
        noisy = [0.8, 0.3, 0.7, 0.2, 0.9]  # 2 below 0.5, orig=1
        self.assertEqual(self._count_flips(1, noisy), 2)

    def test_flip_rate_formula(self):
        self.assertAlmostEqual(3 / 10, 0.3)

    def test_no_flip_on_exact_boundary(self):
        # 0.5 > 0.5 is False → is_fraud=0 == orig=0 → no flip
        self.assertEqual(self._count_flips(0, [0.5]), 0)

    def test_flip_just_above_boundary(self):
        self.assertEqual(self._count_flips(0, [0.501]), 1)


# ===========================================================================
# Tests de métricas de clasificación [Powers, 2011; Fawcett, 2006]
# ===========================================================================

class TestClassificationMetrics(unittest.TestCase):
    """Tests para metrics/classification_metrics.py"""

    def setUp(self):
        # 4 TP, 1 FP, 3 TN, 2 FN
        self.y_true = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        self.y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    def test_confusion_matrix(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        self.assertEqual(cm["tp"], 4)
        self.assertEqual(cm["fp"], 1)
        self.assertEqual(cm["tn"], 3)
        self.assertEqual(cm["fn"], 2)

    def test_accuracy(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        self.assertAlmostEqual(accuracy(cm), 0.7)  # (4+3)/10

    def test_fnr(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        # FN=2, TP=4 → FNR = 2/6 ≈ 0.333
        self.assertAlmostEqual(fnr(cm), 2 / 6)

    def test_f1(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        # 2*4 / (2*4 + 1 + 2) = 8/11
        self.assertAlmostEqual(f1(cm), 8 / 11)

    def test_informedness_range(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        val = informedness(cm)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    def test_markedness_range(self):
        cm = confusion_matrix_counts(self.y_true, self.y_pred)
        val = markedness(cm)
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    def test_mcc_perfect_classifier(self):
        """MCC = 1 for perfect predictions."""
        y = [1, 0, 1, 0]
        cm = confusion_matrix_counts(y, y)
        inf_val = informedness(cm)
        mark_val = markedness(cm)
        self.assertAlmostEqual(mcc(inf_val, mark_val), 1.0)

    def test_classification_report_keys(self):
        report = classification_report(self.y_true, self.y_pred)
        for key in ("confusion_matrix", "accuracy", "fnr", "f1",
                    "informedness", "markedness", "mcc", "n_samples"):
            self.assertIn(key, report)

    def test_empty_predictions_zero_denominator(self):
        """All-zero predictions should not raise."""
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        report = classification_report(y_true, y_pred)
        self.assertEqual(report["fnr"], 1.0)
        self.assertEqual(report["f1"], 0.0)


# ===========================================================================
# Tests de métricas de privacidad-utilidad [Geng et al., 2020]
# ===========================================================================

class TestPrivacyMetrics(unittest.TestCase):
    """Tests para metrics/privacy_metrics.py"""

    def test_utility_retention_perfect(self):
        self.assertAlmostEqual(utility_retention(0.9, 0.9), 1.0)

    def test_utility_retention_degraded(self):
        self.assertAlmostEqual(utility_retention(0.9, 0.45), 0.5)

    def test_utility_retention_zero_original(self):
        self.assertAlmostEqual(utility_retention(0.0, 0.0), 1.0)
        self.assertAlmostEqual(utility_retention(0.0, 0.5), 0.0)

    def test_risk_noise_correlation_uncorrelated(self):
        # All-same noise → zero correlation
        pairs = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1)]
        corr = risk_noise_correlation(pairs)
        self.assertAlmostEqual(corr, 0.0)

    def test_risk_noise_correlation_positive(self):
        # Noise increases with risk → positive correlation
        pairs = [(0.1, 0.01), (0.5, 0.05), (0.9, 0.09)]
        corr = risk_noise_correlation(pairs)
        self.assertGreater(corr, 0.9)

    def test_risk_noise_correlation_insufficient_data(self):
        self.assertAlmostEqual(risk_noise_correlation([]), 0.0)
        self.assertAlmostEqual(risk_noise_correlation([(0.5, 0.1)]), 0.0)

    def test_privacy_utility_summary_keys(self):
        summary = privacy_utility_summary(0.92, 0.85, [(0.8, 0.05), (0.3, 0.04)])
        for key in ("utility_original", "utility_with_dp", "utility_retention",
                    "risk_noise_correlation", "n_samples"):
            self.assertIn(key, summary)


# ===========================================================================
# Tests de monotonía del governor [Kotłowski & Słowiński, 2013]
# ===========================================================================

class TestGovernorMonotonicity(unittest.TestCase):
    """Tests para las funciones de monotonía en governor.py"""

    def test_canonical_schedule_is_monotone(self):
        """MONOTONE_EPSILON_SCHEDULE debe ser estrictamente decreciente."""
        self.assertTrue(validate_monotone_schedule(MONOTONE_EPSILON_SCHEDULE))

    def test_violation_raises(self):
        # high_risk listed first with higher epsilon than very_high_risk → violation
        bad_schedule = {"very_high_risk": 5.0, "high_risk": 1.0}
        with self.assertRaises(ValueError):
            validate_monotone_schedule(bad_schedule)

    def test_epsilon_from_risk_decreasing(self):
        """epsilon_from_risk must be non-increasing as risk increases."""
        risks = [0.05, 0.25, 0.45, 0.65, 0.85]
        epsilons = [epsilon_from_risk(r) for r in risks]
        for i in range(1, len(epsilons)):
            self.assertLessEqual(
                epsilons[i], epsilons[i - 1],
                f"Monotonicity broken at risk={risks[i]}: "
                f"epsilon={epsilons[i]} > previous={epsilons[i-1]}"
            )

    def test_epsilon_from_risk_boundaries(self):
        self.assertEqual(epsilon_from_risk(0.0), MONOTONE_EPSILON_SCHEDULE["very_low_risk"])
        self.assertEqual(epsilon_from_risk(1.0), MONOTONE_EPSILON_SCHEDULE["very_high_risk"])


def run_tests():
    """Ejecutar todos los tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestSimplePrivacyGovernor))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelAPIClient))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestFlipRate))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestGovernorMonotonicity))

    # Ejecutar
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🧪 TESTS: Privacy API")
    print("="*70 + "\n")

    result = run_tests()

    # Resumen
    print("\n" + "="*70)
    if result.wasSuccessful():
        print(f"  ✓ Todos los tests pasaron ({result.testsRun} tests)")
    else:
        print(f"  ✗ {len(result.failures)} fallos, {len(result.errors)} errores")
    print("="*70 + "\n")

