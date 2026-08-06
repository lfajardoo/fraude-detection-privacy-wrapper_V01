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
from governor import GovernanceDecision
from privacy_governor_simple import gobernar_politica


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


def run_tests():
    """Ejecutar todos los tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestSimplePrivacyGovernor))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelAPIClient))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

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

