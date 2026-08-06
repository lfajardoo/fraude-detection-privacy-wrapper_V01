"""
Cliente HTTP para consumir modelo-python
Módulo independiente que abstrae la comunicación con la API del modelo
Fácil de cambiar sin afectar app.py o privacy_layer.py
"""

import requests
from typing import Dict, Any, Optional
import logging
from config import (
    MODEL_API_BASE_URL, MODEL_API_PREDICT_ENDPOINT, MODEL_API_HEALTH_ENDPOINT,
    MODEL_API_FEATURES_ENDPOINT, MODEL_API_TIMEOUT, LOG_LEVEL
)

# Configurar logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelAPIClient:
    """
    Cliente para consumir la API de modelo-python.

    Diseño:
    - Encapsula toda la lógica de comunicación HTTP
    - Manejo de errores centralizado
    - Fácil de mockear en tests
    - Independiente de privacy_layer.py y app.py
    """

    def __init__(self, base_url: str = MODEL_API_BASE_URL, timeout: int = MODEL_API_TIMEOUT):
        """
        Inicializar cliente.

        Args:
            base_url: URL base del modelo-python (ej: http://localhost:8000)
            timeout: Timeout en segundos para requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"✓ Cliente del modelo inicializado: {self.base_url}")

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Llamar al endpoint /predict del modelo-python.

        Args:
            transaction: Diccionario con características de la transacción
                        {Time, V1, V2, ..., V14, Amount}

        Returns:
            Respuesta del modelo: {is_fraud, fraud_probability, confidence_score, message}

        Raises:
            requests.RequestException: Si falla la comunicación
            ValueError: Si la respuesta es inválida
        """
        url = f"{self.base_url}{MODEL_API_PREDICT_ENDPOINT}"

        try:
            logger.debug(f"Enviando request a {url}")
            response = requests.post(
                url,
                json=transaction,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Respuesta recibida: {result}")
            return result

        except requests.exceptions.ConnectionError as e:
            logger.error(f"No se puede conectar a {url}: {e}")
            raise ConnectionError(
                f"No se puede conectar al modelo en {self.base_url}. "
                f"¿Está ejecutándose modelo-python?"
            )

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout conectando a {url}")
            raise TimeoutError(f"Timeout en {url} (>{self.timeout}s)")

        except requests.exceptions.HTTPError as e:
            logger.error(f"Error HTTP: {response.status_code} - {response.text}")
            raise ValueError(f"Error del modelo: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            raise

    def health(self) -> bool:
        """
        Verificar si el modelo-python está disponible.

        Returns:
            True si está sano, False en caso contrario
        """
        url = f"{self.base_url}{MODEL_API_HEALTH_ENDPOINT}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            is_healthy = result.get("status") == "healthy" and result.get("model_loaded", False)
            logger.info(f"Estado del modelo: {'✓ Sano' if is_healthy else '✗ No disponible'}")
            return is_healthy

        except Exception as e:
            logger.warning(f"No se puede verificar salud del modelo: {e}")
            return False

    def get_features(self) -> Dict[str, Any]:
        """
        Obtener lista de características esperadas del modelo.

        Returns:
            {features: [...], total: int}

        Raises:
            requests.RequestException: Si falla la comunicación
        """
        url = f"{self.base_url}{MODEL_API_FEATURES_ENDPOINT}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Características obtenidas: {result['total']} features")
            return result

        except Exception as e:
            logger.error(f"Error obteniendo features: {e}")
            raise


# Instancia global (singleton)
model_client = ModelAPIClient()

