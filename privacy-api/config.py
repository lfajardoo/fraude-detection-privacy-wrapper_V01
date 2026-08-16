"""
Configuración centralizada para la API de Privacy Engineering
Fácil de ajustar sin afectar componentes individuales
"""

import os

# ===== CONFIGURACIÓN DEL MODELO-PYTHON =====
# Leer de variable de entorno, con fallback a localhost
MODEL_API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "http://localhost:8000")
MODEL_API_PREDICT_ENDPOINT = "/predict"
MODEL_API_HEALTH_ENDPOINT = "/health"
MODEL_API_FEATURES_ENDPOINT = "/features"

# Timeout en segundos para llamadas a modelo-python
MODEL_API_TIMEOUT = 10

# ===== CONFIGURACIÓN DE PRIVACIDAD DIFERENCIAL =====
# Mecanismos disponibles: "laplace", "gaussian"
DEFAULT_MECHANISM = "laplace"

# Parámetro epsilon (privacidad): valores más bajos = más privacidad, menos precisión
# Rango típico: [0.1, 10]. Valores comunes: 0.1, 0.5, 1.0, 2.0
EPSILON_FRAUD_PROBABILITY = 2.0  # Para la probabilidad de fraude
EPSILON_CONFIDENCE_SCORE = 2.0   # Para el score de confianza

# Delta (para Gaussian): típicamente muy pequeño
DELTA = 1e-6

# Bounds (límites) para normalizar ruido en valores [0, 1]
LOWER_BOUND_PROBABILITY = 0.0
UPPER_BOUND_PROBABILITY = 1.0
LOWER_BOUND_CONFIDENCE = 0.0
UPPER_BOUND_CONFIDENCE = 1.0

# ===== CONFIGURACIÓN DE LA API =====
PRIVACY_API_HOST = "0.0.0.0"
PRIVACY_API_PORT = 8001  # Puerto diferente al modelo-python (8000)

# ===== CONFIGURACIÓN DE LOGGING =====
DEBUG = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ===== CONFIGURACIÓN DE CACHÉ Y RATE LIMITING =====
# Cache de resultados (para aplicar ruido consistente)
ENABLE_CACHE = False
CACHE_TTL_SECONDS = 300

# ===== CONFIGURACIÓN EXPERIMENTAL =====
# Aplicar privacidad diferencial también a is_fraud (clasificación binaria)?
APPLY_DP_TO_CLASSIFICATION = False

# ===== CONFIGURACIÓN DE EVALUACIÓN POR LOTES =====
# Umbral para binarizar fraud_probability → is_fraud en la evaluación
EVALUATION_THRESHOLD = 0.5
# Número máximo de transacciones por request de evaluación (protección DoS)
EVALUATION_MAX_TRANSACTIONS = 1000

