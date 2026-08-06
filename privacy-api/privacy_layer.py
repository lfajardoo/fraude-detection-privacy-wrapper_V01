"""
Capa de privacidad diferencial
Módulo independiente que encapsula mecanismos de privacidad diferencial (numpy)
Fácil de ajustar y extender sin afectar app.py o client.py
"""


import math
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging
from config import (
   DEFAULT_MECHANISM, EPSILON_FRAUD_PROBABILITY, EPSILON_CONFIDENCE_SCORE,
   DELTA, LOWER_BOUND_PROBABILITY, UPPER_BOUND_PROBABILITY,
   LOWER_BOUND_CONFIDENCE, UPPER_BOUND_CONFIDENCE, LOG_LEVEL
)


if TYPE_CHECKING:
   from governor import GovernanceDecision


# Configurar logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)




class DifferentialPrivacyLayer:
    """
   Capa de privacidad diferencial que aplica ruido calibrado a predicciones.


   Implementa los mecanismos estándar de DP con numpy:
   - Laplace:  noise ~ Laplace(0, sensitivity/epsilon)
   - Gaussian: noise ~ Normal(0, sensitivity * sqrt(2*ln(1.25/delta)) / epsilon)


   Diseño modular:
   - Independiente de app.py y client.py
   - Configurable via config.py
   - Fácil de probar y ajustar
    """


    def __init__(self, mechanism: str = DEFAULT_MECHANISM):
       self.mechanism = mechanism.lower()
       self._validate_mechanism()
       logger.info(f" Capa de privacidad inicializada con mecanismo: {self.mechanism}")


    def _validate_mechanism(self):
       if self.mechanism not in ["laplace", "gaussian"]:
           raise ValueError(f"Mecanismo no soportado: {self.mechanism}")


    def apply_dp_to_probability(self, value: float) -> float:
       return self._apply_dp(
           value,
           LOWER_BOUND_PROBABILITY,
           UPPER_BOUND_PROBABILITY,
           EPSILON_FRAUD_PROBABILITY,
           label="fraud_probability"
       )


    def apply_dp_to_confidence(self, value: float) -> float:
       return self._apply_dp(
           value,
           LOWER_BOUND_CONFIDENCE,
           UPPER_BOUND_CONFIDENCE,
           EPSILON_CONFIDENCE_SCORE,
           label="confidence_score"
       )


    def _apply_dp(self, value: float, lower_bound: float, upper_bound: float,
                                epsilon: float, delta: float = DELTA, label: str = "value",
                                mechanism: Optional[str] = None) -> float:
       """
       Aplica privacidad diferencial con mecanismo Laplace o Gaussian (numpy).


       Laplace:  scale = sensitivity / epsilon         (sensitivity=1 para [0,1])
       Gaussian: scale = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
       """
       try:
           if epsilon <= 0.0:
               return value

           # Sensibilidad global = 1 para valores normalizados en [0,1]
           sensitivity = 1.0


           selected_mechanism = mechanism or self.mechanism
           if selected_mechanism == "laplace":
               scale = sensitivity / epsilon
               noise = np.random.laplace(loc=0.0, scale=scale)
           else:  # gaussian
               scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
               noise = np.random.normal(loc=0.0, scale=scale)


           noisy_value = value + noise


           # Clip al rango válido
           clipped_value = float(np.clip(noisy_value, lower_bound, upper_bound))


           logger.debug(
               f"[DP-{selected_mechanism.upper()}] {label}: "
               f"{value:.4f} → {clipped_value:.4f} (ε={epsilon}, scale={scale:.4f})"
           )


           return clipped_value


       except Exception as e:
           logger.error(f"Error aplicando DP: {e}")
           return value
       
    def apply_dp_to_response(self, response: Dict[str, Any],
                            decision: Optional["GovernanceDecision"] = None) -> Dict[str, Any]:
       """
       Aplica privacidad diferencial usando los parámetros del gobernador.
       Si no se pasa decision, usa la configuración por defecto (config.py).
       """
       # Tomar parámetros del gobernador o de config
       if decision is not None:
           mechanism = decision.mechanism
           epsilon_prob = decision.epsilon if decision.epsilon_in is None else decision.epsilon_in
           epsilon_conf = decision.epsilon if decision.epsilon_out is None else decision.epsilon_out
           delta = decision.delta
       else:
           mechanism = self.mechanism
           epsilon_prob = EPSILON_FRAUD_PROBABILITY
           epsilon_conf = EPSILON_CONFIDENCE_SCORE
           delta = DELTA

      # Guardar valores originales de la respuesta del modelo
       if "fraud_probability" in response:
           response["fraud_probability_original"] = response.get("fraud_probability")           
           response["fraud_probability"] = self._apply_dp(
               response["fraud_probability"],
               LOWER_BOUND_PROBABILITY, UPPER_BOUND_PROBABILITY,
               epsilon_prob, delta=delta, label="fraud_probability", mechanism=mechanism
           )
       if "is_fraud" in response:
           response["is_fraud_original"] = response.get("is_fraud")

       if "confidence_score" in response:
           response["confidence_score_original"] = response.get("confidence_score")
           response["confidence_score"] = self._apply_dp(
               response["confidence_score"],
               LOWER_BOUND_CONFIDENCE, UPPER_BOUND_CONFIDENCE,
               epsilon_conf, delta=delta, label="confidence_score", mechanism=mechanism
           )


       if "message" in response and "fraud_probability" in response:
           is_fraud_updated = 1 if response["fraud_probability"] > 0.5 else 0
           if is_fraud_updated != response.get("is_fraud"):
               response["message"] = (
                   "FRAUDE DETECTADO (con ruido DP)"
                   if is_fraud_updated == 1
                   else "Transacción legítima (con ruido DP)"
               )


       return response


    def get_info(self) -> Dict[str, Any]:
       """Retornar información sobre configuración de privacidad"""
       return {
           "mechanism": self.mechanism,
           "epsilon_fraud_probability": EPSILON_FRAUD_PROBABILITY,
           "epsilon_confidence_score": EPSILON_CONFIDENCE_SCORE,
           "delta": DELTA,
           "bounds_probability": (LOWER_BOUND_PROBABILITY, UPPER_BOUND_PROBABILITY),
           "bounds_confidence": (LOWER_BOUND_CONFIDENCE, UPPER_BOUND_CONFIDENCE),
       }




# Instancia global (singleton)
dp_layer = DifferentialPrivacyLayer()
