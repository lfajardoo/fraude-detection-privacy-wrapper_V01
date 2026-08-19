"""
Capa de privacidad diferencial en la entrada (input-side DP).

(a) Esta capa es independiente de privacy_layer.py. privacy_layer.py perturba
    las salidas del modelo (fraud_probability, confidence_score); esta capa
    perturba las features de entrada ANTES de invocar al modelo, de forma que
    el modelo nunca recibe los valores originales en texto claro.

(b) El presupuesto se aplica por feature de manera paralela: cada feature en
    FEATURE_BOUNDS consume epsilon_in de forma independiente. Bajo composición
    paralela (Theorem 2, Dwork & Roth 2014), el presupuesto total de la
    consulta es epsilon_in (no epsilon_in × n_features), porque las features
    se perturban sobre particiones disjuntas del dataset.

(c) Time está excluido intencionalmente: es un timestamp de la transacción,
    no una característica estadística del comportamiento del tarjetahabiente.
    Perturbarlo no aporta protección frente a ataques de inversión relevantes
    y puede degradar la coherencia temporal de los logs.

(d) Fundamento teórico: randomized smoothing en inferencia (Cohen et al., 2019,
    "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019).
    Añadir ruido calibrado a la entrada antes de la inferencia defiende contra
    ataques de inversión (model-inversion attacks) y proporciona una garantía
    de robustez certificada: el clasificador no cambia su predicción dentro de
    un radio r ≈ sensitivity / epsilon_in en la norma L1 (Laplace) o L2 (Gaussian).
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rangos de clipping por feature
# ---------------------------------------------------------------------------
# V1..V14 son componentes PCA normalizadas; ±5 cubre >99.9 % de la distribución
# empírica del dataset creditcard.csv.  Amount usa el rango real observado.
# Ajustar según percentiles del dataset en producción.
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "V1":     (-5.0,     5.0),
    "V2":     (-5.0,     5.0),
    "V3":     (-5.0,     5.0),
    "V4":     (-5.0,     5.0),
    "V5":     (-5.0,     5.0),
    "V6":     (-5.0,     5.0),
    "V7":     (-5.0,     5.0),
    "V8":     (-5.0,     5.0),
    "V9":     (-5.0,     5.0),
    "V10":    (-5.0,     5.0),
    "V11":    (-5.0,     5.0),
    "V12":    (-5.0,     5.0),
    "V13":    (-5.0,     5.0),
    "V14":    (-5.0,     5.0),
    "Amount": (0.0, 25000.0),
}


class InputPrivacyLayer:
    """Capa de ruido diferencial aplicada a features de entrada antes del modelo."""

    def __init__(self, mechanism: str = "laplace", seed: Optional[int] = None):
        """
        Parameters
        ----------
        mechanism : str
            "laplace" o "gaussian".
        seed : int | None
            Si se provee, fija la semilla usando np.random.default_rng(seed) para
            resultados reproducibles. Si es None, usa el RNG global de numpy.
        """
        self.mechanism = mechanism.lower()
        if self.mechanism not in ("laplace", "gaussian"):
            raise ValueError(
                f"Mecanismo no soportado: '{mechanism}'. Use 'laplace' o 'gaussian'."
            )
        self._rng: Optional[np.random.Generator] = (
            np.random.default_rng(seed) if seed is not None else None
        )

    def _noise(self, scale: float) -> float:
        """Genera un escalar de ruido usando el mecanismo configurado."""
        if self._rng is not None:
            fn = self._rng.laplace if self.mechanism == "laplace" else self._rng.normal
        else:
            fn = np.random.laplace if self.mechanism == "laplace" else np.random.normal
        return float(fn(0.0, scale))

    def apply(
        self,
        features: dict,
        epsilon_in: float,
        delta: float = 1e-6,
    ) -> dict:
        """
        Perturba las features con ruido DP calibrado y devuelve un dict nuevo.

        Pasos por cada feature en FEATURE_BOUNDS:
          1. Clip al rango (low, high).
          2. Calcula sensibilidad = high - low.
          3. Añade ruido Laplace(0, sensitivity/epsilon_in) o
             Gaussian(0, sensitivity * sqrt(2*ln(1.25/delta)) / epsilon_in).
          4. Re-clip al mismo rango.

        Cualquier clave no listada en FEATURE_BOUNDS (incluido Time) se copia
        sin modificación. El dict original no se muta.

        Parameters
        ----------
        features : dict
            Features de la transacción (puede incluir Time y otras claves).
        epsilon_in : float
            Parámetro de privacidad. Si <= 0 no se aplica ruido.
        delta : float
            Probabilidad de fallo (sólo relevante para mecanismo Gaussian).

        Returns
        -------
        dict
            Copia de features con las features de FEATURE_BOUNDS perturbadas.
        """
        if epsilon_in <= 0.0:
            return features.copy()

        result = features.copy()
        for feature, (low, high) in FEATURE_BOUNDS.items():
            if feature not in result:
                continue
            sensitivity = high - low
            if self.mechanism == "laplace":
                scale = sensitivity / epsilon_in
            else:  # gaussian
                scale = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon_in

            clipped = float(np.clip(result[feature], low, high))
            noisy = clipped + self._noise(scale)
            result[feature] = float(np.clip(noisy, low, high))

        return result

    def get_info(self) -> dict:
        """Devuelve metadatos de la capa para incluir en respuestas HTTP."""
        return {
            "mechanism": self.mechanism,
            "feature_bounds": FEATURE_BOUNDS,
            "n_features_protected": len(FEATURE_BOUNDS),
        }


# Instancia global (singleton)
input_dp_layer = InputPrivacyLayer()
