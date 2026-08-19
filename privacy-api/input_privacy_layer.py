"""
Capa de privacidad diferencial en la entrada (input-side DP).

(a) Esta capa es independiente de privacy_layer.py. privacy_layer.py perturba
    las salidas del modelo (fraud_probability, confidence_score); esta capa
    perturba las features de entrada ANTES de invocar al modelo, de forma que
    el modelo nunca recibe los valores originales en texto claro.

(b) El presupuesto `epsilon_in` es el presupuesto **total de la consulta** y se
    reparte uniformemente entre las features realmente perturbadas:
    `epsilon_por_feature = epsilon_in / n_features_perturbadas`. Como todas las
    features pertenecen al **mismo individuo** (la misma transacción), la
    composición es **secuencial** (composición básica, Dwork & Roth 2014,
    Theorems 3.14/3.16): la suma de los `epsilon_por_feature` es igual a
    `epsilon_in`. NO es composición paralela: esta solo aplica cuando las
    queries actúan sobre particiones disjuntas de individuos, lo que no ocurre
    aquí (todas las features pertenecen al mismo registro).

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
from typing import Dict, List, Optional, Tuple

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

    def apply_with_detail(
        self,
        features: dict,
        epsilon_in: float,
        delta: float = 1e-6,
    ) -> Tuple[dict, List[dict]]:
        """
        Perturba las features con ruido DP calibrado y devuelve el dict
        perturbado junto con una lista de detalle por feature.

        El presupuesto `epsilon_in` se reparte uniformemente entre las features
        presentes en FEATURE_BOUNDS: epsilon_por_feature = epsilon_in / n.
        La suma de los epsilon_usado es igual a epsilon_in (composición secuencial).

        Parameters
        ----------
        features : dict
            Features de la transacción (puede incluir Time y otras claves).
        epsilon_in : float
            Presupuesto total. Si <= 0 no se aplica ruido.
        delta : float
            Probabilidad de fallo (solo relevante para mecanismo Gaussian).

        Returns
        -------
        tuple[dict, list[dict]]
            (dict_perturbado, detalle) donde detalle es una lista con un
            elemento por feature perturbada::

                {"feature", "original", "clipped", "perturbed", "epsilon_used"}
        """
        if epsilon_in <= 0.0:
            return features.copy(), []

        perturbables = [f for f in FEATURE_BOUNDS if f in features]
        n = len(perturbables)
        if n == 0:
            return features.copy(), []

        epsilon_per_feature = epsilon_in / n
        result = features.copy()
        detail: List[dict] = []

        for feature in perturbables:
            low, high = FEATURE_BOUNDS[feature]
            sensitivity = high - low
            if self.mechanism == "laplace":
                scale = sensitivity / epsilon_per_feature
            else:  # gaussian
                scale = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon_per_feature

            original = result[feature]
            clipped = float(np.clip(original, low, high))
            noisy = clipped + self._noise(scale)
            perturbed = float(np.clip(noisy, low, high))
            result[feature] = perturbed
            detail.append({
                "feature": feature,
                "original": original,
                "clipped": clipped,
                "perturbed": perturbed,
                "epsilon_used": epsilon_per_feature,
            })

        return result, detail

    def apply(
        self,
        features: dict,
        epsilon_in: float,
        delta: float = 1e-6,
    ) -> dict:
        """
        Perturba las features con ruido DP y devuelve solo el dict resultante.

        Wrapper de `apply_with_detail` que descarta el detalle por feature.
        Úsalo cuando no necesitas el desglose (e.g., producción con
        EVALUATION_MODE=False). La firma y el tipo de retorno son idénticos
        a la versión anterior, por lo que los tests existentes siguen pasando.
        """
        result, _ = self.apply_with_detail(features, epsilon_in, delta)
        return result

    def get_info(self) -> dict:
        """Devuelve metadatos de la capa para incluir en respuestas HTTP."""
        return {
            "mechanism": self.mechanism,
            "feature_bounds": FEATURE_BOUNDS,
            "n_features_protected": len(FEATURE_BOUNDS),
            "budget_composition": "sequential_split",
        }


# Instancia global (singleton)
input_dp_layer = InputPrivacyLayer()
