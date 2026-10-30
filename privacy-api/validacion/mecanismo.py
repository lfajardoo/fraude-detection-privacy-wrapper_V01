"""Utilidades compartidas por los tres niveles de validación.

Regla de diseño: las capas del wrapper (input_privacy_layer.py y
privacy_layer.py) NO se modifican. Todo control del ruido (semilla, observar el
ruido antes del segundo recorte, inyectar una escala errónea) se hace desde
fuera, sustituyendo temporalmente el generador de ruido. Cada context manager
restaura siempre el estado original al salir, incluso ante excepciones.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from unittest import mock

import numpy as np

PRIVACY_API_DIR = Path(__file__).resolve().parents[1]
if str(PRIVACY_API_DIR) not in sys.path:
    sys.path.insert(0, str(PRIVACY_API_DIR))

import privacy_layer  # noqa: E402  (requiere PRIVACY_API_DIR en sys.path)
from input_privacy_layer import FEATURE_BOUNDS, InputPrivacyLayer  # noqa: E402

N_FEATURES: int = len(FEATURE_BOUNDS)
DELTA_V: float = 10.0  # ancho del rango (-5, 5) de V1..V14
DELTA_SALIDA: float = 1.0  # sensibilidad de una probabilidad en [0, 1]

Captura = List[Tuple[float, float]]  # (escala nominal pedida, ruido generado)


def semillas(seed: int, k: int) -> List[int]:
    """k semillas independientes derivadas de `seed` (SeedSequence)."""
    return [int(s) for s in np.random.SeedSequence(seed).generate_state(k)]


def registro_base() -> Dict[str, float]:
    """Registro fijo con todas las features de FEATURE_BOUNDS y Time."""
    base = {f: 0.0 for f in FEATURE_BOUNDS}
    base["Amount"] = 100.0
    base["Time"] = 1000.0
    return base


def vecinos_entrada(feature: str = "V4") -> Tuple[dict, dict]:
    """Par de registros vecinos: `feature` en `low` y en `high`, resto idéntico."""
    if feature not in FEATURE_BOUNDS:
        raise ValueError(f"'{feature}' no está en FEATURE_BOUNDS")
    low, high = FEATURE_BOUNDS[feature]
    base = registro_base()
    return {**base, feature: low}, {**base, feature: high}


def construir_capa_entrada(seed: Optional[int]) -> InputPrivacyLayer:
    """Capa de entrada Laplace con semilla fija (RNG propio, no el global)."""
    return InputPrivacyLayer(mechanism="laplace", seed=seed)


@contextlib.contextmanager
def ruido_entrada_instrumentado(
    capa: InputPrivacyLayer, factor_escala: float = 1.0
) -> Iterator[Captura]:
    """Envuelve `capa._noise` desde fuera.

    - Registra (escala nominal, ruido) de cada llamada. El ruido es el valor
      ANTES del segundo recorte (el que se suma a `clipped`).
    - Si `factor_escala != 1`, el generador real recibe `escala * factor_escala`
      (p. ej. 0.5 inyecta un error b/2) pero la captura guarda la escala nominal.
    """
    original = capa._noise
    previo = capa.__dict__.get("_noise")
    captura: Captura = []

    def _noise(scale: float) -> float:
        ruido = original(scale * factor_escala)
        captura.append((scale, ruido))
        return ruido

    capa._noise = _noise
    try:
        yield captura
    finally:
        if previo is None:
            capa.__dict__.pop("_noise", None)
        else:
            capa._noise = previo


@contextlib.contextmanager
def generador_salida(
    seed: Optional[int] = None,
    factor_escala: float = 1.0,
    captura: Optional[Captura] = None,
) -> Iterator[None]:
    """Sustituye temporalmente `np.random.laplace` (usado por privacy_layer).

    Usa un Generator propio con `seed`; `factor_escala` multiplica la escala
    (inyección de error); `captura` acumula (escala nominal, ruido).
    """
    rng = np.random.default_rng(seed)

    def laplace(loc: float = 0.0, scale: float = 1.0, size=None):
        ruido = rng.laplace(loc, scale * factor_escala, size)
        if captura is not None:
            captura.append((scale, float(ruido)))
        return ruido

    with mock.patch.object(np.random, "laplace", laplace):
        yield


@contextlib.contextmanager
def epsilon_salida(eps_out: float) -> Iterator[None]:
    """Fija temporalmente el ε que `apply_dp_to_probability` toma de config."""
    with mock.patch.object(privacy_layer, "EPSILON_FRAUD_PROBABILITY", eps_out):
        yield


def muestrear_entrada(
    feature: str,
    eps_in: float,
    T: int,
    seed: int,
    factor_escala: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Salida de `feature` en T consultas con v=low y T con v=high.

    Llama a `InputPrivacyLayer.apply` sobre el registro completo (n = 15), de
    modo que ε por feature = eps_in / 15.
    """
    reg_low, reg_high = vecinos_entrada(feature)
    s0, s1 = semillas(seed, 2)
    salidas = []
    for registro, s in ((reg_low, s0), (reg_high, s1)):
        capa = construir_capa_entrada(s)
        with ruido_entrada_instrumentado(capa, factor_escala):
            salidas.append(
                np.array([capa.apply(registro, eps_in)[feature] for _ in range(T)])
            )
    return salidas[0], salidas[1]


def muestrear_salida(
    eps_out: float,
    T: int,
    seed: int,
    factor_escala: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Salida de `apply_dp_to_probability` con entrada 0 y con entrada 1."""
    s0, s1 = semillas(seed, 2)
    salidas = []
    for valor, s in ((0.0, s0), (1.0, s1)):
        with epsilon_salida(eps_out), generador_salida(s, factor_escala):
            salidas.append(
                np.array(
                    [privacy_layer.dp_layer.apply_dp_to_probability(valor) for _ in range(T)]
                )
            )
    return salidas[0], salidas[1]
