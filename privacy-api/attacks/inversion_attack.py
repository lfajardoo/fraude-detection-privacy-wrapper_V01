"""Adversario ciego para inversión inter-request en espacio crudo."""

import math
from typing import Optional

from .api_client import AttackClient


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _logit(value: float) -> float:
    clipped = _clip_probability(value)
    return math.log(clipped / (1.0 - clipped))


def denoise(client: AttackClient, features: dict, K: int) -> float:
    """Promedia K consultas independientes para cancelar ruido de salida."""
    if K < 1:
        raise ValueError("K debe ser >= 1")
    return sum(client.query(dict(features)) for _ in range(K)) / K


def invert_feature(
    client: AttackClient,
    known_features: dict,
    secret_name: str,
    probe_lo: float,
    probe_hi: float,
    K: int,
    p_target: float,
    tol: float = 1e-9,
) -> Optional[float]:
    """Estima una feature secreta mediante una aproximación lineal local."""
    if probe_hi == probe_lo:
        raise ValueError("probe_lo y probe_hi deben ser distintos")

    low_features = dict(known_features)
    high_features = dict(known_features)
    low_features[secret_name] = probe_lo
    high_features[secret_name] = probe_hi

    logit_low = _logit(denoise(client, low_features, K))
    logit_high = _logit(denoise(client, high_features, K))
    slope = (logit_high - logit_low) / (probe_hi - probe_lo)
    if abs(slope) < tol:
        return None
    intercept = logit_low - slope * probe_lo
    return (_logit(p_target) - intercept) / slope