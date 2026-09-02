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
    R: int = 3,
    snr_threshold: float = 5.0,
) -> Optional[float]:
    """Estima una feature secreta mediante aproximación lineal local.

    Mejora respecto al guard fijo (tol=1e-9): repite el sondeo R veces,
    estima la desviación estándar de la pendiente, y se abstiene cuando
    |slope_medio| < snr_threshold · std(slope).  Esto evita dividir por
    ruido cuando no hay señal, sin dejar pasar pendientes reales.

    Parameters
    ----------
    R : int
        Repeticiones independientes del par de sondeos.  Cada una llama a
        denoise (2·K requests), así que el costo total del sondeo pasa de
        2·K a 2·K·R.  R=3 es el mínimo para estimar una std; con R=5 la
        estimación es más estable pero triplica el costo vs. R=1.
    snr_threshold : float
        Cuántas desviaciones estándar debe superar |slope_medio| para
        considerarse señal.  2.0 ≈ 95 % de confianza; 3.0 ≈ 99 %.
    """
    if probe_hi == probe_lo:
        raise ValueError("probe_lo y probe_hi deben ser distintos")

    span = probe_hi - probe_lo
    slopes = []
    for _ in range(R):
        low_features = dict(known_features)
        high_features = dict(known_features)
        low_features[secret_name] = probe_lo
        high_features[secret_name] = probe_hi

        logit_low = _logit(denoise(client, low_features, K))
        logit_high = _logit(denoise(client, high_features, K))
        slopes.append((logit_high - logit_low) / span)


    slope_mean = sum(slopes) / len(slopes)

    # Guard degenerado: si todas las pendientes son idénticas y valen 0,
    # slope_std=0 haría que el guard no dispare y luego se divida por cero.
    if slope_mean == 0.0:
        return None

    # Guard estadístico: ¿la pendiente media es distinguible de cero?
    if R >= 2:
        slope_std = (sum((s - slope_mean) ** 2 for s in slopes) / (R - 1)) ** 0.5
        # El guard juzga la precisión de la MEDIA, no la dispersión de una medición
        # individual: por eso el error estándar (std/sqrt(R)), no std. Usar std
        # directamente hace el guard sqrt(R) veces más estricto y anula el beneficio de R.
        slope_sem = slope_std / (R ** 0.5)
        if slope_sem > 0 and abs(slope_mean) < snr_threshold * slope_sem:
            return None
    else:
        if abs(slope_mean) < 1e-4:
            return None

    # Intercept a partir de un sondeo fresco en probe_lo (barato, 1·K requests).
    # Reutilizar los R sondeos anteriores sesga la varianza del guard, así que
    # se toma una medición limpia para el anclaje.
    logit_lo = _logit(denoise(client, {**known_features, secret_name: probe_lo}, K))
    intercept = logit_lo - slope_mean * probe_lo
    return (_logit(p_target) - intercept) / slope_mean