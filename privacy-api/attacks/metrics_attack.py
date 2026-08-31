"""Métricas del harness, separadas del cliente ciego.

Nota sobre el 'piso' (floor):
    El piso irreducible es el sesgo que sobrevive a K→∞ (el efecto del escudo
    de ENTRADA que no se cancela promediando ruido de salida). NO se puede
    estimar con una sola corrida por transacción: mean(errores) agregando
    sobre transacciones distintas cancela sesgos de signo opuesto y subestima
    el piso. La medición honesta del piso vive en build_bias_variance_curve()
    y requiere R corridas independientes por transacción (problema-3).
    build_curve() se conserva para compatibilidad y reporta MAE/RMSE +
    mean_error (sesgo agregado, con la advertencia anterior).
"""

import math
from typing import Dict, Iterable, List


def mae(errors: Iterable[float]) -> float:
    values = list(errors)
    return sum(abs(value) for value in values) / len(values) if values else 0.0


def rmse(errors: Iterable[float]) -> float:
    values = list(errors)
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def mean_error(errors: Iterable[float]) -> float:
    """Error medio con signo (sesgo agregado). Ver advertencia del módulo:
    NO es el piso por-transacción; solo el promedio de errores con signo."""
    values = list(errors)
    return sum(values) / len(values) if values else 0.0


def success_rate(errors: Iterable[float], tol: float) -> float:
    values = list(errors)
    return sum(abs(value) <= tol for value in values) / len(values) if values else 0.0


def build_curve(
    resultados_por_epsilon: Dict[float, Iterable[float]],
    tol: float = 0.5,
) -> List[dict]:
    """Construye una curva ordenada a partir de errores (estimate - real).

    `tol` debe venir normalizado por feature (p. ej. una fracción de la
    desviación pública del secreto); un 0.5 fijo no es comparable entre una V
    (std≈1) y Amount (dólares). Ver evaluator_attack.secret_scale().
    """
    curve = []
    for epsilon, errors in sorted(resultados_por_epsilon.items()):
        values = list(errors)
        curve.append({
            "epsilon": float(epsilon),
            "mae": mae(values),
            "rmse": rmse(values),
            "mean_error": mean_error(values),   # sesgo agregado (con advertencia)
            "n": len(values),
            "success_rate": success_rate(values, tol),
            "tol_abs": tol,                     # tolerancia absoluta usada, ya normalizada
            # 'floor' honesto NO se reporta aquí: requiere R corridas por
            # transacción (build_bias_variance_curve, problema-3).
        })
    return curve