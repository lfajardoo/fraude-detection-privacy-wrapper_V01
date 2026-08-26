"""Métricas del harness, separadas del cliente ciego."""

import math
from typing import Dict, Iterable, List


def mae(errors: Iterable[float]) -> float:
    values = list(errors)
    return sum(abs(value) for value in values) / len(values) if values else 0.0


def rmse(errors: Iterable[float]) -> float:
    values = list(errors)
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def success_rate(errors: Iterable[float], tol: float) -> float:
    values = list(errors)
    return sum(abs(value) <= tol for value in values) / len(values) if values else 0.0


def build_curve(resultados_por_epsilon: Dict[float, Iterable[float]], tol: float = 0.5) -> List[dict]:
    """Construye una curva ordenada; los errores siguen siendo del harness."""
    curve = []
    for epsilon, errors in sorted(resultados_por_epsilon.items()):
        values = list(errors)
        curve.append({
            "epsilon_in": float(epsilon),
            "mae": mae(values),
            "rmse": rmse(values),
            "n": len(values),
            "success_rate": success_rate(values, tol),
            "floor": mae(values),
        })
    return curve