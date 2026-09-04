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


def build_bias_variance_curve(
    per_target_estimates: List[List[float]],
    true_values: List[float],
    floor_runs: int,
    epsilon_in: float,
    epsilon_out: float,
    K: int,
) -> dict:
    """Aggregate the per-target bias-variance decomposition for one DP point."""
    if len(per_target_estimates) != len(true_values):
        raise ValueError("per_target_estimates and true_values must have the same length")
    if floor_runs < 1:
        raise ValueError("floor_runs must be >= 1")

    biases = []
    variances = []
    standard_deviations = []
    abstentions = 0

    for estimates, true_value in zip(per_target_estimates, true_values):
        abstentions += floor_runs - len(estimates)
        if len(estimates) < 2:
            continue

        estimate_mean = sum(estimates) / len(estimates)
        bias = estimate_mean - true_value
        variance = sum((estimate - estimate_mean) ** 2 for estimate in estimates) / (
            len(estimates) - 1
        )
        biases.append(bias)
        variances.append(variance)
        standard_deviations.append(math.sqrt(variance))

    used_targets = len(biases)
    total_targets = len(true_values)
    # Per the module warning, aggregate |bias(t)|: signed biases can cancel across targets.
    piso_sesgo_abs = sum(abs(bias) for bias in biases) / used_targets if used_targets else None
    piso_sesgo_rms = (
        math.sqrt(sum(bias * bias for bias in biases) / used_targets)
        if used_targets
        else None
    )

    return {
        "piso_sesgo_abs": piso_sesgo_abs,
        "piso_sesgo_rms": piso_sesgo_rms,
        "varianza_media": sum(variances) / used_targets if used_targets else None,
        "desv_media": sum(standard_deviations) / used_targets if used_targets else None,
        "n_targets_usados": used_targets,
        "n_targets_excluidos_por_abstencion": total_targets - used_targets,
        "tasa_abstencion": abstentions / (floor_runs * total_targets) if total_targets else 0.0,
        "floor_runs": floor_runs,
        "epsilon_in": epsilon_in,
        "epsilon_out": epsilon_out,
        "K": K,
    }


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