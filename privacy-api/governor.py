"""
Governor (Fase 1 - Hardcodeado)
Decide la política de privacidad para cada request.
Entrega una GovernanceDecision a privacy_layer.
En Fase 2 esta lógica se reemplazará por reglas dinámicas.

Monotonicity contract (Fase 2 — Kotłowski & Słowiński, 2013):
  The mapping  fraud_risk → epsilon  must be monotone *decreasing*:
    higher fraud risk → lower epsilon → more privacy protection.
  validate_monotone_schedule() enforces this invariant at startup
  and can be called from tests. Phase 2 implementations must pass it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GovernanceDecision:
    """Salida del gobernador hacia privacy_layer."""
    mechanism: str          # "laplace" o "gaussian"
    epsilon: float          # parámetro de privacidad
    delta: float            # usado solo por gaussian
    budget_allowed: bool    # si se permite consumir presupuesto
    epsilon_in: float = None
    epsilon_out: float = None


# ---------------------------------------------------------------------------
# Monotonicity helpers (Phase 2 contract)
# Reference: Kotłowski & Słowiński, IEEE TKDE 25(11), 2013.
# ---------------------------------------------------------------------------

# Canonical discrete schedule: risk breakpoints → epsilon.
# Must be monotone decreasing in epsilon as risk increases.
# Replace or extend this dict when implementing Phase 2.
MONOTONE_EPSILON_SCHEDULE: Dict[str, float] = {
    "very_high_risk":  0.3,   # fraud_probability ≥ 0.8
    "high_risk":       0.5,   # fraud_probability ∈ [0.6, 0.8)
    "medium_risk":     1.0,   # fraud_probability ∈ [0.4, 0.6)
    "low_risk":        2.0,   # fraud_probability ∈ [0.2, 0.4)
    "very_low_risk":   5.0,   # fraud_probability < 0.2
}


def validate_monotone_schedule(schedule: Dict[str, float]) -> bool:
    """
    Return True if epsilon values are non-decreasing in the schedule's declared
    order (high-risk keys first → low-risk keys last).

    Contract: schedule is ordered from highest risk to lowest risk, so epsilon
    must increase (or stay equal) as we iterate: high-risk=low-ε → low-risk=high-ε.
    Raises ValueError with a descriptive message on violation.
    """
    values: List[float] = list(schedule.values())
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            raise ValueError(
                f"Monotonicity violation at position {i}: epsilon={values[i]} < "
                f"previous={values[i-1]}. "
                "Schedule must be ordered high-risk-first with non-decreasing epsilon."
            )
    return True


def epsilon_from_risk(fraud_probability: float) -> float:
    """
    Monotone-decreasing risk → epsilon lookup using MONOTONE_EPSILON_SCHEDULE.
    Phase 2 can replace this with an isotonic regressor trained on data.
    """
    if fraud_probability >= 0.8:
        return MONOTONE_EPSILON_SCHEDULE["very_high_risk"]
    if fraud_probability >= 0.6:
        return MONOTONE_EPSILON_SCHEDULE["high_risk"]
    if fraud_probability >= 0.4:
        return MONOTONE_EPSILON_SCHEDULE["medium_risk"]
    if fraud_probability >= 0.2:
        return MONOTONE_EPSILON_SCHEDULE["low_risk"]
    return MONOTONE_EPSILON_SCHEDULE["very_low_risk"]


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------

class PrivacyGovernor:
    """
    Fase 1: Gobernador hardcodeado.
    Siempre devuelve los mismos parámetros fijos.
    Fase 2: reemplazar decide() con lógica contextual basada en
    epsilon_from_risk() y el árbol de decisión en noise_selector.py.
    """

    # --- parámetros hardcodeados (Fase 1) ---
    _MECHANISM: str   = "laplace"   # cambiar a "gaussian" para probar el otro
    _EPSILON: float   = 1.0
    _DELTA: float     = 1e-6
    _BUDGET_ALLOWED: bool = True

    def decide(self, mechanism: Optional[str] = None) -> GovernanceDecision:
        """
        Devuelve la política de privacidad.
        Fase 1: valores fijos.
        Fase 2: recibirá context (modelo_output, request_metadata) y decidirá
        usando epsilon_from_risk() para garantizar monotonía.
        """
        return GovernanceDecision(
            mechanism=mechanism or self._MECHANISM,
            epsilon=self._EPSILON,
            delta=self._DELTA,
            budget_allowed=self._BUDGET_ALLOWED,
        )


# Instancia global
governor = PrivacyGovernor()

# Validate the canonical schedule at import time — catches config regressions.
validate_monotone_schedule(MONOTONE_EPSILON_SCHEDULE)

