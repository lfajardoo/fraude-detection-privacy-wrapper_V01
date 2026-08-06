"""
Governor (Fase 1 - Hardcodeado)
Decide la política de privacidad para cada request.
Entrega una GovernanceDecision a privacy_layer.
En Fase 2 esta lógica se reemplazará por reglas dinámicas.
"""

from dataclasses import dataclass


@dataclass
class GovernanceDecision:
    """Salida del gobernador hacia privacy_layer."""
    mechanism: str          # "laplace" o "gaussian"
    epsilon: float          # parámetro de privacidad
    delta: float            # usado solo por gaussian
    budget_allowed: bool    # si se permite consumir presupuesto
    epsilon_in: float = None
    epsilon_out: float = None


class PrivacyGovernor:
    """
    Fase 1: Gobernador hardcodeado.
    Siempre devuelve los mismos parámetros fijos.
    Fase 2: reemplazar decide() con lógica contextual.
    """

    # --- parámetros hardcodeados (Fase 1) ---
    _MECHANISM: str   = "laplace"   # cambiar a "gaussian" para probar el otro
    _EPSILON: float   = 1.0
    _DELTA: float     = 1e-6
    _BUDGET_ALLOWED: bool = True

    def decide(self, mechanism: str = None) -> GovernanceDecision:
        """
        Devuelve la política de privacidad.
        Fase 1: valores fijos.
        Fase 2: recibirá context (modelo_output, request_metadata) y decidirá.
        """
        return GovernanceDecision(
            mechanism=mechanism or self._MECHANISM,
            epsilon=self._EPSILON,
            delta=self._DELTA,
            budget_allowed=self._BUDGET_ALLOWED,
        )


# Instancia global
governor = PrivacyGovernor()

