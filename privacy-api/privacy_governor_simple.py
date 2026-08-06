"""Privacy Governor mínimo, hardcodeado y preparado para evolucionar.

Esta versión expresa el árbol de decisión directamente con if/elif/else,
sin tablas externas ni un árbol entrenado. Es una implementación previa a
una futura versión basada en tablas y/o un modelo de decisión entrenado.
"""

ABSOLUTE_THRESHOLD_DISTANCE = 0.02
LOW_CONFIDENCE_THRESHOLD = 0.6
EXHAUSTED_BUDGET_ITERATION = 3


def gobernar_politica(
    fraud_probability: float,
    confidence_score: float,
    iteracion: int,
    umbral_decision: float = 0.50,
) -> dict:
    """Selecciona una política de privacidad para una iteración del pipeline."""
    distancia_umbral = abs(fraud_probability - umbral_decision)

    if distancia_umbral < ABSOLUTE_THRESHOLD_DISTANCE:
        politica = "P1"
        epsilon_in = 3.0
        epsilon_out = 5.0
    elif confidence_score < LOW_CONFIDENCE_THRESHOLD:
        politica = "P2"
        epsilon_in = 1.5
        epsilon_out = 2.0
    else:
        politica = "P3"
        epsilon_in = 0.5
        epsilon_out = 0.8

    if iteracion >= EXHAUSTED_BUDGET_ITERATION:
        politica = "P4"
        epsilon_in = 0.0
        epsilon_out = 0.0

    return {
        "modo": "governed",
        "politica": politica,
        "epsilon_in": epsilon_in,
        "epsilon_out": epsilon_out,
        "distancia_umbral": distancia_umbral,
    }