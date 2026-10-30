"""Nivel 2: predicción analítica (fórmulas cerradas, sin simulación).

Lo analítico PRUEBA (cota superior de ε); lo empírico (Nivel 3) solo REFUTA.
Todas las funciones son puras.

Definiciones de vecindad usadas en este módulo
----------------------------------------------
- "registro": dos registros vecinos pueden diferir en TODAS las features
  (un individuo sustituido por otro). Es la definición de la tesis/del módulo
  input_privacy_layer: la suma de los ε por feature da ε_in (composición básica).
- "una_feature": los registros vecinos difieren en UNA sola feature. Cada
  feature se perturba con ε_in/n y sensibilidad = su rango, luego la garantía
  de entrada es ε_in/n.
"""

from __future__ import annotations

import math

N_FEATURES = 15
DELTA_V = 10.0
DELTA_SALIDA = 1.0

VECINDADES = ("registro", "una_feature")


def escala_entrada(eps_in: float, delta: float = DELTA_V, n: int = N_FEATURES) -> float:
    """Escala b del Laplace de entrada: b = Δ / (ε_in / n) = Δ·n / ε_in.

    Con Δ=10 (V1..V14), n=15, ε_in=1 → b = 150. Para Amount, Δ = 25 000.
    """
    if eps_in <= 0:
        raise ValueError("eps_in debe ser > 0")
    return delta * n / eps_in


def escala_salida(eps_out: float, delta: float = DELTA_SALIDA) -> float:
    """Escala b del Laplace de salida: b = Δ / ε_out, con Δ = 1 para [0, 1].

    Con ε_out=2 → b = 0.5.
    """
    if eps_out <= 0:
        raise ValueError("eps_out debe ser > 0")
    return delta / eps_out


def eps_por_feature(eps_in: float, n: int = N_FEATURES) -> float:
    """ε que recibe cada feature perturbada: ε_in / n (composición básica)."""
    return eps_in / n


def garantia_por_consulta(
    eps_in: float,
    eps_out: float,
    vecindad: str = "registro",
    n: int = N_FEATURES,
    n_salidas: int = 1,
) -> float:
    """Cota superior de ε de UNA consulta: min(ε_entrada, ε_salida).

    Razonamiento (dos cotas independientes, ambas válidas):
    1. La probabilidad publicada es post-procesamiento del vector de entrada ya
       ruidoso (más ruido de salida independiente). El post-procesamiento no
       degrada DP, así que la consulta es ε_entrada-DP, con
       ε_entrada = ε_in ("registro") o ε_in / n ("una_feature").
    2. El Laplace de salida con Δ = 1 hace que, para CUALQUIER par de
       probabilidades del modelo en [0, 1], las salidas sean ε_out-DP; por
       tanto la consulta es ε_out-DP respecto a cualquier par de entradas.
       Si se publican `n_salidas` valores con ruido independiente (p. ej.
       fraud_probability y confidence_score), la composición básica da
       n_salidas·ε_out en este lado.

    Devuelve min(ε_entrada, n_salidas·ε_out). Se asume mecanismo Laplace,
    aritmética exacta (no cubre ataques de punto flotante, ver Mironov 2012).
    """
    if vecindad not in VECINDADES:
        raise ValueError(f"vecindad debe ser una de {VECINDADES}")
    eps_entrada = eps_in if vecindad == "registro" else eps_por_feature(eps_in, n)
    return min(eps_entrada, n_salidas * eps_out)


def composicion_basica(eps: float, k: int) -> float:
    """Composición básica de k consultas: ε_total = k·ε (Dwork & Roth, Thm 3.14)."""
    return k * eps


def prob_limite_alto(v: float, b: float, high: float) -> float:
    """P(salida = high | v) = ½·exp(−(high − v)/b), por el segundo recorte."""
    return 0.5 * math.exp(-(high - v) / b)


def prob_limite_bajo(v: float, b: float, low: float) -> float:
    """P(salida = low | v) = ½·exp(−(v − low)/b)."""
    return 0.5 * math.exp(-(v - low) / b)


def atenuacion(v: float, b: float, low: float = -5.0, high: float = 5.0) -> float:
    """Atenuación a de la señal tras ruido + recorte.

    a = d/dv E[clip(v + X)] = P(low < v + X < high)
      = 1 − ½·exp(−(high − v)/b) − ½·exp(−(v − low)/b),  X ~ Laplace(0, b).

    Con b = 150 y v ≈ 0: a ≈ 0.033. Es una medida de UTILIDAD/coste del
    atacante, no una prueba de privacidad.
    """
    return 1.0 - prob_limite_alto(v, b, high) - prob_limite_bajo(v, b, low)


def consultas_necesarias(a: float) -> float:
    """Orden de magnitud del nº de consultas para recuperar la señal: ~ 1/a².

    Modelo de coste del atacante (promediar ruido de varianza ~ b² exige
    N ∝ 1/a² para que la señal atenuada supere el error estándar). Salvo una
    constante que depende de la varianza; NO es una prueba de privacidad.
    """
    if a <= 0:
        return math.inf
    return 1.0 / (a * a)
