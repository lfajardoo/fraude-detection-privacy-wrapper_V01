"""Nivel 3: auditoría empírica de distinguibilidad (cota INFERIOR de ε).

Definición de DP aplicada directamente: para dos entradas vecinas x₀, x₁ y un
evento S,  P[M(x₁)∈S] ≤ e^ε · P[M(x₀)∈S]. Con T consultas por entrada se
estima p̂₀, p̂₁ y se obtiene, con intervalos de Clopper-Pearson unilaterales,

    ε_LB = ln( CP_inf(p̂₁) / CP_sup(p̂₀) )

que con probabilidad ≥ 1 − α es ≤ ε real del mecanismo. Si ε_LB supera el ε
teórico (Nivel 2) hay una violación: el mecanismo NO cumple su garantía.
Si ε_LB ≤ ε teórico NO se prueba nada: solo no se ha refutado.

α global: se reparte α/2 a cada proporción (cada una unilateral).
No hay corrección por multiplicidad entre auditorías del barrido: cada
auditoría tiene probabilidad ≤ α de falsa violación.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import beta

from . import analitica
from .mecanismo import FEATURE_BOUNDS, N_FEATURES, muestrear_entrada, muestrear_salida

# Barrido por defecto (ε_in total → ε por feature = ε_in/15 ∈ {0.067, 1, 3}).
BARRIDO_EPS_IN = (1.0, 15.0, 45.0)
BARRIDO_EPS_OUT = (0.5, 1.0, 2.0)
# ε tan grande que el ruido (~1e-30) desaparece por redondeo de punto flotante.
EPS_SIN_RUIDO = 1e30
FACTOR_ERROR_INYECTADO = 0.5  # escala b/2

COMPATIBLE = "compatible"
VIOLACION = "VIOLACIÓN"
SIN_POTENCIA = "sin potencia"


# --------------------------------------------------------------------------
# Estadística
# --------------------------------------------------------------------------

def cp_inferior(k: int, T: int, alpha: float) -> float:
    """Límite inferior unilateral de Clopper-Pearson al nivel 1 − alpha."""
    return 0.0 if k <= 0 else float(beta.ppf(alpha, k, T - k + 1))


def cp_superior(k: int, T: int, alpha: float) -> float:
    """Límite superior unilateral de Clopper-Pearson al nivel 1 − alpha."""
    return 1.0 if k >= T else float(beta.ppf(1.0 - alpha, k + 1, T - k))


def eps_lb(k0: int, k1: int, T: int, alpha: float) -> float:
    """ε_LB = ln(CP_inf(p̂₁)/CP_sup(p̂₀)) con α/2 por proporción, truncado en 0."""
    a = alpha / 2.0
    inf1 = cp_inferior(k1, T, a)
    sup0 = cp_superior(k0, T, a)
    if inf1 <= 0.0:
        return 0.0
    return max(0.0, math.log(inf1 / sup0))


def techo_eps_lb(T: int, alpha: float) -> float:
    """Máximo ε_LB alcanzable (k₀=0, k₁=T): ln(a^{1/T}/(1−a^{1/T})), a=α/2.

    T=500, α=0.01 → ≈ 4.54.
    """
    return eps_lb(0, T, T, alpha)


def veredicto(eps_teorico: float, eps_lb_valor: float, techo: float) -> str:
    """'sin potencia' si ε teórico ≥ techo (no resoluble); si no, compara."""
    if eps_teorico >= techo:
        return SIN_POTENCIA
    return VIOLACION if eps_lb_valor > eps_teorico else COMPATIBLE


def eps_lb_esperado(eps_real: float, T: int, alpha: float) -> float:
    """ε_LB que se obtendría con los conteos esperados si el ε REAL es eps_real.

    Con vecinos extremos: p₀ = ½e^{−ε}, p₁ = ½. Sirve para el análisis de
    potencia: ¿puede esta auditoría, con este T, resolver este ε?
    """
    p0 = 0.5 * math.exp(-eps_real)
    return eps_lb(round(T * p0), round(T * 0.5), T, alpha)


# --------------------------------------------------------------------------
# Auditorías por capa
# --------------------------------------------------------------------------

def _auditar(
    capa: str,
    feature: Optional[str],
    eps_param: float,
    eps_teorico: float,
    eps_real: float,
    salidas0: np.ndarray,
    salidas1: np.ndarray,
    evento_valor: float,
    T: int,
    alpha: float,
    tipo: str,
    t0: float,
    **extra,
) -> Dict:
    k0 = int(np.count_nonzero(salidas0 == evento_valor))
    k1 = int(np.count_nonzero(salidas1 == evento_valor))
    a = alpha / 2.0
    techo = techo_eps_lb(T, alpha)
    lb = eps_lb(k0, k1, T, alpha)
    return {
        "capa": capa,
        "tipo": tipo,
        "feature": feature,
        "eps_param": eps_param,
        "eps_teorico": eps_teorico,
        "eps_real_mecanismo": eps_real,
        "T": T,
        "alpha": alpha,
        "k0": k0,
        "k1": k1,
        "p0_hat": k0 / T,
        "p1_hat": k1 / T,
        "p0_teorico": 0.5 * math.exp(-eps_teorico),
        "p1_teorico": 0.5,
        "cp_sup_p0": cp_superior(k0, T, a),
        "cp_inf_p1": cp_inferior(k1, T, a),
        "eps_lb": lb,
        "eps_lb_esperado": eps_lb_esperado(eps_real, T, alpha),
        "resolucion": eps_lb_esperado(eps_real, T, alpha) / eps_teorico,
        "techo": techo,
        "veredicto": veredicto(eps_teorico, lb, techo),
        "tiempo_s": round(time.perf_counter() - t0, 3),
        **extra,
    }


def auditar_entrada(
    feature: str,
    eps_in: float,
    T: int,
    alpha: float,
    seed: int,
    factor_escala: float = 1.0,
    tipo: str = "barrido",
) -> Dict:
    """Capa de entrada. Vecinos: `feature` en low y en high. Evento: salida == high.

    ε teórico = ε_in/n (vecindad una_feature). Con estos vecinos el cociente
    teórico es exactamente e^{ε_in/n}: p₀ = ½e^{−ε_f}, p₁ = ½.
    `factor_escala` ≠ 1 inyecta un error de escala (ε real = ε_f/factor).
    """
    t0 = time.perf_counter()
    low, high = FEATURE_BOUNDS[feature]
    eps_f = analitica.eps_por_feature(eps_in, N_FEATURES)
    eps_real = eps_f / factor_escala
    s0, s1 = muestrear_entrada(feature, eps_in, T, seed, factor_escala)
    return _auditar(
        "entrada", feature, eps_in, eps_f, eps_real, s0, s1, high, T, alpha, tipo, t0,
        factor_escala=factor_escala,
    )


def auditar_salida(
    eps_out: float,
    T: int,
    alpha: float,
    seed: int,
    factor_escala: float = 1.0,
    tipo: str = "barrido",
) -> Dict:
    """Capa de salida. Vecinos: probabilidad 0 y 1 (Δ = 1). Evento: salida == 1.

    Cociente teórico e^{ε_out}: p₀ = ½e^{−ε_out}, p₁ = ½.
    """
    t0 = time.perf_counter()
    s0, s1 = muestrear_salida(eps_out, T, seed, factor_escala)
    return _auditar(
        "salida", None, eps_out, eps_out, eps_out / factor_escala, s0, s1, 1.0, T, alpha,
        tipo, t0, factor_escala=factor_escala,
    )


# --------------------------------------------------------------------------
# Opcional: extremo a extremo contra la API (mode=legacy, con el modelo)
# --------------------------------------------------------------------------

TRANSACCION_E2E = {
    "Time": 0.0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38, "V5": -0.33,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
    "V12": -0.62, "V13": -0.99, "V14": -0.31, "Amount": 149.62,
}


def auditar_e2e(
    url: str,
    feature: str,
    eps_in: float,
    eps_out: float,
    T: int,
    alpha: float,
    timeout: float = 30.0,
) -> Dict:
    """Una corrida contra la API en mode=legacy: el MODELO forma parte de la cadena.

    Vecinos: `feature` en low y en high (resto fijo). Evento fijado a priori:
    is_fraud == 1 en la salida ruidosa. Una petición por muestra
    (noise_repetitions=1) para que las T muestras sean independientes; el
    servidor no está sembrado, así que esta corrida no es reproducible.
    Por post-procesamiento ε_LB no puede superar min(ε_in/n, ε_out) si todo
    está bien. Se prueban ambos sentidos del evento (α/2 cada uno).
    """
    import requests  # dependencia opcional: solo para este chequeo

    t0 = time.perf_counter()
    low, high = FEATURE_BOUNDS[feature]
    ks = []
    with requests.Session() as sesion:
        for valor in (low, high):
            k = 0
            for _ in range(T):
                payload = {
                    **TRANSACCION_E2E, feature: valor, "noise_repetitions": 1,
                    "epsilon_in": eps_in, "epsilon_out": eps_out,
                }
                r = sesion.post(f"{url.rstrip('/')}/predict", params={"mode": "legacy"},
                                json=payload, timeout=timeout)
                r.raise_for_status()
                k += int(r.json()["output_shield"]["samples"][0]["is_fraud"] == 1)
            ks.append(k)
    k0, k1 = ks
    a = alpha / 2.0
    lb = max(eps_lb(k0, k1, T, a), eps_lb(k1, k0, T, a))
    eps_teorico = min(analitica.eps_por_feature(eps_in, N_FEATURES), eps_out)
    techo = techo_eps_lb(T, a)
    return {
        "capa": "e2e_legacy", "tipo": "e2e", "feature": feature,
        "eps_param": eps_in, "eps_out": eps_out, "eps_teorico": eps_teorico,
        "T": T, "alpha": alpha, "k0": k0, "k1": k1, "p0_hat": k0 / T, "p1_hat": k1 / T,
        "eps_lb": lb, "techo": techo, "veredicto": veredicto(eps_teorico, lb, techo),
        "tiempo_s": round(time.perf_counter() - t0, 3),
    }


# --------------------------------------------------------------------------
# Controles obligatorios
# --------------------------------------------------------------------------

def _anotar_control(fila: Dict, esperado: str, ok: bool, detalle: str) -> Dict:
    fila["control_esperado"] = esperado
    fila["control_ok"] = bool(ok)
    fila["control_detalle"] = detalle
    return fila


def control_mecanismo_correcto(fila: Dict) -> Dict:
    """Esperado: ε_LB ≤ ε teórico. Si lo supera es un HALLAZGO."""
    ok = fila["veredicto"] != VIOLACION
    return _anotar_control(
        fila, "eps_lb <= eps_teorico", ok,
        "compatible con la garantía" if ok else "HALLAZGO: ε_LB supera el ε teórico",
    )


def control_sin_ruido(capa: str, feature: str, T: int, alpha: float, seed: int) -> Dict:
    """Sin ruido (ε enorme): ε_LB debe chocar con el techo → el test tiene potencia."""
    if capa == "entrada":
        fila = auditar_entrada(feature, EPS_SIN_RUIDO * N_FEATURES, T, alpha, seed,
                               tipo="control_sin_ruido")
    else:
        fila = auditar_salida(EPS_SIN_RUIDO, T, alpha, seed, tipo="control_sin_ruido")
    # Con ruido por debajo de la resolución de float, p₁ = 1: el máximo es el techo.
    fila["eps_lb_esperado"] = fila["techo"]
    fila["resolucion"] = fila["techo"] / fila["eps_teorico"]
    ok = fila["eps_lb"] >= fila["techo"] - 1e-9
    return _anotar_control(
        fila, "eps_lb == techo", ok,
        "alcanza el techo: la auditoría tiene potencia" if ok
        else "HALLAZGO: no alcanza el techo, la auditoría no distingue ni un mecanismo sin ruido",
    )


def control_error_inyectado(
    auditar: Callable[..., Dict], kwargs: Dict, factor: float = FACTOR_ERROR_INYECTADO
) -> Dict:
    """Error inyectado (escala b·factor): ε_LB debe superar el ε teórico.

    Solo se exige detección cuando el análisis de potencia a priori
    (`eps_lb_esperado` con el ε real inyectado) predice que es detectable;
    si no, el control se marca 'sin potencia esperada' y no cuenta como fallo.
    """
    fila = auditar(factor_escala=factor, tipo="control_error_inyectado", **kwargs)
    detectable = fila["eps_lb_esperado"] > fila["eps_teorico"]
    detectado = fila["eps_lb"] > fila["eps_teorico"]
    fila["veredicto"] = VIOLACION if detectado else (
        SIN_POTENCIA if not detectable else COMPATIBLE
    )
    if not detectable:
        return _anotar_control(
            fila, "eps_lb > eps_teorico (si es detectable)", True,
            "sin potencia esperada para este ε con este T (no cuenta como fallo)",
        )
    return _anotar_control(
        fila, "eps_lb > eps_teorico (si es detectable)", detectado,
        "bug inyectado detectado" if detectado
        else "HALLAZGO: el bug inyectado NO fue detectado pese a ser detectable",
    )


def ejecutar_auditoria(
    T: int,
    alpha: float,
    seed: int,
    feature: str = "V4",
    eps_in_barrido: Sequence[float] = BARRIDO_EPS_IN,
    eps_out_barrido: Sequence[float] = BARRIDO_EPS_OUT,
    progreso: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Barrido por capa y ε + los tres controles. Devuelve filas y metadatos."""
    log = progreso or (lambda _m: None)
    techo = techo_eps_lb(T, alpha)
    eps_f_max = max(analitica.eps_por_feature(e) for e in eps_in_barrido)
    eps_max = max([eps_f_max, *eps_out_barrido])
    if eps_max >= techo:
        raise ValueError(
            f"T={T} es insuficiente: techo ε_LB={techo:.3f} ≤ máximo ε del barrido={eps_max:.3f}. "
            "Aumenta T."
        )
    filas = []
    contador = 0
    for capa, valores in (("entrada", eps_in_barrido), ("salida", eps_out_barrido)):
        for eps in valores:
            kwargs = {"T": T, "alpha": alpha, "seed": seed + contador}
            contador += 1
            if capa == "entrada":
                auditar = auditar_entrada
                kwargs.update(feature=feature, eps_in=eps)
            else:
                auditar = auditar_salida
                kwargs.update(eps_out=eps)
            fila = auditar(**kwargs)
            filas.append(control_mecanismo_correcto(fila))
            log(f"{capa} eps={eps}: {fila['veredicto']} (eps_lb={fila['eps_lb']:.3f})")
            filas.append(control_error_inyectado(auditar, {**kwargs, "seed": kwargs["seed"] + 5000}))
        filas.append(control_sin_ruido(capa, feature, T, alpha, seed + 9000 + contador))
        log(f"{capa}: control sin ruido -> eps_lb={filas[-1]['eps_lb']:.3f} (techo {techo:.3f})")
    return {
        "T": T,
        "alpha": alpha,
        "seed": seed,
        "feature": feature,
        "techo": techo,
        "n_features": N_FEATURES,
        "filas": filas,
        "controles_ok": all(f["control_ok"] for f in filas),
    }
