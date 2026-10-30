"""Baseline externo: respuesta aleatorizada (Warner 1965) sobre la decisión `is_fraud`.

Mecanismo: se publica la etiqueta real con probabilidad e^ε/(1+e^ε) y la
contraria con probabilidad 1/(1+e^ε). Implementación:
`diffprivlib.mechanisms.Binary`. Es agnóstico al modelo: solo necesita la
etiqueta de salida.

QUÉ PROTEGE CADA ENFOQUE (no son intercambiables):
  - Respuesta aleatorizada: protege el RESULTADO de la consulta (la decisión
    publicada). Supone que el contenido de la consulta (las features) llega
    sin ruido a quien evalúa el modelo.
  - Input shield del wrapper: protege el CONTENIDO de la consulta (las
    features) antes de que llegue al modelo.
  - Output shield del wrapper: protege la probabilidad publicada (Laplace,
    Δ = 1); la decisión derivada (`prob > 0.5`) es post-procesamiento.

Por eso la comparación de este módulo es a igual parámetro ε nominal (utilidad
y fuga medida), no a igual garantía frente a un mismo adversario.

La auditoría reutiliza la maquinaria del Nivel 3 (`distinguibilidad`): mismos
intervalos de Clopper-Pearson, mismo techo y mismas filas, de modo que el
baseline entra sin fricción en las tablas y CSV/JSON existentes. Con vecinos
extremos (decisión False/True) la respuesta aleatorizada alcanza exactamente el
cociente e^ε, así que su ε_LB debe quedar muy cerca del ε teórico: sirve de
CALIBRACIÓN del instrumento.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import analitica, distinguibilidad
from .distinguibilidad import (
    COMPATIBLE,
    EPS_SIN_RUIDO,
    FACTOR_ERROR_INYECTADO,
    VIOLACION,
    _anotar_control,
    _auditar,
    control_error_inyectado,
    control_mecanismo_correcto,
    eps_lb,
    techo_eps_lb,
)
from .mecanismo import N_FEATURES, epsilon_salida, generador_salida, privacy_layer, semillas

BARRIDO_EPS_RR = (0.5, 1.0, 2.0, 3.0)

# Probabilidades del modelo sobre las que se promedia la utilidad del output
# shield. Su acierto depende de p (cerca de 0.5 el ruido voltea más), así que se
# fija un conjunto simétrico, sin p = 0.5 (allí la decisión real es ambigua).
PROBS_REPRESENTATIVAS = (0.01, 0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95, 0.99)
CAPA_RR = "baseline_rr"


# --------------------------------------------------------------------------
# Mecanismo y teoría
# --------------------------------------------------------------------------

def _binary(epsilon: float, rng):
    """`Binary` de diffprivlib (parámetros solo por nombre). `rng`: int o RandomState."""
    from diffprivlib.mechanisms import Binary

    return Binary(epsilon=epsilon, value0="0", value1="1", random_state=rng)


def _randomizar(mecanismo, decision: bool, rs: np.random.RandomState) -> bool:
    if mecanismo is None:
        # Binary rechaza ε = 0 (ε = δ = 0); el límite es una moneda justa.
        return bool(rs.random_sample() < 0.5)
    # exp(ε) desborda para ε ~ 1e30 (control sin ruido): el resultado sigue siendo correcto.
    with np.errstate(over="ignore", invalid="ignore"):
        return mecanismo.randomise("1" if decision else "0") == "1"


def voltear(decision: bool, epsilon: float, rng) -> bool:
    """Respuesta aleatorizada de una decisión: la voltea con prob. 1/(1+e^ε).

    `rng` es una semilla entera o un `np.random.RandomState`. Con una semilla
    entera la función es pura (misma entrada, misma salida); para encadenar
    muchas llamadas independientes hay que pasar un `RandomState` compartido.
    """
    rs = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)
    mecanismo = None if epsilon == 0 else _binary(epsilon, rs)
    return _randomizar(mecanismo, decision, rs)


def prob_volteo(epsilon: float) -> float:
    """P(publicar la etiqueta contraria) = 1/(1+e^ε)."""
    return 1.0 / (1.0 + math.exp(epsilon)) if epsilon < 700 else 0.0


def prob_acierto(epsilon: float) -> float:
    """P(la etiqueta publicada coincide con la real) = e^ε/(1+e^ε). Utilidad (forma cerrada)."""
    return 1.0 - prob_volteo(epsilon)


def epsilon_teorico_rr(epsilon: float) -> float:
    """Garantía exacta: ε (vecindad máxima, dos decisiones opuestas)."""
    return epsilon


def utilidad_wrapper_teorica(eps_out: float, probs: Sequence[float] = PROBS_REPRESENTATIVAS) -> float:
    """Forma cerrada del acierto del output shield, promediado sobre `probs`.

    El recorte a [0, 1] no cambia el signo de (valor − 0.5), así que
    P(acierto | p) = 1 − ½·exp(−ε·|p − ½|).
    """
    return float(np.mean([1.0 - 0.5 * math.exp(-eps_out * abs(p - 0.5)) for p in probs]))


def utilidad_wrapper_por_p(
    eps_out: float,
    probs: Sequence[float] = PROBS_REPRESENTATIVAS,
    n: int = 4000,
    seed: int = 0,
) -> Dict[float, float]:
    """Acierto del output shield por valor de p, SIMULADO con la capa real.

    Para cada p: n llamadas a `apply_dp_to_probability(p)` con ε_out dado; la
    decisión publicada es `ruidosa > 0.5` y se compara con la real `p > 0.5`.
    """
    resultado: Dict[float, float] = {}
    for p, s in zip(probs, semillas(seed, len(probs))):
        with epsilon_salida(eps_out), generador_salida(s):
            ruidosas = np.array(
                [privacy_layer.dp_layer.apply_dp_to_probability(p) for _ in range(n)]
            )
        resultado[p] = float(np.mean((ruidosas > 0.5) == (p > 0.5)))
    return resultado


def utilidad_wrapper(
    eps_out: float,
    probs: Sequence[float] = PROBS_REPRESENTATIVAS,
    n: int = 4000,
    seed: int = 0,
) -> float:
    """P(decisión del output shield == decisión real), promedio simple sobre `probs`.

    Depende de p, por eso se simula sobre `PROBS_REPRESENTATIVAS`
    (0.01, 0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95, 0.99).
    """
    return float(np.mean(list(utilidad_wrapper_por_p(eps_out, probs, n, seed).values())))


# --------------------------------------------------------------------------
# Auditoría (reusa el Nivel 3)
# --------------------------------------------------------------------------

def muestrear_rr(
    epsilon: float, T: int, seed: int, factor_escala: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """T publicaciones con decisión real False y T con True.

    `factor_escala` ≠ 1 inyecta un error: el volteo usa ε/factor (0.5 → la
    probabilidad de volteo de 2ε en vez de la de ε).
    """
    eps_real = epsilon / factor_escala
    salidas = []
    for decision, s in zip((False, True), semillas(seed, 2)):
        rng = np.random.RandomState(s)
        mecanismo = None if eps_real == 0 else _binary(eps_real, rng)
        salidas.append(np.array([_randomizar(mecanismo, decision, rng) for _ in range(T)]))
    return salidas[0], salidas[1]


def auditar_rr(
    epsilon: float,
    T: int,
    alpha: float,
    seed: int,
    factor_escala: float = 1.0,
    tipo: str = "barrido",
) -> Dict:
    """Fila del Nivel 3 para la respuesta aleatorizada. Vecinos: False/True. Evento: salida == True.

    p₀ = 1/(1+e^ε), p₁ = e^ε/(1+e^ε): el cociente es exactamente e^ε.
    """
    t0 = time.perf_counter()
    eps_real = epsilon / factor_escala
    s0, s1 = muestrear_rr(epsilon, T, seed, factor_escala)
    fila = _auditar(
        CAPA_RR, None, epsilon, epsilon_teorico_rr(epsilon), eps_real, s0, s1, True, T, alpha,
        tipo, t0, factor_escala=factor_escala,
    )
    # `_auditar` asume p₀ = ½e^{−ε}, p₁ = ½ (capas Laplace): aquí son otras.
    p0_real, p1_real = prob_volteo(eps_real), prob_acierto(eps_real)
    esperado = eps_lb(round(T * p0_real), round(T * p1_real), T, alpha)
    fila.update(
        p0_teorico=prob_volteo(epsilon),
        p1_teorico=prob_acierto(epsilon),
        eps_lb_esperado=esperado,
        resolucion=esperado / epsilon,
        tiempo_s=round(time.perf_counter() - t0, 3),
    )
    return fila


def control_sin_ruido_rr(T: int, alpha: float, seed: int) -> Dict:
    """Sin ruido (ε enorme): ε_LB debe chocar con el techo."""
    fila = auditar_rr(EPS_SIN_RUIDO, T, alpha, seed, tipo="control_sin_ruido")
    fila["eps_lb_esperado"] = fila["techo"]
    fila["resolucion"] = fila["techo"] / fila["eps_teorico"]
    ok = fila["eps_lb"] >= fila["techo"] - 1e-9
    return _anotar_control(
        fila, "eps_lb == techo", ok,
        "alcanza el techo: la auditoría tiene potencia" if ok
        else "HALLAZGO: no alcanza el techo, la auditoría no distingue ni un mecanismo sin ruido",
    )


def ejecutar_auditoria_rr(
    T: int,
    alpha: float,
    seed: int,
    eps_barrido: Sequence[float] = BARRIDO_EPS_RR,
    progreso: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Barrido de ε + los tres controles (mecanismo correcto, error inyectado, sin ruido)."""
    log = progreso or (lambda _m: None)
    techo = techo_eps_lb(T, alpha)
    if max(eps_barrido) >= techo:
        raise ValueError(
            f"T={T} es insuficiente: techo ε_LB={techo:.3f} ≤ máximo ε del barrido={max(eps_barrido):.3f}. "
            "Aumenta T."
        )
    filas = []
    for i, eps in enumerate(eps_barrido):
        kwargs = {"epsilon": eps, "T": T, "alpha": alpha, "seed": seed + i}
        fila = auditar_rr(**kwargs)
        filas.append(control_mecanismo_correcto(fila))
        log(f"rr eps={eps}: {fila['veredicto']} (eps_lb={fila['eps_lb']:.3f})")
        filas.append(control_error_inyectado(auditar_rr, {**kwargs, "seed": seed + i + 5000},
                                             FACTOR_ERROR_INYECTADO))
    filas.append(control_sin_ruido_rr(T, alpha, seed + 9000))
    return {
        "T": T, "alpha": alpha, "seed": seed, "techo": techo, "filas": filas,
        "controles_ok": all(f["control_ok"] for f in filas),
    }


def ajuste(fila: Dict) -> float:
    """ε_LB/ε teórico: 1 = la auditoría recupera exactamente la garantía."""
    return fila["eps_lb"] / fila["eps_teorico"]


# --------------------------------------------------------------------------
# Comparación de utilidad a igual ε nominal
# --------------------------------------------------------------------------

def comparacion_utilidad(
    eps_barrido: Sequence[float],
    T: int,
    alpha: float,
    seed: int,
    filas_rr: Optional[Sequence[Dict]] = None,
    n_utilidad: int = 4000,
    feature: str = "V4",
) -> List[Dict]:
    """Una fila por ε: utilidad y fuga medida de la respuesta aleatorizada y del wrapper.

    El mismo ε nominal se aplica como ε_out (output shield) y como ε_in
    (input shield, ε por feature = ε/15). `wrapper_eps_lb` es el del output
    shield (el que protege el mismo objeto que RR: el resultado);
    `wrapper_in_eps_lb` es el del input shield sobre `feature`.
    """
    por_eps = {f["eps_param"]: f for f in (filas_rr or []) if f["tipo"] == "barrido"}
    filas = []
    for i, eps in enumerate(eps_barrido):
        rr = por_eps.get(eps) or auditar_rr(eps, T, alpha, seed + i)
        sal = distinguibilidad.auditar_salida(eps, T, alpha, seed + 100 + i)
        ent = distinguibilidad.auditar_entrada(feature, eps, T, alpha, seed + 200 + i)
        por_p = utilidad_wrapper_por_p(eps, n=n_utilidad, seed=seed + 300 + i)
        rango = 10.0
        filas.append({
            "epsilon": eps,
            "rr_prob_acierto": prob_acierto(eps),
            "wrapper_out_prob_acierto": float(np.mean(list(por_p.values()))),
            "wrapper_out_prob_acierto_peor_p": min(por_p.values()),
            "wrapper_in_b_relativo": analitica.escala_entrada(eps, delta=rango, n=N_FEATURES) / rango,
            "rr_eps_lb": rr["eps_lb"],
            "wrapper_eps_lb": sal["eps_lb"],
            "wrapper_in_eps_lb": ent["eps_lb"],
            "rr_ajuste": ajuste(rr),
            "wrapper_ajuste": ajuste(sal),
            "wrapper_in_ajuste": ajuste(ent),
            "wrapper_in_eps_teorico_por_feature": ent["eps_teorico"],
        })
    return filas


def lectura_comparacion(filas: Sequence[Dict]) -> List[str]:
    """Texto de lectura (neutral) que acompaña a la tabla; cita cifras del ε = 1 si está."""
    f1 = next((f for f in filas if f["epsilon"] == 1.0), filas[0])
    e = f1["epsilon"]
    return [
        "Lectura de la tabla (neutral; compara a igual parametro epsilon nominal, no a igual garantia ante un mismo adversario).",
        f"A epsilon={e:g}, la respuesta aleatorizada acierta la decision con prob. {f1['rr_prob_acierto']:.1%} (forma cerrada).",
        f"El output shield a eps_out={e:g} acierta {f1['wrapper_out_prob_acierto']:.1%} en promedio sobre "
        f"p en {list(PROBS_REPRESENTATIVAS)} (peor p: {f1['wrapper_out_prob_acierto_peor_p']:.1%}).",
        f"El input shield a eps_in={e:g} usa b/rango = {f1['wrapper_in_b_relativo']:.3g} (n/eps_in, n=15): "
        "con b muy superior al rango, las features quedan casi destruidas en una consulta.",
        "Esto no invalida ninguno de los enfoques: protegen cosas distintas. La respuesta aleatorizada protege el "
        "resultado publicado y supone que el contenido de la consulta llega sin ruido a quien evalua el modelo; "
        "el input shield protege el contenido de la consulta; el output shield, la probabilidad publicada.",
        "Los epsilon nominales tampoco son directamente equiparables: eps_in es un presupuesto por registro repartido "
        "entre 15 features (eps_in/15 por feature), eps_out cubre una probabilidad (delta=1) y RR una etiqueta binaria.",
        "Fuga medida: rr_eps_lb es la del baseline; wrapper_eps_lb la del output shield; wrapper_in_eps_lb la del input "
        "shield (feature auditada). *_ajuste = eps_lb / eps teorico.",
    ]


def escribir_comparacion_csv(ruta: Path, filas: Sequence[Dict]) -> None:
    """CSV con la lectura como comentarios '#' al inicio (leer con pandas: comment='#')."""
    with Path(ruta).open("w", newline="", encoding="utf-8") as fh:
        for linea in lectura_comparacion(filas):
            fh.write(f"# {linea}\n")
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
