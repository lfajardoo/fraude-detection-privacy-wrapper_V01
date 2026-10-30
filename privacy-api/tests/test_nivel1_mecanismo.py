"""Nivel 1: tests del mecanismo (segundos, sin modelo ni API).

Semillas fijas => resultados deterministas. Las tolerancias estadísticas se
justifican en cada test: con semilla fija el test no es aleatorio, pero la
tolerancia se fija como k·(error estándar teórico) para que no sea un ajuste
a medida del valor observado.

Los comportamientos inseguros del wrapper actual se documentan con
`xfail(strict=True)`: el test describe el comportamiento DESEADO y falla hoy.
Si alguien corrige el wrapper, el XPASS estricto avisará de que hay que
quitar el xfail. No se corrige nada aquí (el wrapper no se modifica).
"""

import math

import numpy as np
import pytest
from scipy import stats

import privacy_layer
from governor import GovernanceDecision
from input_privacy_layer import FEATURE_BOUNDS
from validacion import analitica
from validacion.mecanismo import (
    N_FEATURES,
    construir_capa_entrada,
    epsilon_salida,
    generador_salida,
    registro_base,
    ruido_entrada_instrumentado,
)

SEED = 20240601
FEATURES = list(FEATURE_BOUNDS)
V_FEATURES = [f for f in FEATURES if f.startswith("V")]


def ci_binomial(k: int, n: int, nivel: float = 0.999):
    """Intervalo exacto de Clopper-Pearson (bilateral) para una proporción."""
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=nivel, method="exact")
    return ci.low, ci.high


def ruido_entrada(eps_in: float, n_llamadas: int, seed: int = SEED):
    """(escalas, ruidos) por feature, ruido observado ANTES del segundo recorte."""
    capa = construir_capa_entrada(seed)
    reg = registro_base()
    with ruido_entrada_instrumentado(capa) as cap:
        for _ in range(n_llamadas):
            capa.apply(reg, eps_in)
    escalas = np.array([c[0] for c in cap]).reshape(n_llamadas, -1)
    ruidos = np.array([c[1] for c in cap]).reshape(n_llamadas, -1)
    return escalas, ruidos


def ruido_salida(eps_out: float, n_llamadas: int, seed: int = SEED):
    cap = []
    with epsilon_salida(eps_out), generador_salida(seed, captura=cap):
        for _ in range(n_llamadas):
            privacy_layer.dp_layer.apply_dp_to_probability(0.5)
    return np.array([c[0] for c in cap]), np.array([c[1] for c in cap])


# --------------------------------------------------------------------------
# 1. Escala del ruido
# --------------------------------------------------------------------------

@pytest.mark.parametrize("eps_in", [1.0, 15.0, 45.0])
def test_escala_ruido_entrada(eps_in):
    n_llamadas = 1000
    escalas, ruidos = ruido_entrada(eps_in, n_llamadas)
    idx_v = [FEATURES.index(f) for f in V_FEATURES]
    idx_amount = FEATURES.index("Amount")

    # La escala pedida al generador es exactamente Δ·n/ε_in.
    assert np.allclose(escalas[:, idx_v], 10.0 * N_FEATURES / eps_in)
    assert np.allclose(escalas[:, idx_amount], 25000.0 * N_FEATURES / eps_in)

    # EMV de b para Laplace(0, b): b̂ = media(|ruido|). Con M observaciones su
    # error estándar relativo es 1/√M; tolerancia 5·EE.
    muestra = ruidos[:, idx_v].ravel()
    b_hat = np.mean(np.abs(muestra))
    b_teorico = analitica.escala_entrada(eps_in)
    tol = 5.0 / math.sqrt(muestra.size)
    assert b_hat == pytest.approx(b_teorico, rel=tol)

    # Bondad de ajuste: no se rechaza Laplace(0, b) (p > 0.001, semilla fija).
    assert stats.kstest(muestra, "laplace", args=(0.0, b_teorico)).pvalue > 1e-3


@pytest.mark.parametrize("eps_out", [0.5, 1.0, 2.0, 5.0])
def test_escala_ruido_salida(eps_out):
    n = 20000
    escalas, ruidos = ruido_salida(eps_out, n)
    assert np.allclose(escalas, 1.0 / eps_out)
    b_hat = np.mean(np.abs(ruidos))
    assert b_hat == pytest.approx(analitica.escala_salida(eps_out), rel=5.0 / math.sqrt(n))
    assert stats.kstest(ruidos, "laplace", args=(0.0, 1.0 / eps_out)).pvalue > 1e-3


# --------------------------------------------------------------------------
# 2. Masa en los límites (comprueba el recorte posterior)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("v", [3.0, -4.0, 0.0])
def test_masa_en_limites_entrada(v):
    eps_in, n = 30.0, 4000  # b = 5
    b = analitica.escala_entrada(eps_in)
    low, high = FEATURE_BOUNDS["V4"]
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "V4": v}
    salidas = np.array([capa.apply(reg, eps_in)["V4"] for _ in range(n)])
    for valor, p_teo in (
        (high, analitica.prob_limite_alto(v, b, high)),
        (low, analitica.prob_limite_bajo(v, b, low)),
    ):
        k = int(np.count_nonzero(salidas == valor))
        lo, hi = ci_binomial(k, n)
        assert lo <= p_teo <= hi, (valor, k / n, p_teo)


@pytest.mark.parametrize("v", [0.3, 0.9])
def test_masa_en_limites_salida(v):
    eps_out, n = 2.0, 6000  # b = 0.5
    b = analitica.escala_salida(eps_out)
    with epsilon_salida(eps_out), generador_salida(SEED):
        salidas = np.array([privacy_layer.dp_layer.apply_dp_to_probability(v) for _ in range(n)])
    for valor, p_teo in ((1.0, analitica.prob_limite_alto(v, b, 1.0)),
                         (0.0, analitica.prob_limite_bajo(v, b, 0.0))):
        k = int(np.count_nonzero(salidas == valor))
        lo, hi = ci_binomial(k, n)
        assert lo <= p_teo <= hi, (valor, k / n, p_teo)


# --------------------------------------------------------------------------
# 3. Reparto de ε
# --------------------------------------------------------------------------

def test_reparto_epsilon_uniforme():
    capa = construir_capa_entrada(SEED)
    eps_in = 6.0
    _, detalle = capa.apply_with_detail(registro_base(), eps_in)
    assert len(detalle) == len(FEATURE_BOUNDS) == N_FEATURES == 15
    assert all(d["epsilon_used"] == pytest.approx(eps_in / 15) for d in detalle)
    assert sum(d["epsilon_used"] for d in detalle) == pytest.approx(eps_in)
    assert {d["feature"] for d in detalle} == set(FEATURE_BOUNDS)


def test_n_es_el_numero_de_features_presentes():
    # Documentado: si faltan features, n baja y cada feature presente recibe MÁS
    # ε (y menos ruido); la suma sigue siendo ε_in (composición secuencial).
    capa = construir_capa_entrada(SEED)
    parcial = {"V1": 0.0, "V2": 0.0, "Amount": 10.0, "Time": 5.0}
    _, detalle = capa.apply_with_detail(parcial, 3.0)
    assert [d["feature"] for d in detalle] == ["V1", "V2", "Amount"]
    assert all(d["epsilon_used"] == pytest.approx(1.0) for d in detalle)


# --------------------------------------------------------------------------
# 4. Features sin ruido
# --------------------------------------------------------------------------

def test_time_y_claves_extra_sin_ruido():
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "Time": 123456.789, "otra_clave": "x"}
    _, detalle = capa.apply_with_detail(reg, 1.0)
    for _ in range(200):
        salida = capa.apply(reg, 1.0)
        assert salida["Time"] == reg["Time"]
        assert salida["otra_clave"] == "x"
    assert "Time" not in {d["feature"] for d in detalle}
    assert "Time" not in FEATURE_BOUNDS


# --------------------------------------------------------------------------
# 5. Recorte previo y entradas patológicas
# --------------------------------------------------------------------------

@pytest.mark.parametrize("valor,limite", [(100.0, 5.0), (-100.0, -5.0),
                                          (float("inf"), 5.0), (float("-inf"), -5.0)])
def test_fuera_de_rango_se_recorta_antes_del_ruido(valor, limite):
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "V4": valor}
    _, detalle = capa.apply_with_detail(reg, 15.0)
    d = next(x for x in detalle if x["feature"] == "V4")
    assert d["clipped"] == limite
    # Tras recortar, la masa en el límite es ½ (ruido simétrico), no ~1: el valor
    # absurdo no se "arrastra" a la salida.
    n = 3000
    salidas = [capa.apply(reg, 15.0)["V4"] for _ in range(n)]
    assert all(-5.0 <= s <= 5.0 for s in salidas)
    k = sum(1 for s in salidas if s == limite)
    lo, hi = ci_binomial(k, n)
    assert lo <= analitica.prob_limite_alto(abs(limite), 10.0, 5.0) <= hi


def test_nan_en_entrada_se_propaga_documentado():
    capa = construir_capa_entrada(SEED)
    salida = capa.apply({**registro_base(), "V4": float("nan")}, 15.0)
    assert math.isnan(salida["V4"])


@pytest.mark.xfail(strict=True, reason="NaN en una feature se propaga a la salida del input "
                   "shield (np.clip no lo sanea); debería rechazarse o saturarse.")
def test_nan_en_entrada_deberia_sanearse():
    capa = construir_capa_entrada(SEED)
    try:
        salida = capa.apply({**registro_base(), "V4": float("nan")}, 15.0)
    except ValueError:
        return
    assert math.isfinite(salida["V4"])


def test_valor_no_numerico_lanza_typeerror():
    capa = construir_capa_entrada(SEED)
    with pytest.raises(TypeError):
        capa.apply({**registro_base(), "V4": "abc"}, 1.0)


@pytest.mark.xfail(strict=True, reason="NaN como fraud_probability se propaga a la salida del "
                   "output shield (np.clip no lo sanea).")
def test_nan_en_salida_deberia_sanearse():
    with generador_salida(SEED):
        valor = privacy_layer.dp_layer._apply_dp(float("nan"), 0.0, 1.0, 1.0)
    assert math.isfinite(valor)


@pytest.mark.xfail(strict=True, reason="_apply_dp captura cualquier excepción y devuelve el valor "
                   "CRUDO (fail-open): un fallo del generador filtra la probabilidad sin ruido.")
def test_fallo_interno_no_deberia_devolver_valor_crudo():
    def laplace_roto(*_args, **_kwargs):
        raise RuntimeError("generador roto")

    from unittest import mock

    with mock.patch.object(np.random, "laplace", laplace_roto):
        try:
            salida = privacy_layer.dp_layer._apply_dp(0.37, 0.0, 1.0, 1.0)
        except RuntimeError:
            return
    assert salida != 0.37


# --------------------------------------------------------------------------
# 6. ε ≤ 0 y ε muy grande
# --------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.0, -1.0])
@pytest.mark.xfail(strict=True, reason="ε_in ≤ 0 devuelve las features CRUDAS sin ruido "
                   "(fail-open); debería rechazar la petición.")
def test_eps_entrada_no_positivo_no_deberia_devolver_crudo(eps):
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "V4": 1.2345}
    try:
        salida = capa.apply(reg, eps)
    except ValueError:
        return
    assert salida["V4"] != reg["V4"]


@pytest.mark.parametrize("eps", [0.0, -1.0])
@pytest.mark.xfail(strict=True, reason="ε_out ≤ 0 devuelve la probabilidad CRUDA sin ruido "
                   "(fail-open); debería rechazar la petición.")
def test_eps_salida_no_positivo_no_deberia_devolver_crudo(eps):
    with generador_salida(SEED):
        try:
            salida = privacy_layer.dp_layer._apply_dp(0.37, 0.0, 1.0, eps)
        except ValueError:
            return
    assert salida != 0.37


def test_eps_no_positivo_comportamiento_actual_documentado():
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "V4": 1.2345}
    for eps in (0.0, -5.0):
        salida, detalle = capa.apply_with_detail(reg, eps)
        assert salida == reg and detalle == []
        assert privacy_layer.dp_layer._apply_dp(0.37, 0.0, 1.0, eps) == 0.37


def test_eps_muy_grande_equivale_a_no_poner_ruido():
    capa = construir_capa_entrada(SEED)
    reg = {**registro_base(), "V4": 1.2345}
    salida = capa.apply(reg, 1e12)
    assert salida["V4"] == pytest.approx(1.2345, abs=1e-6)
    assert capa.apply(reg, float("inf"))["V4"] == 1.2345
    with generador_salida(SEED):
        assert privacy_layer.dp_layer._apply_dp(0.37, 0.0, 1.0, 1e12) == pytest.approx(0.37, abs=1e-6)


@pytest.mark.xfail(strict=True, reason="ε = NaN produce salida NaN en lugar de ser rechazado.")
def test_eps_nan_deberia_rechazarse():
    capa = construir_capa_entrada(SEED)
    try:
        salida = capa.apply(registro_base(), float("nan"))
    except ValueError:
        return
    assert math.isfinite(salida["V4"])


# --------------------------------------------------------------------------
# 7. Salida en rango
# --------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1e-3, 0.5, 2.0, 100.0])
@pytest.mark.parametrize("valor", [0.0, 0.3, 1.0])
def test_salida_siempre_en_unidad(eps, valor):
    with epsilon_salida(eps), generador_salida(SEED):
        salidas = [privacy_layer.dp_layer.apply_dp_to_probability(valor) for _ in range(2000)]
    assert min(salidas) >= 0.0 and max(salidas) <= 1.0


def test_apply_dp_to_response_en_rango():
    decision = GovernanceDecision(mechanism="laplace", epsilon=0.5, delta=1e-6,
                                  budget_allowed=True, epsilon_in=0.5, epsilon_out=0.8)
    with generador_salida(SEED):
        for _ in range(500):
            resp = privacy_layer.dp_layer.apply_dp_to_response(
                {"fraud_probability": 0.3, "confidence_score": 0.7, "is_fraud": 0, "message": "m"},
                decision=decision,
            )
            assert 0.0 <= resp["fraud_probability"] <= 1.0
            assert 0.0 <= resp["confidence_score"] <= 1.0


def test_entrada_siempre_dentro_de_bounds():
    capa = construir_capa_entrada(SEED)
    reg = registro_base()
    for _ in range(300):
        salida = capa.apply(reg, 1.0)
        for f, (low, high) in FEATURE_BOUNDS.items():
            assert low <= salida[f] <= high


# --------------------------------------------------------------------------
# Infraestructura: los parches se restauran siempre
# --------------------------------------------------------------------------

def test_parches_se_restauran_incluso_ante_excepcion():
    original = np.random.laplace
    capa = construir_capa_entrada(SEED)
    with pytest.raises(RuntimeError):
        with generador_salida(SEED), epsilon_salida(7.0), ruido_entrada_instrumentado(capa):
            assert "_noise" in capa.__dict__
            raise RuntimeError("fallo a propósito")
    assert np.random.laplace is original
    assert "_noise" not in capa.__dict__
    assert privacy_layer.EPSILON_FRAUD_PROBABILITY == 2.0
