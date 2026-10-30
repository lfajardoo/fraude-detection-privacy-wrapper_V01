"""Baseline externo: respuesta aleatorizada (diffprivlib.mechanisms.Binary)."""

import math

import numpy as np
import pytest

pytest.importorskip("diffprivlib")

from validacion import baseline_rr, distinguibilidad  # noqa: E402

T = 2000
ALPHA = 0.01
SEED = 11


def _frecuencia_publicar(decision, epsilon, n, seed):
    rs = np.random.RandomState(seed)
    return np.mean([baseline_rr.voltear(decision, epsilon, rs) for _ in range(n)])


@pytest.mark.parametrize("eps", [0.5, 1.0, 3.0])
def test_prob_acierto_coincide_con_frecuencia_empirica(eps):
    n = 20000
    freq = _frecuencia_publicar(True, eps, n, SEED)  # no-volteo = publicar True dado True
    tol = 5 * math.sqrt(0.25 / n)  # 5 desviaciones estándar binomiales
    assert freq == pytest.approx(baseline_rr.prob_acierto(eps), abs=tol)
    assert baseline_rr.epsilon_teorico_rr(eps) == eps


def test_prob_acierto_limites():
    assert baseline_rr.prob_acierto(0.0) == 0.5
    assert baseline_rr.prob_acierto(30.0) > 1 - 1e-12
    assert baseline_rr.prob_acierto(1e30) == 1.0
    assert baseline_rr.prob_acierto(1.0) == pytest.approx(math.e / (1 + math.e))


def test_epsilon_cero_es_moneda_justa():
    n = 20000
    freq = _frecuencia_publicar(True, 0.0, n, SEED)
    assert freq == pytest.approx(0.5, abs=5 * math.sqrt(0.25 / n))


def test_mecanismo_simetrico():
    eps, n = 1.0, 20000
    p_1_dado_0 = _frecuencia_publicar(False, eps, n, SEED)
    p_0_dado_1 = 1.0 - _frecuencia_publicar(True, eps, n, SEED + 1)
    esperado = baseline_rr.prob_volteo(eps)
    tol = 5 * math.sqrt(0.25 / n)
    assert p_1_dado_0 == pytest.approx(esperado, abs=tol)
    assert p_0_dado_1 == pytest.approx(esperado, abs=tol)
    assert p_1_dado_0 == pytest.approx(p_0_dado_1, abs=2 * tol)


def test_voltear_con_semilla_entera_es_pura():
    assert [baseline_rr.voltear(True, 1.0, 5) for _ in range(5)] == [baseline_rr.voltear(True, 1.0, 5)] * 5


@pytest.mark.parametrize("eps", baseline_rr.BARRIDO_EPS_RR)
def test_auditoria_no_supera_eps_teorico(eps):
    fila = baseline_rr.auditar_rr(eps, T, ALPHA, SEED)
    assert fila["eps_lb"] <= fila["eps_teorico"]
    assert fila["veredicto"] == distinguibilidad.COMPATIBLE
    # La auditoría recupera una fracción apreciable de ε (calibración del instrumento).
    assert fila["eps_lb"] > 0.5 * eps
    assert fila["p0_teorico"] == pytest.approx(baseline_rr.prob_volteo(eps))


@pytest.mark.parametrize("eps", baseline_rr.BARRIDO_EPS_RR)
def test_error_inyectado_se_detecta(eps):
    fila = distinguibilidad.control_error_inyectado(
        baseline_rr.auditar_rr, {"epsilon": eps, "T": T, "alpha": ALPHA, "seed": SEED}
    )
    assert fila["eps_lb"] > eps
    assert fila["veredicto"] == distinguibilidad.VIOLACION
    assert fila["control_ok"]


def test_control_sin_ruido_alcanza_el_techo():
    fila = baseline_rr.control_sin_ruido_rr(T, ALPHA, SEED)
    assert fila["k0"] == 0 and fila["k1"] == T
    assert fila["eps_lb"] == pytest.approx(fila["techo"])
    assert fila["control_ok"]


def test_ejecutar_auditoria_rr_controles_ok():
    res = baseline_rr.ejecutar_auditoria_rr(T, ALPHA, SEED)
    assert res["controles_ok"]


def test_fila_tiene_las_claves_del_nivel3():
    rr = baseline_rr.auditar_rr(1.0, 500, ALPHA, SEED)
    sal = distinguibilidad.auditar_salida(1.0, 500, ALPHA, SEED)
    assert set(sal) <= set(rr)


def test_determinismo():
    a = baseline_rr.auditar_rr(1.0, T, ALPHA, SEED)
    b = baseline_rr.auditar_rr(1.0, T, ALPHA, SEED)
    for k in a:
        if k != "tiempo_s":
            assert a[k] == b[k]
    assert baseline_rr.auditar_rr(1.0, T, ALPHA, SEED + 1)["k0"] != a["k0"]


def test_utilidad_wrapper_simulada_coincide_con_forma_cerrada():
    for eps in (0.5, 1.0, 2.0):
        sim = baseline_rr.utilidad_wrapper(eps, n=4000, seed=SEED)
        assert sim == pytest.approx(baseline_rr.utilidad_wrapper_teorica(eps), abs=0.01)


def test_utilidad_wrapper_usa_la_capa_real_con_epsilon_dado():
    # ε grande → casi sin ruido → decisión casi siempre correcta
    assert baseline_rr.utilidad_wrapper(1e6, n=500, seed=SEED) > 0.999
    # ε = 0.5 es peor que ε = 2
    assert baseline_rr.utilidad_wrapper(0.5, n=2000, seed=SEED) < baseline_rr.utilidad_wrapper(2.0, n=2000, seed=SEED)


def test_comparacion_columnas_y_valores():
    filas = baseline_rr.comparacion_utilidad((1.0,), T, ALPHA, SEED, n_utilidad=1000)
    f = filas[0]
    assert f["rr_prob_acierto"] == pytest.approx(0.7311, abs=1e-4)
    assert f["wrapper_in_b_relativo"] == pytest.approx(15.0)
    for col in ("epsilon", "rr_prob_acierto", "wrapper_out_prob_acierto", "wrapper_in_b_relativo",
                "rr_eps_lb", "wrapper_eps_lb"):
        assert col in f
