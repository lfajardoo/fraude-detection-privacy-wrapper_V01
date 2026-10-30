"""Nivel 2: fórmulas cerradas y su contraste con Monte Carlo sobre la capa real."""

import math

import numpy as np
import pytest

from validacion import analitica
from validacion.mecanismo import construir_capa_entrada, registro_base

SEED = 7


def test_escalas_valores_conocidos():
    assert analitica.escala_entrada(1.0) == pytest.approx(150.0)
    assert analitica.escala_entrada(15.0) == pytest.approx(10.0)
    assert analitica.escala_salida(2.0) == pytest.approx(0.5)
    assert analitica.escala_salida(0.5) == pytest.approx(2.0)
    # Amount: Δ = 25 000
    assert analitica.escala_entrada(1.0, delta=25000.0) == pytest.approx(375000.0)


@pytest.mark.parametrize("f", [analitica.escala_entrada, analitica.escala_salida])
def test_escalas_rechazan_eps_no_positivo(f):
    with pytest.raises(ValueError):
        f(0.0)


def test_garantia_por_consulta():
    # vecindad "registro": cambia todo el registro → ε_in; min con ε_out
    assert analitica.garantia_por_consulta(1.0, 2.0) == 1.0
    assert analitica.garantia_por_consulta(45.0, 2.0) == 2.0
    # vecindad "una_feature": ε_in / n
    assert analitica.garantia_por_consulta(15.0, 2.0, vecindad="una_feature") == pytest.approx(1.0)
    assert analitica.garantia_por_consulta(45.0, 2.0, vecindad="una_feature") == pytest.approx(2.0)
    # dos salidas publicadas (probabilidad + confianza): el lado de salida compone
    assert analitica.garantia_por_consulta(45.0, 2.0, n_salidas=2) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        analitica.garantia_por_consulta(1.0, 1.0, vecindad="otra")


def test_composicion_basica():
    assert analitica.composicion_basica(0.5, 10) == pytest.approx(5.0)


def test_atenuacion_valor_conocido():
    a = analitica.atenuacion(0.0, 150.0)
    assert a == pytest.approx(0.033, abs=5e-4)
    assert a == pytest.approx(1.0 - math.exp(-5.0 / 150.0))
    assert analitica.consultas_necesarias(a) == pytest.approx(1.0 / a**2)
    assert math.isinf(analitica.consultas_necesarias(0.0))


def test_atenuacion_simetrica_y_acotada():
    assert analitica.atenuacion(2.0, 10.0) == pytest.approx(analitica.atenuacion(-2.0, 10.0))
    assert 0.0 < analitica.atenuacion(5.0, 1.0) < analitica.atenuacion(0.0, 1.0) <= 1.0


@pytest.mark.parametrize("eps_in,v,n", [(30.0, 0.0, 6000), (15.0, 4.0, 6000), (1.0, 0.0, 20000)])
def test_atenuacion_vs_monte_carlo_capa_real(eps_in, v, n):
    """Pendiente de E[salida] respecto a la entrada, medida en la capa real.

    Números aleatorios comunes (misma semilla en ambas capas) => el ruido es
    idéntico en v+h y v−h, y la diferencia de salidas vale 2h solo cuando
    ninguno de los dos se recorta: la media es la pendiente. Error estándar
    ≈ √(a(1−a)/n); tolerancia 5·EE más el sesgo de h finito (h/b·a).
    """
    h = 0.5
    b = analitica.escala_entrada(eps_in)
    a_teorico = analitica.atenuacion(v, b)
    capa_hi, capa_lo = construir_capa_entrada(SEED), construir_capa_entrada(SEED)
    base = registro_base()
    dif = np.array([
        capa_hi.apply({**base, "V4": v + h}, eps_in)["V4"]
        - capa_lo.apply({**base, "V4": v - h}, eps_in)["V4"]
        for _ in range(n)
    ])
    a_mc = dif.mean() / (2.0 * h)
    tol = 5.0 * math.sqrt(a_teorico * (1.0 - a_teorico) / n) + h / b * a_teorico
    assert abs(a_mc - a_teorico) < tol, (a_mc, a_teorico, tol)
