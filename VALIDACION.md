# Validación del privacy wrapper

Se valida el **privacy wrapper** (`privacy-api/`), no el modelo de fraude. La validación mide
directamente la propiedad que importa —*¿entradas vecinas son distinguibles a la salida?*— en
tres niveles; los dos primeros no dependen del modelo ni de la API.

> **Principio rector** (NIST SP 800-226 §2.7; Jagielski et al., 2020):
> *lo analítico prueba, lo empírico solo refuta.* El Nivel 2 da la garantía (cota superior de ε);
> el Nivel 3 da una cota inferior empírica que debe ser compatible con ella.

## Qué valida cada nivel

| Nivel | Qué es | Dónde | Coste |
|---|---|---|---|
| 1 | Tests del mecanismo: escala del ruido, masa en los límites (segundo recorte), reparto de ε, Time sin ruido, entradas patológicas, salida en rango | `privacy-api/tests/test_nivel1_mecanismo.py` | segundos |
| 2 | Predicción analítica: escalas `b`, garantía por consulta, composición, atenuación y coste del atacante; contraste con Monte Carlo sobre la capa real | `privacy-api/validacion/analitica.py`, `tests/test_nivel2_analitica.py` | segundos |
| 3 | Auditoría empírica de distinguibilidad: cota inferior ε_LB con Clopper-Pearson, más tres controles | `privacy-api/validacion/distinguibilidad.py` | ~5 s (T=2000) |

Las capas bajo prueba **no se modifican**: el control del ruido (semilla, observar el ruido
antes del segundo recorte, inyectar escala b/2) se hace desde fuera con context managers de
`validacion/mecanismo.py` que sustituyen temporalmente `np.random.laplace` / `_noise` y siempre
restauran el original.

### Definiciones usadas

- **Input shield:** `scale = Δ·n/ε_in` (Δ = ancho del rango: 10 para V1…V14, 25 000 para Amount;
  n = 15 features en `FEATURE_BOUNDS`; Time no se perturba). Orden: recorte → ruido → recorte.
- **Output shield:** `scale = 1/ε_out` (Δ = 1), recorte a [0, 1].
- **Vecindad** (`analitica.garantia_por_consulta`):
  - `"registro"` (por defecto): los vecinos pueden diferir en *todas* las features → ε_entrada = ε_in.
  - `"una_feature"`: difieren en una sola feature → ε_entrada = ε_in / n.
  - Garantía por consulta = `min(ε_entrada, n_salidas·ε_out)`. Si se publican a la vez
    `fraud_probability` y `confidence_score` con ruido independiente, el lado de salida compone
    (`n_salidas=2`); la cota `ε_entrada` no se ve afectada (post-procesamiento).
- **Composición básica** de k consultas: `k·ε`. El wrapper **no acumula presupuesto** hoy.

## Cómo correrlo

Desde `privacy-api/`:

```bash
pip install -r requirements-dev.txt
pytest                                   # toda la batería (incluye tests.py y test_input_privacy.py)
pytest tests/                            # solo niveles 1 y 2
python -m validacion.run_validacion --nivel todos --T 2000 --alpha 0.01 --seed 1 \
       --feature V4 --out results/validacion/
```

- `--nivel {2,3,todos}`, `--feature` acepta cualquier clave de `FEATURE_BOUNDS`,
  `--eps-in 1,15,45` y `--eps-out 0.5,1,2` cambian el barrido.
- Una ruta `--out` relativa se interpreta respecto a la **raíz del repositorio**
  (salida por defecto: `results/validacion/`).
- Salidas: `nivel2_analitica.{json,csv}`, `nivel2_garantias.csv`, `nivel3_distinguibilidad.{json,csv}`.
- Código de salida `1` si hay `VIOLACIÓN` o un control falla (hallazgo, no se ajusta el test).
- Opcional extremo a extremo (el **modelo entra en la cadena**; el servidor no está sembrado, así
  que no es reproducible): levantar modelo + API y añadir `--e2e-url http://localhost:8001`
  (`mode=legacy`, `noise_repetitions=1`, evento `is_fraud == 1`, ambos sentidos con α/2 cada uno).
  Por post-procesamiento ε_LB no puede superar `min(ε_in/n, ε_out)`.

## Nivel 3: método

Dos entradas vecinas, T consultas cada una, un evento fijado a priori y

```
ε_LB = ln( CP_inf(p̂₁) / CP_sup(p̂₀) )
```

con intervalos de Clopper-Pearson **unilaterales** (`scipy.stats.beta`) y α repartido α/2 por
proporción. Con probabilidad ≥ 1 − α, ε_LB ≤ ε real. No hay corrección por multiplicidad entre
las auditorías del barrido (cada una tiene probabilidad ≤ α de falsa violación).

- **Entrada:** `feature` (por defecto V4) en `low` vs `high`, resto del registro fijo; evento
  `salida_feature == high`. Cociente teórico exacto `e^{ε_in/n}` (`p₀ = ½e^{−ε_f}`, `p₁ = ½`).
- **Salida:** probabilidades 0 vs 1 (Δ = 1) con `apply_dp_to_probability`; evento `salida == 1`.
  Cociente teórico `e^{ε_out}`.
- **Techo:** el máximo ε_LB alcanzable es el de (k₀=0, k₁=T): `ln(a^{1/T}/(1−a^{1/T}))`, `a=α/2`.
  T=500, α=0.01 → 4.54; **T=2000, α=0.01 → 5.93**. `run_validacion` aborta si algún ε del barrido
  queda por encima del techo (hay que subir T).

### Cómo leer el veredicto

| Veredicto | Significa |
|---|---|
| `compatible` | ε_LB ≤ ε teórico. **No prueba nada**: solo que no se ha refutado la garantía. |
| `VIOLACIÓN` | ε_LB > ε teórico. El mecanismo no cumple su garantía: es un hallazgo. |
| `sin potencia` | ε teórico ≥ techo: la auditoría no puede resolverlo con este T. |

Mire también `eps_lb_esperado` y `resolucion` (= `eps_lb_esperado/eps_teorico`): son el ε_LB que
se obtendría con los conteos esperados si el mecanismo fuera exactamente el nominal. Si es ≈ 0,
`compatible` es poco informativo (con T=2000, ε_in=1 → ε por feature = 0.067 no es resoluble).

### Controles obligatorios

1. **Sin ruido** (ε = 1e30): ε_LB debe igualar el techo → el test tiene potencia. Con ε "grande
   pero finito" (p. ej. 1e6) el ruido no desaparece por redondeo y `p₁` queda en ½ (la mitad del
   ruido cae por debajo de `high`), por lo que ε_LB queda *por debajo* del techo; con 1e30 el ruido
   es inferior a la resolución de punto flotante y `high + ruido == high` siempre.
2. **Error inyectado** (escala b/2): ε_LB debe superar el ε teórico. Solo se exige cuando el análisis
   de potencia a priori (`eps_lb_esperado`, con el ε real inyectado) predice que es detectable; si no,
   se marca `sin potencia` y no cuenta como fallo.
3. **Mecanismo correcto:** ε_LB ≤ ε teórico. Si lo supera, es un hallazgo.

### Resultado de referencia (`--T 2000 --alpha 0.01 --seed 1 --feature V4`)

| Capa | ε teórico | ε_LB (correcto) | ε_LB con b/2 | Veredicto |
|---|---|---|---|---|
| entrada (ε_in=1 → 0.067/feature) | 0.067 | 0.000 | 0.031 (no detectable) | compatible |
| entrada (ε_in=15 → 1/feature) | 1 | 0.830 | 1.746 | compatible; bug detectado |
| entrada (ε_in=45 → 3/feature) | 3 | 2.555 | 5.179 | compatible; bug detectado |
| salida (ε_out=0.5) | 0.5 | 0.329 | 0.853 | compatible; bug detectado |
| salida (ε_out=1) | 1 | 0.706 | 1.621 | compatible; bug detectado |
| salida (ε_out=2) | 2 | 1.703 | 3.328 | compatible; bug detectado |
| sin ruido (ambas capas) | — | 5.932 = techo | — | potencia confirmada |
| e2e `legacy` (ε_in=45, ε_out=2) | 2 | 0.184 | — | compatible |

La corrida e2e usó el servicio de modelo que respondía en `localhost:8000` (modelo de ejemplo,
con poca sensibilidad a V4); es solo ilustrativa.

## Qué NO valida

- **No es una garantía sobre los datos de entrenamiento** ni sobre el modelo: valida el ruido que
  el wrapper añade a una consulta, no la privacidad del entrenamiento (DP-SGD, etc.).
- Las pruebas empíricas **no prueban DP**: solo pueden refutar o dejar de refutar. Una auditoría
  `compatible` con ε_LB ≪ ε no certifica nada.
- El Nivel 3 audita un único par de vecinos extremos y un único evento por capa; otros pares o
  eventos podrían (en un mecanismo defectuoso) distinguir más.
- Solo se cubre el mecanismo **Laplace**; el Gaussian no se valida.
- La aritmética de punto flotante de `np.random.laplace` no está cubierta (Mironov, 2012).
- No hay corrección por multiplicidad entre auditorías del barrido.

## Hallazgos del Nivel 1 (marcados `xfail`, no corregidos aquí)

Cada `xfail(strict=True)` describe el comportamiento *deseado* y falla hoy; si se corrige el
wrapper el XPASS estricto avisará de que hay que quitar la marca.

- `ε_in ≤ 0` y `ε_out ≤ 0` devuelven el valor **crudo** sin ruido (fail-open).
- `_apply_dp` captura cualquier excepción y devuelve el valor crudo (un fallo del generador filtra la probabilidad).
- `NaN` en una feature o en la probabilidad se propaga a la salida; `ε = NaN` produce salida NaN.
- Documentado (no es fallo): valores fuera de rango o `±inf` se recortan antes del ruido; si faltan
  features, `n` baja y cada feature presente recibe más ε (la suma sigue siendo ε_in).

## Trabajo pendiente (fuera de alcance, no implementado)

- Sustituir `np.random.laplace` por un mecanismo seguro frente a punto flotante (OpenDP,
  diffprivlib) o aplicar la mitigación de Mironov.
- Endurecer el endpoint: `EVALUATION_MODE = False`, quitar `ground_truth` y `perturbation_detail`,
  ignorar overrides de ε y `mode=raw` fuera del laboratorio, que la política P4 rechace en vez de
  devolver el valor crudo, y revisar `epsilon_prob = decision.epsilon if decision.epsilon_in is None else decision.epsilon_in`
  en `privacy_layer.apply_dp_to_response` (en modo `governed` el ε de *entrada* de la política se
  aplica al ruido de la probabilidad de *salida*).
- Acumulador de presupuesto por cliente / rate limit (hoy `budget_allowed` no acumula).

## Caso de estudio: ataque de inversión

La validación anterior era un ataque de inversión sobre V4 (≈710 líneas, ≈10 h por experimento).
Su código está en el tag `v0-ataque-inversion` y sus resultados y documentación en
[`archivo/ataque_inversion/`](archivo/ataque_inversion/README.md). Son un caso de estudio
ilustrativo, no la validación principal. La atenuación `a` del Nivel 2
(`consultas_necesarias ~ 1/a²`) es el modelo de coste que explica por qué ese ataque necesita
tantas consultas.
