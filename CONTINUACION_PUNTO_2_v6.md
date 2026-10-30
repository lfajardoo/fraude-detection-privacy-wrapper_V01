# Continuación Punto 2 — Diagnóstico del `evaluator_attack.py` (v6)

> **Propósito.** (a) Material de repaso y (b) contexto autocontenido para abrir una
> consulta nueva **en modo incógnito**. Pegar tal cual y continuar sin reconstruir el hilo.
> La edición del proyecto se hace con **Claude Code**.
>
> **ESTADO: el Punto 2 está funcionando.** La curva de protección es limpia (§4.6).
> Quedan tareas de rigor estadístico (§7) y dos experimentos (§10), no depuración.
>
> **Cambios v5 → v6:** fix del SEM aplicado ✅ · umbral Bonferroni 5.0 ✅ · barrido
> completo con curva monótona ✅ · **hallazgo nuevo: varianza corrida a corrida de ±35%**
> (§4.7) — es el bloqueante principal para reportar resultados.

---

## 0. Contexto del proyecto (para quien lo lea desde cero)

- **Repo:** `fraude-detection-privacy-wrapper_V01`.
- **Objetivo general:** una `privacy-api` que envuelve un modelo de detección de fraude
  con dos capas de privacidad diferencial (DP):
  - **Input shield** (`input_privacy_layer.py`): ruido Laplace a las *features de entrada*
    antes de que el modelo prediga. Presupuesto `epsilon_in`, repartido entre 15 features
    perturbables (`V1..V14`, `Amount`; `Time` excluido).
  - **Output shield** (`privacy_layer.py`): ruido Laplace a la *probabilidad de salida*,
    recortada a `[0,1]`. Presupuesto `epsilon_out`.
- **`modelo-python/`**: modelo de prueba (Regresión Logística + StandardScaler sobre 16
  features: `Time, V1..V14, Amount`). **No se toca ni se evalúa en exceso**; es
  intercambiable. Entrenado con muestra balanceada (~500 fraudes + 5000 normales).
  `coef_[V4] = +1.270` (el mayor), `coef_[V14] = -1.861`.
- **Punto 2:** un **ataque de inversión inter-request** que reconstruye una feature
  secreta consultando la API muchas veces. El objetivo NO es "romper" la defensa, sino
  **medir la curva de protección** y su **piso irreducible**.

### Arquitectura de servicios
```
attacks/api_client.py  ──HTTP──▶  privacy-api:8001  ──HTTP──▶  modelo-financiero:8000
   (HttpAttackClient)              (app.py, shields)             (modelo-python)
```
Ambos saltos viajan como **dict con claves nombradas** (FastAPI valida por nombre). El
único punto donde se arma un vector posicional es `modelo-python/model.py:26-33`, que
itera `self.features` (= `features.pkl`, orden exacto de entrenamiento). **No hay
posibilidad de desorden de features** — verificado.

### Archivos relevantes
- `privacy-api/evaluator_attack.py` — **harness**: orquesta el experimento, parte el
  dataset en *pool público* vs *targets*, calcula probes, barre epsilons, compara contra
  el valor real (que el atacante NO ve) y calcula métricas. Incluye el bloque `[diag]`.
- `privacy-api/attacks/inversion_attack.py` — **atacante ciego**: `denoise()` (promedia K
  consultas) e `invert_feature()` (inversión lineal de 2 puntos en espacio logit + guard).
- `privacy-api/attacks/metrics_attack.py` — métricas (`build_curve`, MAE/RMSE, etc.).
- `privacy-api/attacks/api_client.py` — `HttpAttackClient`.
- `privacy-api/sanity_check.py` — verifica sin ruido ni promediado que mover una feature
  mueve la salida. **Herramienta de bisección; correr ante cualquier anomalía.**

### Glosario
- **Harness:** el código que orquesta y mide; no es el atacante ni la defensa ni el
  modelo. Tiene acceso al valor secreto real para computar el error; el atacante no.
- **Span de sondeo:** `probe_hi - probe_lo`, la distancia entre los dos valores de prueba.
- **R:** repeticiones del par de sondeos, para estimar la incertidumbre de la pendiente.
- **SNR del guard:** `|slope_mean| / SEM(slope)`. Si no supera el umbral, el atacante se
  **abstiene** y cae a `public_mean`.
- **Abstención:** el atacante declina estimar. Correcto cuando no hay señal; su efecto es
  que el MAE de esa fila queda **exactamente en la baseline**.

---

## 1. Cómo funciona el ataque (mecánica)

Sin ruido, la regresión logística cumple `logit(p) = w · x + b`: el logit es **lineal**
en las features (incluso en espacio crudo, porque el StandardScaler es afín).

`invert_feature` explota eso con una **inversión de 2 puntos**:
1. Feature secreta en `probe_lo`, promedia K consultas → `p_low` → `logit_low`.
2. Feature secreta en `probe_hi`, promedia K consultas → `p_high` → `logit_high`.
3. `slope = (logit_high - logit_low) / (probe_hi - probe_lo)`.
4. `intercept = logit_low - slope · probe_lo`.
5. `estimate = (logit(p_target) - intercept) / slope`.

`p_target = denoise(target, K)` es la probabilidad promedio del target con su valor real.

**La pendiente va en el DENOMINADOR.** Si `slope → 0`, la estimación explota. De ahí el
guard estadístico (§6).

**Costo:** cada `invert_feature` ≈ `2·K·R + 2·K` requests. Con K=400, R=8, 30 targets ×
7 epsilons ≈ **1.5M requests**. Para calibrar usar `--K 50` y `--epsilons-in 500`.

---

## 2. 🔴 CAUSA RAÍZ HISTÓRICA (resuelta — no repetir la depuración)

> Esta sección explica por qué todo fallaba antes. **Ya está arreglado.** Se conserva
> porque reinterpreta los resultados viejos y porque la lección metodológica importa.

### El síntoma
`sanity_check.py` barrió V4 con ruido desactivado (`epsilon=1e9`) contra
`privacy-api:8001`:
```
V4=-5 p=0.62   V4=-3 p=1.00   V4=-1 p=1.00   V4=0 p=0.56
V4= 1 p=0.00   V4= 3 p=1.00   V4= 4 p=0.42   V4=5 p=0.00
```
**Imposible para una regresión logística** (estrictamente monótona en cada feature).

### La bisección
El mismo barrido contra `modelo-financiero:8000` **directamente** salió monótono
(0.000626 → 0.229368) ⇒ el modelo estaba bien; el problema estaba en la `privacy-api`.

### El diagnóstico
La respuesta en modo `raw` mostró `"epsilon_in_total": 1.0` — se pedía `1e9` y lo
ignoraba. Con ε=1.0 entre 15 features → ε_por_feature = 0.067 → escala Laplace ≈ 150
para las V (rango 10) y ≈3.7e5 para `Amount`. **Ruido ≫ rango** ⇒ `np.clip` saturaba cada
feature en un extremo aleatorio (±5). **El modelo recibía ruido puro en las 15 features,
en cada request.**

### La causa
El contenedor `privacy-api` corría una **imagen Docker obsoleta** (anterior al 25-ago):
`Dockerfile` hace `COPY app.py .`; `docker-compose.yml` **no monta volumen** ni usa
`--reload`. Los cambios del host nunca llegaban al contenedor: la imagen tenía el ruido de
entrada (18-ago) pero **no el override de `epsilon_in`** (commit `990b153`, 25-ago), así
que el governor imponía su default hardcodeado (`governor.py:97`, `epsilon_in=1.0`).

### El fix (aplicado y verificado)
```bash
docker compose up -d --build privacy-api
```

### Reinterpretación de lo observado antes del fix
| Observación | Interpretación vieja | Interpretación correcta |
|---|---|---|
| Curva plana / 100% abstención | guard mal calibrado, span, ε_out | no había señal: entrada aleatorizada |
| Pendientes con signos mezclados | varianza tapando señal débil | lectura fiel de ruido puro |
| `slope_mean → 0` al subir K | convergencia a pendiente pequeña | convergencia a la media del ruido (=0) |
| No monotonía MAE(ε_in) | inestabilidad numérica del logit | artefacto del contenedor |
| Camino 1 (R 3→8) no ayudó | — | R no podía ayudar (bug del SEM, §6) |

**El guard nunca falló.** Se abstenía correctamente porque no había nada que recuperar.

### Lección metodológica (vale para la tesis)
Había desincronización **repo ↔ disco** (ya advertida) y una segunda capa: **disco ↔
contenedor en ejecución**. Antes de depurar un instrumento de medición, verificar que el
sistema bajo medición responde como su modelo teórico predice. Ese es el propósito de
`sanity_check.py`.

**Mejora de flujo pendiente (opcional):** montar `./privacy-api` como volumen en
`docker-compose.yml` y arrancar con `--reload`, para que host y contenedor no divergan.

---

## 3. ⚠️ Qué quedó INVALIDADO de versiones anteriores

| Sección vieja | Estado | Motivo |
|---|---|---|
| Problema A (barrer ε_in, no ε_out) | ✅ **VÁLIDO** | El eje informativo sigue siendo ε_in. |
| Problema B (falsos positivos del guard) | ✅ **VÁLIDO y CONFIRMADO** | Cuantificado en §4.5; resuelto con Bonferroni. |
| Problema C (sesgo de clipping ε_out) | ⚠️ **PARCIAL** | Mecanismo real, pero la premisa "ε_out=50 basta para p>0.01" **no se cumple**: las `p` reales van de 1e-6 a 1e-1. Con ε_out=50 la señal se comprimía ~10×. Se usa **ε_out=10000**. |
| Problema D (no monotonía por logit) | ❌ **INVALIDADO** | Observado con el contenedor roto. **Reescribir desde cero.** Ver §4.7: la no-monotonía residual es varianza de muestreo, no inestabilidad del logit. |
| Tabla de verificación del modelo réplica | ❌ **INVALIDADA** | Asumía pendiente 2.66 y `p~1e-8`; el modelo real tiene otro régimen. Descartar. |
| Camino 2 (span p25/p75) | ❌ **SIN JUSTIFICACIÓN** | Se adoptó para compensar el Problema D, inexistente. **Recalibrar** (§8). |
| Camino 1 (R=3→8) | ✅ **AHORA SÍ OPERANTE** | Con el fix del SEM (§6), subir R por fin mejora el SNR. |
| Resultados previos al rebuild | ❌ **OBSOLETOS** | Midieron ruido. Descartar todos. |
| Narrativa de los 3 experimentos | ✅ **VÁLIDA** | Sigue siendo el plan correcto (§10). |
| Restricciones del proyecto | ✅ **VÁLIDAS** | §11. |

---

## 4. ✅ Estado verificado actual

### 4.1 Sanity check — pipeline limpio
```
V4=-5 p=0.000266   V4=-1 p=0.003122   V4=2 p=0.019513   V4=4 p=0.063916
V4=-3 p=0.000912   V4= 0 p=0.005767   V4=3 p=0.035552   V4=5 p=0.112274
V4=-2 p=0.001688   V4= 1 p=0.010630
```
**Estrictamente monótono creciente.** Coincide con el barrido directo al modelo. ✅

### 4.2 Sensibilidad local por feature (target[0], ±2 alrededor del valor real)
```
p base = 0.007926
V14 |Δ|=0.030759   V13 |Δ|=0.009608   V6 |Δ|=0.008585   V11 |Δ|=0.003087
V12 |Δ|=0.026838   V8  |Δ|=0.008708   V7 |Δ|=0.004594
V10 |Δ|=0.014549
```
**V4 no aparece en el top 8**, pese a tener el mayor coeficiente global (`+1.270`).
El ataque mide la pendiente **local**, y este target vive en `p≈0.008`, donde V4 aporta
poco. La recuperabilidad depende de *dónde vive el target en el espacio de probabilidad*,
no solo del peso global del modelo. **Resultado reportable en la tesis.**

### 4.3 Comparación V4 vs V14 (ε_in=500, corrida única)
| secreto | baseline | MAE | MAE/baseline | reducción |
|---|---:|---:|---:|---:|
| V4  | 1.1389 | 0.2080 | 0.183 | **82%** |
| V14 | 0.7461 | 0.3058 | 0.410 | **59%** |

### 4.4 ⚠️ Comparabilidad de métricas entre features
El MAE está en **las unidades de cada feature**: 0.2080 y 0.3058 **no viven en la misma
escala**. La baseline lo delata (1.1389 vs 0.7461): es el MAE de adivinar la media
pública, así que aproxima la **dispersión** de la feature. V14 es intrínsecamente más
compacta ⇒ *cualquier* estimador tendrá menor MAE ahí.

**Métrica comparable = MAE/baseline** (o su complemento, la reducción %). En esa escala
**V4 se recupera más del doble de bien que V14**. Reportar la razón como métrica
principal; el MAE crudo como dato secundario.

Mismo problema con `success_rate`: `tol_abs` está definido como fracción de la std pública
(correcto), pero eso implica que el `success_rate` de dos features **no mide lo mismo en
unidades absolutas**. Aclararlo al reportar.

### 4.5 Por qué V14 abstiene más que V4 (contraintuitivo pero explicado)
V14 es **más sensible por unidad** (`|slope_mean|≈0.60` vs `0.53`) pero su distribución es
**mucho más compacta**: span p25/p75 = 0.502 vs 1.479 (2.9× menor). Como
`slope = Δlogit / span`, un span 3× menor amplifica el ruido de la pendiente 3×:

| | V4 | V14 | ratio |
|---|---:|---:|---:|
| span p25/p75 | 1.479 | 0.502 | 2.9 |
| `slope_std` medio | 0.106 | 0.341 | 3.2 |

Los ratios coinciden ⇒ **el span, no la sensibilidad, gobierna el SNR.** Motiva §8.2.

### 4.6 🎯 LA CURVA DE PROTECCIÓN (resultado principal del Punto 2)

```bash
python3 privacy-api/evaluator_attack.py --K 400 --limit 130 --public-size 100 \
  --secret V4 --epsilon-out 10000 --snr-threshold 5.0
```
```
[diag] ε_in=500, ε_out=10000, R=8, K=400
  target[0]  slope_mean=+0.5114  slope_std=0.1184  SNR=12.22  signos 8/0  [PASA]
  target[15] slope_mean=+0.6065  slope_std=0.0775  SNR=22.14  signos 8/0  [PASA]
  target[29] slope_mean=+0.5553  slope_std=0.0911  SNR=17.24  signos 8/0  [PASA]

linea_base MAE = 1.1389
epsilon_in   MAE_ataque   ¿gana?
       1       1.1389       no      ← abstención total (protección completa)
      10       1.1389       no      ← abstención total
      30       1.1530       no      ← +1.2% sobre baseline (ruido, ver §4.7)
      50       1.0395       SI
     100       0.7548       SI
     300       0.3999       SI
     500       0.2046       SI      ← -82% vs baseline
```

**Forma correcta:** plana en la baseline con protección fuerte, luego monótona
decreciente. La **rodilla** (donde el input shield deja de proteger) está entre ε_in=30
y 50.

Efecto del umbral Bonferroni en la región baja (misma configuración, solo cambia el
umbral):

| ε_in | umbral 2.0 | umbral 5.0 |
|---:|---:|---:|
| 1 | 1.3208 | **1.1389** ✅ |
| 10 | 1.1787 | **1.1389** ✅ |
| 30 | 1.4473 | 1.1530 |
| 50 | 1.1573 | 1.0395 |

Los falsos positivos eran la causa del desorden en la región baja. ✅

### 4.7 🔴 HALLAZGO NUEVO: varianza corrida a corrida (bloqueante para reportar)

En la región alta (ε_in ≥ 100) **todos los targets pasan el guard con SNR de 12–22**, muy
por encima de ambos umbrales. Ahí el umbral no puede cambiar nada. Y sin embargo:

| ε_in | corrida A | corrida B | diferencia |
|---:|---:|---:|---:|
| 100 | 0.5387 | 0.7548 | **+40%** |
| 300 | 0.3885 | 0.3999 | +3% |
| 500 | 0.3103 | 0.2046 | **-34%** |

Esas diferencias **no las explica el umbral**: son varianza corrida a corrida de **±35%**.
La baseline sí es determinista (1.1389 en ambas) ⇒ la partición del dataset es estable y
la varianza viene enteramente de la **estocasticidad del ataque** (ruido Laplace en cada
request).

**Consecuencias:**
1. **El bulto en ε_in=30 (+1.2%) casi seguro es ruido**, no un fenómeno. No explicarlo en
   la tesis; probablemente desaparece al repetir. (Prueba falsable: repetir con otra
   semilla; si el bulto se mueve de epsilon, es ruido.)
2. **La posición de la rodilla no se puede afirmar desde una sola corrida.** Y la rodilla
   es el número interesante: es donde el input shield deja de proteger.
3. **Ningún número de una corrida única debe ir a la tesis sin barras de error.**

---

## 5. `privacy-api/sanity_check.py` (archivo existente)

Herramienta de bisección. **Correr ante cualquier resultado anómalo, antes de tocar
hiperparámetros.** Barre una feature con `epsilon=1e9` en ambas capas, K=1 (sin
promediado), y ranquea la sensibilidad local de todas las features.

**Criterio de aceptación:** el barrido debe ser **monótono**. Si no lo es, hay un bug de
sistema — no seguir depurando el harness.

---

## 6. ✅ Fix del SEM + umbral Bonferroni (APLICADO)

### El bug (resuelto)
El guard comparaba `slope_mean` contra la **desviación estándar de las pendientes
individuales**. Lo correcto es el **error estándar**, `slope_std / √R`. Es la diferencia
entre "cuánto varía una medición" y "cuán precisa es la media de R mediciones".

Consecuencias del bug: el guard era **√8 ≈ 2.83× más estricto**, y **subir R no podía
ayudar** a pasar el guard (por eso el Camino 1 no cambió nada).

### La evidencia que lo confirmó (independiente del guard)
En V14, `target[0]` y `target[29]` dieron **8 pendientes negativas de 8** y aun así
abstuvieron. Bajo H₀ ("pendiente = 0"), P(8/8 mismo signo) = 1/128 ≈ **0.8%**. El test de
signos —no paramétrico, independiente del guard— dice que la señal es real con >99% de
confianza. Eran **falsos negativos demostrables**.

### El código actual (`inversion_attack.py::invert_feature`)
```python
    slope_mean = sum(slopes) / len(slopes)

    # Guard degenerado: si todas las pendientes son idénticas y valen 0,
    # slope_std=0 haría que el guard no dispare y luego se divida por cero.
    if slope_mean == 0.0:
        return None

    if R >= 2:
        slope_std = (sum((s - slope_mean) ** 2 for s in slopes) / (R - 1)) ** 0.5
        # El guard juzga la precisión de la MEDIA, no la dispersión de una medición
        # individual: por eso el error estándar (std/sqrt(R)), no std. Usar std
        # directamente hace el guard sqrt(R) veces más estricto y anula el beneficio de R.
        slope_sem = slope_std / (R ** 0.5)
        if slope_sem > 0 and abs(slope_mean) < snr_threshold * slope_sem:
            return None
    else:
        if abs(slope_mean) < 1e-4:
            return None
```
El bloque `[diag]` de `evaluator_attack.py` también divide por `√R`, para que ambos
números signifiquen lo mismo.

### El umbral: 5.0, por Bonferroni
`|slope_mean| / SEM` es un **estadístico t** contra H₀: pendiente = 0, con `df = R-1 = 7`.
Se hacen **30 tests simultáneos** (uno por target) en cada fila del barrido. Sin corregir,
el error familiar es `1-(0.95)³⁰ ≈ 79%`: casi se garantiza al menos un falso positivo por
fila. Umbrales t con df=7:

| corrección | α=0.05 | α=0.01 |
|---|---:|---:|
| sin corregir | 2.36 | 3.50 |
| Bonferroni (30 targets) | **4.94** | 6.50 |

Se usa **5.0**. Verificado contra datos reales: los SNR de V4 a ε_in=500 son 12–22, muy
por encima. La señal real sobrevive; el ruido no.

**Justificación para la tesis:** no es aflojar/apretar el umbral para que gane el ataque;
es corregir un estadístico mal construido y aplicar la corrección estándar por
comparaciones múltiples. **El umbral se elige por el argumento estadístico, nunca mirando
cuál da mejor MAE.**

**Si hubiera que ajustar:** si el umbral 5.0 se comiera parte de la región alta, bajar a
3.5 (α=0.01 sin corregir) como punto medio defendible. Ahora mismo no hace falta.

---

## 7. 🔧 PENDIENTE #1 (prioritario) — Barras de error con múltiples semillas

**Es el bloqueante para poder reportar cualquier número.** Ver §4.7.

1. Añadir un flag `--seed` a `evaluator_attack.py` que se pase a `random.seed()` y
   `np.random.seed()` al inicio. Sin él no se puede justificar por qué se reporta una
   corrida y no otra.
2. Repetir el barrido completo **3–5 veces** con semillas distintas.
3. Reportar **media ± desviación** por cada ε_in.

Con eso se puede: afirmar la forma de la curva con barras de error, dar la posición de la
rodilla **con un intervalo**, y demostrar que el bulto en ε_in=30 no es real.

---

## 8. 🔧 PENDIENTE #2 — Recalibrar el span

El span p25/p75 se adoptó para compensar el Problema D, que era un artefacto del
contenedor (§3). Además §4.5 muestra que percentiles fijos **no se adaptan a la escala de
la feature** (V14 sale penalizada).

### 8.1 Barrido de calibración
```bash
# ancho (descartado antes — puede ser el correcto ahora)
... --secret V4 --epsilon-out 10000 --snr-threshold 5.0 --probe-lo-pct 5  --probe-hi-pct 95
# medio (actual)
... --probe-lo-pct 25 --probe-hi-pct 75
# estrecho
... --probe-lo-pct 40 --probe-hi-pct 60
```
Comparar MAE/baseline y tasa de abstención. **Hacerlo con varias semillas** (§7), o la
varianza de ±35% enmascarará la diferencia entre spans.

### 8.2 Propuesta metodológica: span en unidades de σ
Percentiles fijos son frágiles entre features. Definir
`probe_lo/hi = media_pública ± c · std_pública(secreto)` (p.ej. `c = 1.5`):
- se adapta automáticamente a la escala de cada feature,
- evita recalibrar percentiles al cambiar de secreto,
- es un párrafo fácil de justificar en la tesis.

Implica añadir un flag `--probe-sigma` alternativo a `--probe-lo-pct/--probe-hi-pct`.

---

## 9. Plan de trabajo (orden recomendado)

1. **`--seed` + 3–5 corridas del barrido** → barras de error (§7). **Prioritario.**
2. **Calibrar el span** (§8), ya con barras de error para que la comparación sea válida.
3. **Reescribir el §2D de la tesis** con lo que se observe ahora, no con lo viejo.
4. **Experimentos 2 y 3** (§10).
5. (Opcional) Repetir con V14/V12 para la narrativa de "qué features son recuperables"
   (§4.2), reportando **MAE/baseline**, no MAE crudo (§4.4).

---

## 10. Los tres experimentos que cierran el Punto 2

1. **Entrada aislada** ✅ *(hecho, falta repetir con semillas)*: `ε_out=10000` fijo (output
   shield transparente), barrer `ε_in`. Mide protección del input shield. Piso irreducible
   = asíntota cuando K→∞.
2. **Salida aislada** ⬜: `ε_in` muy alto (input casi transparente), barrer `ε_out`
   (1..500). Mide protección del output shield; aquí el sesgo de clipping **ES** parte de
   la defensa medida, no un artefacto.
3. **Realista** ⬜ (un solo punto): `ε_in=1`, `ε_out=2` (defaults del governor). Protección
   combinada tal como la sufriría un atacante contra la API desplegada. Es el número
   "¿qué tan segura es la API?".

Los dos primeros son diagnósticos (aíslan mecanismos); el tercero es el resultado.

### Punto conceptual para la defensa
Que el ataque falle a ε_in bajo NO es un defecto: **es la defensa funcionando**, y es
exactamente la cantidad que se quiere cuantificar. Reportar un solo punto en máxima
protección ("el ataque falla") no demuestra nada — hay que mostrar la curva completa. La
mitad "el ataque gana con ε_in alto" **calibra el instrumento**: prueba que el ataque es
real, y por tanto que su fracaso bajo protección fuerte es informativo.

Ahora esa calibración está **demostrada empíricamente**: el ataque reduce el error un 82%
con las capas casi transparentes, y cae a abstención total con ε_in ≤ 10. Ya no es una
afirmación teórica.

> *Nota:* este enmarcado es una sugerencia de presentación, no un hecho técnico.
> Ajustarlo si no encaja con la narrativa de la tesis.

---

## 11. Restricciones del proyecto (respetar)
- **NO** tocar `modelo-python/` ni evaluarlo en exceso (es intercambiable).
- **NO** modificar la lógica de las capas de protección (`input_privacy_layer.py`,
  `privacy_layer.py`, `governor.py`) para "hacer ganar" el ataque. Todos los cambios
  legítimos son en el **harness** (`evaluator_attack.py`, `attacks/inversion_attack.py`),
  que es donde corresponde ajustar la medición.
- **Excepción documentada:** el rebuild del contenedor (§2) no es una modificación de la
  defensa, sino la corrección de una desincronización que impedía que la defensa se
  ejecutara según su propio código.

---

## 12. Camino 3 (futuro, Punto 4)
Cuando se cambie de modelo (p.ej. árbol), la linealidad desaparece. Ahí se necesita
`invert_feature_search` (búsqueda 1-D con Brent/bisección): en vez de despejar de una
recta, se sondean varios valores de `x_s` hasta encontrar el que produce
`logit(denoise) = logit(p_target)`. Robusto a no-linealidades y al span. No se implementa
ahora.

---

## 13. Checklist antes de confiar en cualquier resultado
- [ ] `docker compose up -d --build privacy-api` tras cualquier cambio en `privacy-api/`.
- [ ] `sanity_check.py` da un barrido **monótono**.
- [ ] La respuesta reporta el `epsilon_in_total` que se **pidió**, no el default 1.0.
- [ ] El `[diag]` muestra **signos consistentes** en las R pendientes.
- [ ] `slope_mean` **no colapsa hacia 0** al subir K (si colapsa, es ruido).
- [ ] El código en disco coincide con el commiteado (`git status`).
- [ ] **El número reportado viene de ≥3 semillas, no de una corrida única** (§4.7).

---

## 14. Comandos de referencia

```bash
# Rebuild tras cualquier cambio en privacy-api/
docker compose up -d --build privacy-api

# Sanity check (segundos) — SIEMPRE ante una anomalía
python3 privacy-api/sanity_check.py

# Calibración rápida (segundos): K bajo, un solo epsilon
python3 privacy-api/evaluator_attack.py --K 50 --limit 130 --public-size 100 \
  --secret V4 --epsilon-out 10000 --epsilons-in 500 --snr-threshold 5.0

# Barrido completo (decenas de minutos) — el de la tesis
python3 privacy-api/evaluator_attack.py --K 400 --limit 130 --public-size 100 \
  --secret V4 --epsilon-out 10000 --snr-threshold 5.0

# Experimento 2 (salida aislada) — pendiente de implementar el barrido de ε_out
# Experimento 3 (realista): --epsilon-out 2 --epsilons-in 1
```
