# Explicación completa — Experimentos 1, 2 y 3 del Punto 2

*Documento de lectura, no de consulta rápida. Explica el montaje desde cero, qué hace el
ataque paso a paso, qué significa cada número de las tablas, qué conclusión sostiene cada
uno, y —sobre todo— cuáles NO sostiene. Reemplaza a `RESUMEN_curva_proteccion.md`, que
quedó obsoleto (ver §10).*

---

## 0. Cómo leer este documento

Está pensado para leerse de corrido. Las secciones 1 a 3 explican el montaje: qué se está
midiendo y con qué instrumento. Las secciones 4 a 6 recorren los tres experimentos uno a
uno. La sección 7 es la síntesis: por qué las dos capas de defensa se comportan de forma
distinta, que es el hallazgo conceptual del conjunto. Las secciones 8 a 11 son de higiene:
qué no se puede afirmar todavía, qué documentos se contradicen entre sí y por qué, qué falta
por correr, y qué preguntas previsiblemente hará un tribunal.

Marcas que uso a lo largo del texto:

- **Medido**: sale directamente de los artefactos de las corridas.
- **Inferido**: se deduce de lo medido con un razonamiento que explicito.
- **Hipótesis**: explicación plausible que los datos actuales *no* distinguen de sus
  alternativas. Cuando marco algo así, digo también qué corrida la resolvería.

---

## 1. El sistema bajo prueba

### 1.1 Qué hay detrás de la API

Hay un modelo de detección de fraude que recibe una transacción (un vector de features:
`Time`, `V1`…`V14`, `Amount`) y devuelve una probabilidad de fraude. Ese modelo está
envuelto por un *privacy wrapper* —el «governor»— que interpone dos capas de defensa
independientes entre el cliente y el modelo:

**El input shield.** Antes de que el vector de features llegue al modelo, se le añade ruido
Laplace a cada feature. El parámetro que lo controla es `ε_in`. Un `ε_in` bajo significa
mucho ruido; un `ε_in` alto significa poco ruido. Con `ε_in = 10000` la capa es
prácticamente transparente: el modelo ve casi el vector original.

**El output shield.** Después de que el modelo produce su probabilidad, se le añade ruido
Laplace al score antes de devolverlo al cliente, y además se recorta (*clipping*) a un rango
válido. El parámetro es `ε_out`, con la misma convención: bajo = mucho ruido, alto = poco.
Con `ε_out = 10000` la capa es prácticamente transparente.

Los valores **por defecto del governor en producción son `ε_in = 1` y `ε_out = 2`**: ambas
capas en su configuración más protectora. Ese es el punto que mide el Experimento 3.

### 1.2 Por qué hay que barrer los dos ε por separado

Si solo se midiera la configuración desplegada (`ε_in=1`, `ε_out=2`) y el ataque fallara, no
se sabría nada útil. No se sabría si falla porque la defensa funciona o porque el ataque es
malo. Y aunque se supiera que la defensa funciona, no se sabría **cuál de las dos capas** lo
consigue, ni **cuánto margen** hay antes de que deje de funcionar.

De ahí el diseño de tres experimentos:

| | qué fija | qué barre | qué pregunta responde |
|---|---|---|---|
| **Exp 1** | `ε_out = 10000` (salida transparente) | `ε_in` de 1 a 500 | ¿Cuánto protege la capa de entrada, ella sola? |
| **Exp 2** | `ε_in = 10000` (entrada transparente) | `ε_out` de 1 a 500 | ¿Cuánto protege la capa de salida, ella sola? |
| **Exp 3** | ambas en sus defaults | nada (punto único) | ¿Qué tan segura es la API tal como está desplegada? |

Los experimentos 1 y 2 son **espejos exactos**: en cada uno, la capa que no se barre se deja
en 10000 (transparente), de modo que lo que se mide es la capa barrida en aislamiento. Esa
simetría es lo que permite comparar los dos ejes en §7.

### 1.3 Una advertencia sobre lo que significa «ε» aquí

Conviene tenerla clara desde el principio porque contamina varias comparaciones más
adelante.

En privacidad diferencial, la escala del ruido Laplace que hay que añadir para conseguir un
presupuesto ε es **`b = Δ/ε`**, donde `Δ` es la *sensibilidad*: cuánto puede cambiar la
cantidad que se protege cuando cambia un solo registro. La sensibilidad de la capa de
entrada (relacionada con el rango de recorte de una feature; en el sondeo, el span de V4 es
1.4790) **no es la misma cantidad** que la sensibilidad de la capa de salida (relacionada
con el rango del score o del logit).

Consecuencia práctica: **`ε_in = 500` y `ε_out = 500` no son «la misma cantidad de ruido».**
Son la misma *etiqueta de presupuesto* aplicada a dos cantidades con escalas distintas.
Cuando en §7 se compare el comportamiento de los dos ejes, la afirmación defendible será *«a
igual ε nominal, la capa de salida deja pasar más»*, y no *«la capa de entrada protege
más»*. La segunda requiere convertir ambos ejes a `b = Δ/ε` con las sensibilidades reales,
que están en el código del servidor y todavía no se han extraído.

---

## 2. El ataque, paso a paso

Esta sección es la más importante para entender los resultados, porque casi todas las
lecturas interesantes salen de entender qué hace el atacante internamente y dónde puede
romperse.

### 2.1 El modelo de amenaza

El atacante quiere recuperar el valor de **una feature secreta** de un registro concreto: en
todos estos experimentos, `V4` de una transacción objetivo. Sus supuestos son:

1. **Conoce todas las demás features del objetivo.** Sabe `Time`, `V1`, `V2`, …, `V14`
   (excepto `V4`) y `Amount` de la transacción de la víctima. Es un supuesto fuerte y hay
   que declararlo en la tesis (§11).
2. **Puede consultar la API tantas veces como quiera**, con vectores que él construye.
3. **Conoce los metadatos públicos de la distribución de `V4`**: su media, su desviación
   típica y sus percentiles. Esto es razonable: son estadísticas agregadas del dataset,
   típicamente publicadas.
4. **Ha observado el score del objetivo.** Es decir, tiene acceso a la respuesta de la API
   para el registro completo de la víctima, incluido el `V4` verdadero que él no conoce.

El nombre «inversión inter-request» viene del punto 4 combinado con el 2: la fuga no ocurre
dentro de una sola respuesta, sino al **comparar** la respuesta del objetivo con las
respuestas de consultas fabricadas por el atacante.

### 2.2 El truco: linealidad local en el logit

El modelo devuelve una probabilidad `p`. Las probabilidades están comprimidas contra 0 y 1,
lo que arruina cualquier aritmética de diferencias. Por eso el atacante trabaja en el
**logit**: `logit(p) = log(p/(1−p))`. En el logit, un clasificador de tipo regresión
logística es *lineal* en las features, y casi cualquier otro modelo es aproximadamente
lineal en un tramo corto.

El atacante explota eso así:

1. Toma el registro conocido de la víctima y **sustituye `V4` por un valor de sondeo bajo**,
   `probe_lo` = percentil 25 de la distribución pública de V4 = **−0.2257**. Consulta la
   API. Obtiene `p_lo`.
2. Repite con un **valor de sondeo alto**, `probe_hi` = percentil 75 = **+1.2534**. Obtiene
   `p_hi`.
3. Calcula la **pendiente** de la respuesta respecto a V4:

   ```
   slope = ( logit(p_hi) − logit(p_lo) ) / ( probe_hi − probe_lo )
   ```

   El denominador es el *span* del sondeo: 1.2534 − (−0.2257) = **1.4790**.
4. Consulta con el registro **verdadero** de la víctima y obtiene `p_target`.
5. Despeja el valor secreto:

   ```
   V4_estimado = probe_lo + ( logit(p_target) − logit(p_lo) ) / slope
   ```
6. Recorta el resultado al rango público conocido de V4 (`PUBLIC_BOUNDS`), porque un valor
   fuera de ese rango es seguro que está mal.

Es una regla de tres en el espacio del logit. Si el modelo fuera exactamente lineal en el
logit y no hubiera ruido, este procedimiento recuperaría `V4` **exactamente**. El
Experimento 2 en su extremo transparente muestra que, en la práctica, casi lo hace: MAE
0.094 sobre una baseline de 1.139, con 29 de 30 targets reconstruidos dentro de tolerancia.

### 2.3 El promediado: qué es `K` y por qué importa tanto

Las dos capas de defensa añaden ruido aleatorio **fresco en cada request**. Si el atacante
consulta dos veces con el mismo vector, recibe dos respuestas distintas.

La contramedida obvia es promediar. La función `denoise(client, vector, K)` envía **el mismo
vector `K` veces** y promedia las respuestas. Con `K = 400`, cada uno de los tres valores
(`p_lo`, `p_hi`, `p_target`) es en realidad la media de 400 consultas.

Esto es central para entender todo lo que sigue:

> **`K` es el presupuesto de consultas del atacante. El ruido de media cero se atenúa
> proporcionalmente a `1/√K`.** Para reducir a la mitad el efecto del ruido hay que
> cuadruplicar `K`. El ruido que *no* es de media cero —el sesgo de clipping, la atenuación
> de la señal— **no se atenúa nada** por más que se suba `K`.

Esa distinción entre «lo que se promedia» y «lo que no se promedia» es el eje de toda la
sección 7.

### 2.4 El guard: `R`, el SNR y la abstención

Aquí está la parte no obvia del diseño, y la que hace que estos experimentos midan algo
honesto.

El paso 5 del ataque **divide por `slope`**. Si `slope` es pequeño, la división amplifica
cualquier error. Si `slope` es puro ruido y por azar sale próximo a cero, la estimación
**explota**: sale un número absurdo, se recorta a los límites públicos, y el error de ese
target es enorme. Un solo target así arrastra el MAE de los 30 y hace que el ataque parezca
*peor que no atacar*.

Por eso el atacante lleva un mecanismo de autoprotección:

1. En vez de estimar la pendiente una vez, la estima **`R = 8` veces de forma
   independiente**. Cada una de esas 8 estimaciones consume sus propias `2·K` consultas (K
   para `probe_lo`, K para `probe_hi`).
2. Con esas 8 pendientes calcula:
   - `slope_mean`: la media de las 8.
   - `slope_std`: la desviación típica muestral de las 8.
   - `SEM = slope_std / √R`: el error estándar de la media.
   - **`SNR = |slope_mean| / SEM`**.
3. Si `SNR ≥ umbral`, se fía de la pendiente y estima. Se registra `[PASA]`.
4. Si `SNR < umbral`, **declina estimar**. Se registra `[ABSTIENE]`, y para ese target
   devuelve simplemente `public_mean`: la media pública de V4.

El `SNR` así definido **es exactamente un estadístico t** contrastando la hipótesis nula
«la pendiente verdadera es cero», con `df = R − 1 = 7` grados de libertad. No es una
heurística: es un test de hipótesis.

### 2.5 Por qué el umbral vale 5.0 y por qué eso es defendible

El umbral no se eligió mirando cuál daba mejor MAE. Esto importa mucho más de lo que parece,
así que conviene poder explicarlo bien.

Se hacen **30 tests simultáneos**, uno por target, en cada punto de la curva. Con un nivel de
significación α = 0.05 por test, la probabilidad de que **al menos uno** de los 30 dé un
falso positivo es:

```
1 − (0.95)³⁰ ≈ 0.79
```

Es decir, **casi el 79%**: prácticamente se garantiza que en cada fila de la tabla haya al
menos un target que pase el guard por puro azar. Y ese target es precisamente el que divide
por una pendiente que es ruido y hace explotar el promedio.

La corrección de Bonferroni consiste en exigir α = 0.05/30 por test. El valor crítico de una
t de Student con df = 7 a ese nivel es **4.94**, redondeado a **5.0**.

La evidencia empírica de que esto era necesario está en las corridas previas con umbral 2.0
(recogidas en `RESUMEN_curva_proteccion.md` §2):

| ε_in | MAE con umbral 2.0 | MAE con umbral 5.0 | baseline |
|---:|---:|---:|---:|
| 1 | 1.3208 | **1.1389** | 1.1389 |
| 10 | 1.1787 | **1.1389** | 1.1389 |
| 30 | 1.4473 | 1.1530 | 1.1389 |
| 50 | 1.1573 | 1.0395 | 1.1389 |

Con umbral 2.0 el ataque quedaba **por encima de la baseline**, o sea que atacar era peor que
no atacar. Eso no era protección de la API: era el atacante haciéndose daño a sí mismo por
un guard mal calibrado. Un revisor podría haber dicho, con razón, que ese resultado no
mide nada.

**El argumento a sostener en la defensa es este:** el umbral se fija por un criterio
estadístico *a priori* (Bonferroni sobre 30 tests, t con df=7), nunca eligiendo el que
produce el resultado más favorable. Si hubiera que ajustarlo, el punto intermedio defendible
sería 3.5 (α = 0.01 sin corregir), también elegido por argumento y no por resultado.

### 2.6 Por qué el guard hace más honesto el experimento, no menos

Es tentador ver la abstención como «el ataque se rinde» y por tanto como una debilidad del
instrumento. Es lo contrario.

Un atacante real no ejecuta ciegamente un procedimiento que sabe que a veces produce basura.
Un atacante real comprueba si tiene señal y, si no la tiene, se queda con la mejor
conjetura disponible sin gastar más. Al dotar al ataque de ese criterio, se está midiendo
contra un **atacante competente**, que es el escenario que interesa. Y se elimina una fuente
de ruido que confundía el resultado: sin guard, el MAE mezclaba «cuánta información fuga la
API» con «cuántas veces explotó la división».

El efecto secundario útil es que **la abstención deja una firma aritmética inconfundible**,
que se explica en §3.3.

### 2.7 El coste de todo esto en consultas

Por cada target y cada punto de la curva, el ataque gasta aproximadamente:

```
2·K·R   (las R=8 estimaciones de pendiente, cada una con 2 sondeos de K consultas)
+ 2·K   (el score del objetivo y el margen del procedimiento)
```

Con `K = 400` y `R = 8` eso son **7.200 consultas por target y por epsilon**. Multiplicado
por 30 targets y 7 epsilons: **≈1,5 millones de consultas por corrida** del Experimento 1.
Las 5 corridas del Exp 1 tardaron **≈10 horas**, de donde se **infiere** un caudal efectivo
del orden de 200 requests/segundo contra el contenedor. Ese número es aproximado y sirve
solo para planificar, no para reportar.

---

## 3. Las métricas: qué significa exactamente cada columna

### 3.1 La baseline: 1.1389

Es el **MAE de no atacar**: el error medio absoluto que comete alguien que, sin consultar
nada, contesta «el V4 de este target es la media pública de V4» para los 30 targets.

```
baseline_mae = media sobre los 30 targets de | public_mean − V4_verdadero |
```

Es el listón. **Cualquier ataque que no baje de 1.1389 no ha extraído información
utilizable.** Y es determinista: vale exactamente 1.1389 en las 5 corridas del Exp 1, las 3
del Exp 2 y las 3 del Exp 3, porque la partición del dataset en pool público (100 filas) y
targets (30 filas) no depende de la semilla. Que sea idéntica en los once artefactos es en
sí mismo una verificación de que las tres tandas son comparables entre sí.

### 3.2 MAE

Error medio absoluto de las estimaciones del atacante sobre los 30 targets. Menor = el
ataque funciona mejor = la defensa protege peor. Es la métrica principal de las tres tablas.

### 3.3 La firma «1.1389 exacto»: cómo se lee una abstención total

Este detalle merece su propio apartado porque es el que sostiene la afirmación más fuerte de
todo el trabajo.

Si el ataque se abstiene en un target, ese target contribuye al MAE con exactamente el mismo
error que la baseline (porque devuelve `public_mean`, que es lo que la baseline devuelve).
Si se abstiene en **los 30**, el MAE del ataque es **numéricamente idéntico a la baseline**:
1.1388595856794768, hasta el último decimal.

Por eso, cuando en una tabla aparece `1.1389` y no `1.1390` ni `1.1387`, **no significa «el
ataque estimó mal»**. Significa algo mucho más específico:

> El ataque examinó los 30 targets, en los 30 detectó que no había pendiente distinguible de
> cero, y en los 30 declinó estimar.

Es una afirmación cualitativamente distinta de «el error fue grande». Es protección
*medida*, no supuesta, y es observable a simple vista en el CSV.

### 3.4 `success_rate` y `tol_abs`

`success_rate` es el porcentaje de los 30 targets cuya estimación cae a menos de `tol_abs`
del valor verdadero. La tolerancia es **`tol_abs = 0.3189`**, que es un cuarto de la
desviación típica pública de V4. Es la misma en los tres experimentos, así que las cifras de
`success_rate` son directamente comparables entre ellos.

Es la métrica más legible para un lector no técnico: *«con la protección desplegada el
atacante reconstruye 2 de cada 30 transacciones; sin protección, 29 de cada 30»*.

**Ojo con el suelo.** El valor **6.7% = 2 de 30 es el nivel de azar**, no un logro del
atacante: son los 2 targets cuyo V4 verdadero cae, por casualidad, a menos de 0.3189 de la
media pública. Cuando el ataque se abstiene en los 30, `success_rate` vale 6.7%
automáticamente. Un `success_rate` de 6.7% significa **cero información extraída**.

### 3.5 `mean_error`: el sesgo, con signo

`mean_error` es el error medio **con signo** (no en valor absoluto). Dice si el estimador
tiende a sobreestimar o subestimar.

Tiene un valor de referencia muy útil: **cuando el ataque abstiene en los 30 targets,
`mean_error` vale exactamente +0.159**, que es el sesgo puro de `public_mean` respecto a los
targets. Ese número aparece en el Exp 1 a `ε_in = 1` y en el Exp 2 a `ε_out = 1`, y sirve
como confirmación independiente de la abstención total.

Cuando `mean_error` se aleja de +0.159, está diciendo algo sobre las estimaciones que sí se
produjeron. Si se acerca a 0, el estimador se ha vuelto insesgado (hay señal real). Si sube
**por encima** de +0.159, ocurre algo más interesante: las estimaciones que sobreviven al
guard están sesgadas hacia arriba **más de lo que estaría alguien que no atacara**. Eso
pasa en el Exp 2 y se analiza en §5.6.

### 3.6 Las barras de error: por qué `n` corridas y por qué la `t` cambia

Las tres tandas repiten cada configuración con varias semillas (`--seed`) y reportan media ±
desviación típica más un intervalo de confianza al 95% construido con una t de Student.

El valor crítico de la t depende de los grados de libertad, que son `n − 1`:

| experimento | n | df | t crítica (95%) |
|---|---:|---:|---:|
| Exp 1 | 5 | 4 | 2.776 |
| Exp 2 | 3 | 2 | **4.303** |
| Exp 3 | 3 | 2 | **4.303** |

Esto explica por qué los IC del Exp 2 y del Exp 3 parecen anchos en relación con su
desviación típica: con solo 3 corridas, la penalización por incertidumbre sobre la propia
varianza es grande (4.303 frente a 2.776, un 55% más ancho). No es que esas tandas sean más
ruidosas; es que se sabe menos sobre su ruido.

**La regla de lectura que se usa en los tres documentos:** si el IC 95% *contiene* la
baseline (1.1389), la diferencia frente a no atacar **no es estadísticamente
significativa**, sin importar cuánto se aleje la media. Si el IC queda enteramente por
debajo, el ataque gana de verdad.

### 3.7 Una nota sobre `--seed` que hay que declarar en la tesis

La semilla **no hace reproducibles las corridas**, y esto está verificado: dos ejecuciones
con `--seed 1` idéntico dieron MAE 0.5427 y 0.6689.

La razón es estructural: el ruido Laplace lo genera **el servidor**, dentro del contenedor
`privacy-api`, fuera del alcance del generador de números aleatorios del cliente. La semilla
solo fija la aleatoriedad del *harness* del atacante y sirve para trazabilidad en el log.

**Esto es metodológicamente correcto y conviene presentarlo como una decisión, no como una
limitación.** La varianza corrida a corrida que se observa es exactamente la que enfrentaría
un atacante real contra la API desplegada. Suprimirla sembrando el RNG del servidor
implicaría tocar la capa de defensa para medirla, que es justo lo que no se debe hacer.
Reportarla es más honesto.

Lo que sí es determinista y se verifica idéntico en las once corridas: la partición
pool/targets, la baseline (1.1389), el span de sondeo (`[−0.2257, +1.2534]`) y `tol_abs`
(0.3189).

---

## 4. Experimento 1 — la entrada aislada

**Configuración:** secreto `V4`, `ε_out = 10000` (salida transparente), `K = 400`, `R = 8`,
umbral 5.0, span p25/p75, 30 targets, pool de 100 filas, **n = 5** semillas. Coste ≈10 horas.

**Qué pregunta responde:** si la única defensa fuera el input shield, ¿cuánta protección
daría, y a partir de qué `ε_in` deja de darla?

### 4.1 La curva

| ε_in | MAE (media ± std) | IC 95% | CV | Reducción vs baseline | ¿gana? |
|---:|:---:|:---:|---:|---:|:---:|
| 1 | 1.1389 ± 0.0000 | — | 0.0% | 0.0% | no — **protección total** |
| 10 | 1.1389 ± 0.0000 | — | 0.0% | 0.0% | no — **protección total** |
| 30 | 1.1316 ± 0.0209 | [1.1057, 1.1575] | 1.8% | 0.6% | **no significativo** |
| 50 | 1.0700 ± 0.0375 | [1.0234, 1.1165] | 3.5% | 6.1% | sí |
| 100 | 0.7292 ± 0.0537 | [0.6624, 0.7959] | 7.4% | **36.0%** | sí |
| 300 | 0.3685 ± 0.0450 | [0.3126, 0.4243] | 12.2% | 67.6% | sí |
| 500 | 0.2654 ± 0.0160 | [0.2456, 0.2852] | 6.0% | **76.7%** | sí |

La forma es la buscada: **plana sobre la baseline mientras la protección es fuerte, luego
monótona decreciente** conforme el shield se vuelve transparente.

### 4.2 Protección total a `ε_in ≤ 10` — y qué significa exactamente

Las dos primeras filas dan `1.1389` con desviación típica **cero** a lo largo de las cinco
semillas. Aplicando §3.3: en `5 corridas × 2 epsilons = 10 configuraciones`, y en cada una
de ellas en los 30 targets, el ataque detectó ausencia de pendiente y se abstuvo. Son 300
abstenciones consecutivas sin una sola excepción.

Ese es el resultado más limpio de todo el trabajo. Y es cualitativamente distinto de «el
error fue alto»: el ataque no falló al estimar, **decidió que no había nada que estimar**.

### 4.3 El bulto de `ε_in = 30` no existe

En corridas individuales previas, `ε_in = 30` aparecía a veces por encima y a veces por
debajo de la baseline, y se llegó a teorizar sobre una posible inestabilidad numérica del
logit en esa zona.

Los datos crudos de las cinco semillas resuelven la duda:

| ε_in | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---:|---:|---:|---:|---:|---:|
| 30 | 1.1036 | 1.1398 | 1.1389 | 1.1184 | 1.1574 |

Dos por debajo de la baseline, tres por encima, y la media 1.1316 ± 0.0209 con IC
[1.1057, **1.1575**]. **El intervalo contiene la baseline (1.1389)**, luego la diferencia no
es significativa. Se reporta como «sin efecto detectable» y **no hay que teorizar sobre
ello en la tesis**. Era ruido de muestreo.

Este apartado tiene valor metodológico más allá del número: es un ejemplo concreto de por
qué ninguna cifra de una corrida única debe entrar en el documento final.

### 4.4 La rodilla, y una ambigüedad de definición que conviene resolver

La reducción salta de 6.1% (`ε_in=50`) a 36.0% (`ε_in=100`), y los intervalos no se solapan:
[1.0234, 1.1165] frente a [0.6624, 0.7959]. El salto es real y está bien delimitado.

Pero hay una ambigüedad en el término «rodilla» que atraviesa los documentos y hay que
zanjar antes de escribir:

- **Definición A — primer punto donde el ataque gana significativamente.** Es `ε_in = 50`
  (IC [1.0234, 1.1165], enteramente por debajo de 1.1389).
- **Definición B — intervalo donde la protección se derrumba.** Es entre `ε_in = 50` y 100,
  que es el salto grande con IC disjuntos.

`RESULTADOS_EXPERIMENTO_1.md` usa la definición B. `RESUMEN_curva_proteccion.md` usaba la A
sobre datos de una sola corrida y por eso decía «entre 30 y 50». **Son dos preguntas
distintas y conviene reportar las dos por separado y con nombres distintos**, en vez de usar
la misma palabra para ambas: por ejemplo «umbral de fuga detectable» para A y «colapso de la
protección» para B.

### 4.5 La varianza real es del 5–12%, no del ±35%

Durante la depuración se estimó una varianza corrida a corrida de ±35% a partir de **dos**
corridas (0.5387 frente a 0.7548 en `ε_in = 100`). Ese ±35% se propagó a
`RESUMEN_curva_proteccion.md`, donde ocupa una sección entera marcada en rojo.

Con n = 5, el coeficiente de variación real está entre el 5% y el 12%, y aquellas dos
corridas resultaron ser los extremos de la distribución. **La estimación de ±35% era una
sobreextrapolación de dos puntos.** Vale la pena conservar el episodio como recordatorio
metodológico: dos observaciones no definen una varianza, y un intervalo construido sobre dos
puntos casi siempre subestima o exagera de forma dramática.

Nótese además que el CV crece con `ε_in` (del 1.8% al 12.2%) **no porque las corridas se
vuelvan más ruidosas en términos absolutos** —la desviación típica se mueve entre 0.016 y
0.054— sino porque el denominador (la media) se derrumba. Es un artefacto de la métrica
relativa, no un fenómeno.

### 4.6 `success_rate` — la lectura para un lector no técnico

Del CSV de la corrida 5 (⚠️ solo esa corrida; consolidarlo con las cinco sigue pendiente):

| ε_in | success_rate | interpretación |
|---:|---:|---|
| 1–30 | 6.7% | 2 de 30 = **nivel de azar, cero información** |
| 50 | 13.3% | 4 de 30 |
| 100 | 33.3% | 10 de 30 |
| 300 | 66.7% | 20 de 30 |
| 500 | 73.3% | 22 de 30 |

En una frase: *«con protección fuerte el atacante reconstruye 2 de cada 30 transacciones,
que son las que habría acertado sin atacar; con el shield casi apagado, 22 de 30»*.

### 4.7 El sesgo cambia de signo

`mean_error` pasa de **+0.159** en `ε_in = 1` a **−0.051** en `ε_in = 500`. La primera cifra
es la firma de la abstención total (§3.5): es el sesgo de `public_mean`. La segunda dice que,
con señal real disponible, el estimador se vuelve casi insesgado. Es consistente con lo
esperado y funciona como validación secundaria del procedimiento.

### 4.8 El bloque `[diag]` del Exp 1: el instrumento está calibrado

Antes de cada barrido se inspeccionan 3 targets en detalle en `ε_in = 500` (el extremo de
máxima señal). Las 15 mediciones (3 targets × 5 corridas) fueron **todas `[PASA]`**:

| estadístico | rango observado |
|---|---|
| `slope_mean` | +0.462 … +0.625 |
| `slope_std` | 0.068 … 0.181 |
| SNR | 8.74 … 25.49 |
| signos +/− | **8/0 en las 15** |

«Signos 8/0» quiere decir que las 8 estimaciones independientes de la pendiente salieron
todas positivas. Con señal real eso es lo esperado; con ruido puro saldría ~4/4.

Esto cumple dos funciones. Primera, demuestra que el ataque funciona cuando hay señal, lo
que convierte su fracaso a `ε_in` bajo en evidencia de protección y no en incompetencia.
Segunda, **confirma que el umbral de Bonferroni no está sacrificando señal legítima**: con
SNR de 8.7 a 25.5 frente a un umbral de 5.0, hay margen de sobra. El umbral solo actúa
filtrando falsos positivos en la región de `ε_in` baja, que es justo para lo que se puso.

### 4.9 Un parámetro sin justificación: el span p25/p75

De la tabla de configuración validada del Exp 1, todos los parámetros tienen una razón
documentada salvo uno:

| parámetro | valor | por qué |
|---|---|---|
| `ε_out` | 10000 | Transparente de verdad. Con 50 la señal se comprimía ~10×: la premisa de «sesgo despreciable para p>0.01» no se cumple, porque las `p` reales van de 1e−6 a 1e−1. |
| `K` | 400 | Promediado suficiente; `slope_std` cae como 1/√K. |
| `R` | 8 | Repeticiones para el guard. Solo es útil desde la corrección del SEM. |
| `snr_threshold` | 5.0 | Bonferroni sobre 30 tests, t con df=7 (§2.5). |
| guard | `\|slope_mean\|/(slope_std/√R)` | Estadístico t contra H₀: pendiente=0. La versión previa dividía por `slope_std` directo, era √R veces más estricta y anulaba el beneficio de R. |
| **span** | **p25/p75** | **Sin justificación sólida.** Heredado de una compensación por un bug ya corregido. |

**Por qué el span importa más de lo que parece.** El span es el denominador de la pendiente
(§2.2) y gobierna directamente el SNR: un span más ancho produce una diferencia de logits
mayor para el mismo nivel de ruido, luego más SNR, luego el guard pasa más veces, luego el
ataque parece más fuerte. Hay evidencia directa: en `V14` (distribución compacta, span
0.502) el span estrecho penalizaba el SNR. **El span, y no la sensibilidad del modelo,
gobierna el SNR.**

La consecuencia es incómoda: **el span es un parámetro de sintonía del atacante, y elegirlo
mal subestima la potencia del ataque, lo que a su vez sobreestima la protección medida.** Un
revisor puede preguntar «¿y si con otro span el ataque hubiera roto la defensa a `ε_in=1`?»,
y ahora mismo no hay respuesta.

Hay además una tensión real que conviene nombrar: un span ancho mejora el SNR pero **rompe
el supuesto de linealidad local** del logit (§2.2), introduciendo sesgo de linealización. Un
span estrecho conserva la linealidad pero mata el SNR. El óptimo no es evidente y por eso
hace falta calibrarlo empíricamente.

**Propuesta:** comparar p5/p95, p25/p75 y p40/p60 con varias semillas, y considerar definir
el span en unidades de σ (`media ± 1.5·std`) en vez de percentiles fijos, para que se adapte
a la escala de cada feature y no haya que recalibrar al cambiar de secreto.

---

## 5. Experimento 2 — la salida aislada

**Configuración:** secreto `V4`, `ε_in = 10000` (entrada transparente), `K = 200`, `R = 8`,
umbral 5.0, span p25/p75, 30 targets, pool de 100 filas, **n = 3** semillas. Código nuevo:
`--axis out` → `run_experiment_out()`, añadido en el commit `0542d54`.

**Qué pregunta responde:** el espejo del Exp 1. Si la única defensa fuera el output shield,
¿cuánta protección daría?

Hay una diferencia conceptual importante frente al Exp 1, y está escrita en el propio
docstring del código: **aquí el sesgo de clipping del output shield ES parte de la defensa
medida, no un artefacto.** Por eso `ε_out` se barre hasta valores bajos a propósito, mientras
que en el Exp 1 el `ε_out` se fijaba deliberadamente altísimo para que el clipping no
contaminara la medición del input shield.

### 5.1 La curva

| ε_out | MAE (media ± std) | IC 95% | CV | Reducción vs baseline | ¿gana? |
|---:|:---:|:---:|---:|---:|:---:|
| 1 | 1.1308 ± 0.0140 | [1.0959, 1.1657] | 1.2% | 0.7% | **no significativo** |
| 2 | 1.1200 ± 0.0122 | [1.0897, 1.1503] | 1.1% | 1.7% | **no significativo** |
| 5 | 1.0772 ± 0.0134 | [1.0438, 1.1106] | 1.3% | 5.4% | sí |
| 10 | 1.0318 ± 0.0245 | [0.9708, 1.0927] | 2.4% | 9.4% | sí |
| 50 | 0.7026 ± 0.0553 | [0.5652, 0.8399] | 7.9% | **38.3%** | sí |
| 100 | 0.4210 ± 0.0620 | [0.2671, 0.5749] | 14.7% | 63.0% | sí |
| 500 | 0.0941 ± 0.0211 | [0.0416, 0.1466] | 22.4% | **91.7%** | sí |

La curva es **monótona decreciente en las medias, sin bultos**. Es más limpia que la del
Exp 1, que tenía la anomalía de `ε_in = 30`.

### 5.2 El hallazgo central: en el eje de salida no hay protección total en ningún punto

Este es el contraste que hay que explicar con cuidado, porque es fácil de leer mal.

En el Exp 1, `ε_in ≤ 10` daba `1.1389` exacto en **10 de 10** configuraciones. Aquí, el
punto más protector del barrido (`ε_out = 1`) lo da solo en **2 de 3** corridas:

| ε_out | seed 1 | seed 2 | seed 3 |
|---:|---:|---:|---:|
| 1 | **1.1389** | 1.1146 | **1.1389** |
| 2 | 1.1112 | 1.1149 | 1.1339 |

La seed 2 a `ε_out = 1` produjo 1.1146: algún target pasó el guard y estimó. Y en
`ε_out = 2` —**el default del governor**— ninguna de las tres corridas abstiene por completo.
Las tres producen estimaciones.

La distinción a reportar es **categórica, no de grado**:

| | Exp 1 (`ε_in≤10`) y Exp 3 | Exp 2 (`ε_out = 1, 2`) |
|---|---|---|
| MAE frente a baseline | idéntico (1.1389 exacto) | ligeramente inferior |
| significancia estadística | no aplica: no hay estimación que evaluar | **no significativa** |
| qué hace el ataque | detecta ausencia de señal y **declina** | **estima, pero mal** |
| etiqueta correcta | protección total | **sin ventaja detectable** |

Son cosas distintas y no conviene colapsarlas en una sola frase en la tesis. «Sin ventaja
detectable **a K = 200**» es una afirmación condicionada al presupuesto de consultas del
atacante. «Protección total», tal como la sostiene el `[diag]` del Exp 3, no lo está. La
sección 5.7 explica por qué esa condicionalidad es el punto débil del Exp 2.

### 5.3 La rodilla del eje de salida, y su peor resolución

El único salto limpio del barrido está **entre `ε_out = 10` y 50**: la reducción pasa de
9.4% a 38.3% y los IC no se solapan ([0.9708, 1.0927] frente a [0.5652, 0.8399]).

Los demás tramos no separan:
- de 5 a 10 los intervalos se solapan ampliamente;
- de 50 a 100 se solapan al borde (0.5652–0.5749), lo que no permite afirmar un salto.

**Pero hay un problema de resolución que hay que declarar.** El bracket 10→50 es un factor
**5×**, mientras que la rodilla del Exp 1 quedó acotada a un factor **2×** (50→100). El
barrido no tiene puntos intermedios. Es decir, la rodilla del eje de salida está
significativamente **peor localizada** que la del eje de entrada, y no por naturaleza del
fenómeno sino por elección de los puntos del barrido. Añadir `ε_out ∈ {20, 30}` la cerraría
a 2× e igualaría la precisión de los dos ejes.

Si se aplica la definición A de §4.4 (primer punto que gana significativamente), el número
es `ε_out = 5`.

### 5.4 El default del governor, por sí solo, no es la defensa

`ε_out = 2` en aislamiento: MAE 1.1200, reducción del 1.7%, IC [1.0897, 1.1503] que contiene
la baseline. No hay ventaja detectable, pero tampoco hay abstención.

Combinando esto con los otros dos experimentos, la conclusión se cierra en las dos
direcciones:

- **Exp 1:** `ε_in = 1` con la salida transparente → protección total.
- **Exp 3:** `ε_in = 1` **más** `ε_out = 2` → protección total, exactamente igual.
- **Exp 2:** `ε_out = 2` con la entrada transparente → sin ventaja detectable, pero sin
  abstención.

> **El input shield es necesario y suficiente para la protección observada en la
> configuración desplegada. El output shield en su valor por defecto no es ni lo uno ni lo
> otro: aislado no llega a producir abstención, y añadido al input shield no cambia nada.**

Esto es más fuerte de lo que podía decir el Exp 3 por sí solo. El Exp 3 solo mostraba que
`ε_out = 2` **no degrada** la protección. Con el Exp 2 se sabe además que **no aporta**, y se
sabe por qué: la protección que ese valor da por sí mismo es estadísticamente nula.

Importante matizarlo bien: «redundante» aquí significa *redundante frente a este ataque
concreto*. El output shield podría ser la única defensa efectiva frente a un ataque que no
manipule la entrada (por ejemplo, un ataque de pertenencia basado solo en observar scores de
registros reales). Esa afirmación no está medida por estos experimentos y no debe deducirse
de ellos.

### 5.5 El techo del ataque, y por qué recalibra la lectura del Exp 1

A `ε_out = 500`, con ambas capas prácticamente transparentes: **MAE 0.0941 ± 0.0211**, un 8%
de la baseline, con **`success_rate` del 96.7%** (29 de 30 targets reconstruidos dentro de
tolerancia, run 1). El ataque reconstruye V4 **casi exactamente**.

Esto obliga a releer el Exp 1. Allí el mejor punto era 0.2654 (−76.7%) en `ε_in = 500`, y se
interpretó como «el ataque es real y capaz». Ahora se sabe que **0.2654 no era el techo del
instrumento**: era todavía defensa residual del input shield actuando a `ε_in = 500`. El
techo real del ataque está en torno a 0.09.

La consecuencia es favorable al argumento de la tesis, no desfavorable: el instrumento es
sustancialmente mejor de lo que sugería el Exp 1, lo que refuerza —y no debilita— la
afirmación de que su fracaso completo en la configuración desplegada es protección real y no
incapacidad del atacante.

### 5.6 El sesgo de clipping actuando como defensa activa

⚠️ *Los datos de este apartado salen únicamente del CSV de `exp2_run_1`. Consolidarlo con
las tres corridas es un pendiente bloqueante antes de escribirlo en la tesis.*

| ε_out | success_rate | interpretación | `mean_error` |
|---:|---:|---|---:|
| 1 | 6.7% | 2 de 30 = nivel de azar | **+0.159** |
| 2 | 10.0% | 3 de 30 | +0.206 |
| 5 | 16.7% | 5 de 30 | +0.195 |
| 10 | 16.7% | 5 de 30 | +0.222 |
| 50 | 36.7% | 11 de 30 | **+0.267** |
| 100 | 50.0% | 15 de 30 | +0.058 |
| 500 | 96.7% | 29 de 30 | +0.036 |

Dos lecturas independientes.

**Primera: la confirmación aritmética de la abstención.** `ε_out = 1` da `success_rate`
exactamente 6.7% —el nivel de azar de §3.4— y `mean_error` exactamente **+0.1591**, que
coincide con el +0.159 del Exp 1 a `ε_in = 1`. Ese número es la firma del sesgo puro de
`public_mean`. Confirma por una vía distinta al MAE que la seed 1 se abstuvo en los 30
targets.

**Segunda, y más interesante: el sesgo SUBE por encima del piso de abstención en la zona
media.** Va de +0.159 (abstención pura) a **+0.267** en `ε_out = 50`, y solo después se
derrumba a +0.036 en 500.

Merece la pena decir lo que esto significa despacio. Las estimaciones que sobreviven al
guard en la zona media **no son simplemente imprecisas: están sistemáticamente sesgadas
hacia arriba, y más sesgadas de lo que estaría alguien que no hubiera atacado en absoluto.**
El atacante no solo no gana información: se lleva una conclusión desviada en una dirección
concreta.

Esa es la firma esperable del **sesgo de recorte del output shield operando como defensa
activa**, exactamente lo que anticipa el docstring de `run_experiment_out`, y es consistente
con la nota del Exp 1 §4 de que a `ε_out = 50` la señal se comprimía unas 10×. El mecanismo
plausible: el clipping comprime la pendiente medida, el atacante divide por una pendiente
demasiado pequeña, y las estimaciones se desplazan hacia el extremo alto del rango de
recorte.

Si el efecto aguanta con las tres corridas, **es el mejor resultado del Experimento 2 por sí
solo**: dice que el output shield no se limita a añadir ruido, sino que envenena la
estimación de forma dirigida. Es un tipo de defensa cualitativamente distinto y más
interesante que «añadir varianza».

### 5.7 El hueco metodológico del Exp 2

Este es el punto que un tribunal va a encontrar, y conviene cerrarlo antes de la defensa.

Leyendo `run_experiment_out()` en el commit: el bloque `[diag]` se ancla en
`max(epsilons_out)` = 500, es decir en el extremo de **máxima señal**. El comentario del
código lo justifica de forma razonable: sirve para verificar que el instrumento funciona
cuando ninguna capa protege, y si allí abstuviera habría que bisecar antes de reportar nada.
Cumplió esa función perfectamente (§5.8).

**Pero deja la zona protegida sin evidencia diagnóstica.** Y el argumento fuerte del Exp 3
fue precisamente el contrario: `[diag]` **en el punto protegido**, mostrando pendientes
indistinguibles de cero. Eso es lo que convierte «el ataque se abstuvo» en «no había señal
que extraer».

En el Exp 2 no existe ese bloque para `ε_out ∈ {1, 2}`. Y encima se corrió con `K = 200`, la
mitad que en los otros dos experimentos. Así que la pregunta *«¿se abstuvo porque la salida
protege, o porque no promediaste suficiente?»* **queda abierta**, y es exactamente la
pregunta que el Exp 3 sí supo responder para el otro eje.

Lo bueno es que **las dos hipótesis son distinguibles y el propio `[diag]` las distingue**:

| si en `ε_out = 1, 2` el `[diag]` mostrara… | significaría | consecuencia |
|---|---|---|
| signos ~4/4, `slope_mean` ≈ 0, SNR < 5 | **no hay señal**, como en el Exp 3 | protección real; subir K no ayudaría |
| signos **8/0**, `slope_mean` ≈ +0.55, SNR < 5 | **hay señal, tapada por varianza** | protección erosionable: más K la rompe |
| signos **8/0**, `slope_mean` ≈ +0.05 (comprimida) | **clipping**, no varianza | protección por sesgo, parcialmente corregible por el atacante si conoce los límites de recorte |

Nótese que la tercera fila es nueva respecto a lo que se había planteado antes, y sale
directamente de §5.6: si el clipping comprime la pendiente a valores bajos, se puede tener
SNR < 5 sin que la señal haya desaparecido. Ese escenario es intermedio y probablemente el
más realista a `ε_out` bajo.

**La corrida que falta**, barata (dos epsilons en vez de siete, con cuatro veces más
promediado, ≈1.1 corridas del barrido actual):

```bash
for s in 1 2 3; do
  python3 privacy-api/evaluator_attack.py --axis out --K 800 --limit 130 --public-size 100 \
    --secret V4 --epsilon-in-fixed 10000 --epsilons-out 1 2 --snr-threshold 5.0 --seed $s \
    --output-dir results/exp2_K800_run_$s
done
```

Con `--epsilons-out 1 2`, el `[diag]` se anclará automáticamente en `ε_out = 2`, que es justo
la zona que interesa inspeccionar. Es una consecuencia afortunada de que el ancla sea
`max(epsilons_out)`.

**Predicción falsable:** si el MAE a `ε_out ∈ {1, 2}` cae por debajo de la baseline de forma
significativa al cuadruplicar `K`, entonces el output shield aislado no protege: solo
encarece el ataque.

### 5.8 El `[diag]` del Exp 2 y la comparación de pendientes

Las 9 mediciones (3 targets × 3 corridas) a `ε_in = 10000`, `ε_out = 500` fueron **todas
`[PASA]` con signos 8/0**. Puestas junto a las del Exp 1:

| estadístico | Exp 2 (`ε_out=500`, K=200) | Exp 1 (`ε_in=500`, K=400) |
|---|---|---|
| `slope_mean` | +0.539 … +0.621 | +0.462 … +0.625 |
| `slope_std` | **0.020 … 0.054** | **0.068 … 0.181** |
| SNR | 29.7 … 82.4 | 8.7 … 25.5 |
| signos +/− | 8/0 en las 9 | 8/0 en las 15 |

**La pendiente media es la misma en ambos (~0.55–0.60). Lo único que cambia es su
varianza.** Es decir: a ε = 500 nominal, ninguna de las dos capas comprime la señal —solo la
ensucian—, y la capa de entrada la ensucia unas **3.4×** más.

Y ese factor está subestimado. El Exp 2 corrió con `K = 200`, la mitad que el Exp 1, así que
en igualdad de condiciones su `slope_std` debería haber sido **√2 mayor**. Corrigiendo por
eso, el factor real es de **≈4.7×**.

Esto es una confirmación independiente del mismo fenómeno que se ve en el MAE (§7), pero
medida sobre una cantidad mucho más interpretable: la pendiente misma, que es el objeto
físico que el atacante necesita.

---

## 6. Experimento 3 — la configuración desplegada

**Configuración:** secreto `V4`, `ε_in = 1`, `ε_out = 2` (**ambos defaults del governor**),
`K = 400`, `R = 8`, umbral 5.0, span p25/p75, 30 targets, pool de 100 filas, `[diag]`
encendido a propósito, **n = 3** semillas. Coste: bastante menos de una hora las tres
corridas.

**Qué pregunta responde:** no aísla ningún mecanismo. Mide la protección que un atacante
sufriría **contra la API tal como está desplegada**. Es *el* número del Punto 2.

### 6.1 El resultado

| métrica | valor |
|---|---|
| MAE del ataque (media ± std, n=3) | **1.1396 ± 0.0013** |
| IC 95% (t, df=2) | [1.1364, 1.1429] |
| CV | 0.11% |
| Baseline | **1.1389** |
| Reducción vs baseline | **−0.06% (≈ 0%)** |
| ¿supera la baseline? | **no — protección total** |

**El IC 95% contiene la baseline**, luego el MAE del ataque es estadísticamente
indistinguible de adivinar la media pública. Con la configuración real, el ataque **no le da
al atacante ninguna ventaja sobre no atacar**.

Los datos crudos:

| seed | MAE | lectura |
|---:|---:|---|
| 1 | 1.1411 | 29 targets abstienen; 1 pasa el guard por azar y estima peor que la media |
| 2 | 1.1389 | abstención total (MAE = baseline exacto) |
| 3 | 1.1389 | abstención total (MAE = baseline exacto) |

### 6.2 Por qué la seed 1 queda por *encima* de la baseline

Es un detalle que parece un defecto y en realidad refuerza la conclusión, así que conviene
saber explicarlo.

Si los 30 targets abstuvieran, el MAE sería 1.1389 exacto, como en las seeds 2 y 3. Que la
seed 1 quede un pelo por arriba (+0.19%) significa que **un target —de los 27 que el `[diag]`
no inspecciona— pasó el guard por azar** y produjo una estimación *peor* que `public_mean`.
Es un falso positivo residual del guard, perfectamente esperable: el umbral de Bonferroni
reduce la tasa de falsos positivos, no la anula.

Lo que dice ese target es que, con esta configuración, **el ataque ocasionalmente se hace
daño a sí mismo**: el atacante ni siquiera consigue de forma fiable *no empeorar* respecto a
no atacar. Es una conclusión más fuerte que «el ataque no gana».

### 6.3 El `[diag]`: la pieza que convierte la abstención en evidencia

Esta es la sección que sostiene el argumento entero, y es la que hay que saber defender.

La objeción previsible de un revisor es: *«¿se abstuvo porque la API protege, o porque no
promediaste lo suficiente?»*. Es una objeción legítima: la abstención se dispara cuando el
SNR es bajo, y el SNR sube con `K`, así que en principio una abstención podría ser un
artefacto del presupuesto de consultas.

El bloque `[diag]` la responde. Las 9 mediciones (3 targets × 3 corridas) fueron **todas
`[ABSTIENE]`**:

```
seed 1  target[0]  slope_mean=+0.0057  SNR=0.21  signos 3/5  [ABSTIENE]
        target[15] slope_mean=-0.0325  SNR=1.29  signos 3/5  [ABSTIENE]
        target[29] slope_mean=+0.0370  SNR=1.05  signos 5/3  [ABSTIENE]
seed 2  target[0]  slope_mean=-0.0391  SNR=2.38  signos 2/6  [ABSTIENE]
        target[15] slope_mean=+0.0304  SNR=0.80  signos 5/3  [ABSTIENE]
        target[29] slope_mean=-0.0245  SNR=0.56  signos 3/5  [ABSTIENE]
seed 3  target[0]  slope_mean=+0.0399  SNR=2.10  signos 6/2  [ABSTIENE]
        target[15] slope_mean=-0.0223  SNR=0.55  signos 4/4  [ABSTIENE]
        target[29] slope_mean=+0.0038  SNR=0.14  signos 4/4  [ABSTIENE]
```

Tres cosas hay que mirar en ese bloque, y las tres apuntan en la misma dirección:

1. **`slope_mean` está en ±0.04**, frente a los +0.55 que se miden cuando hay señal real.
   La pendiente prácticamente ha desaparecido.
2. **Los signos son ~50/50 y mezclados.** No hay una sola medición con 8/0 ni 7/1. Cuando
   hay señal, las 8 estimaciones independientes coinciden en el signo (es lo que se ve en
   los Exps 1 y 2). Aquí la dirección del efecto cambia de una repetición a otra: eso es la
   definición operativa de «no hay efecto».
3. **El SNR va de 0.14 a 2.38**, todos muy por debajo del umbral de 5.0, y no marginalmente:
   el más alto está a menos de la mitad.

El argumento que se construye con esto:

> No hay pendiente que promediar. Subir `K` no acercaría la medición a la pendiente
> verdadera, porque la medición ya está convergiendo a la media del ruido, que es cero. La
> abstención es la respuesta correcta del atacante, no un déficit de muestreo.

Y el contraste con el `[diag]` del Exp 1 es la prueba, porque son el mismo instrumento
midiendo dos situaciones distintas:

| estadístico | Exp 3 (config. real) | Exp 1 (`ε_in=500`, transparente) |
|---|---|---|
| `slope_mean` | −0.039 … +0.040 | +0.46 … +0.63 |
| SNR | 0.14 … 2.38 | 8.7 … 25.5 |
| signos +/− | ~50/50, mezclados | 8/0 |
| veredicto | ABSTIENE (9/9) | PASA (15/15) |

**La diferencia entre los dos bloques no es de calibración del atacante: es la presencia o
ausencia de señal.** Donde hay señal recuperable, la pendiente es estable, unidireccional y
con SNR de dos dígitos. Donde no la hay, es indistinguible de cero.

### 6.4 Comparabilidad con el Exp 1 y la lectura sobre el piso de protección

Misma baseline (1.1389), mismos 30 targets, mismo pool, misma `K`, mismo umbral, misma
`tol_abs`. En el Exp 1, `ε_in = 1` con la salida **transparente** (`ε_out = 10000`) ya daba
1.1389 exacto. Añadir ahora `ε_out = 2` **no degrada esa protección**: la deja igual, solo
empujada +0.0007 por el falso positivo ocasional de la seed 1.

> **El piso de protección lo pone el input shield. El output shield a `ε_out = 2` no lo
> mueve.** A `ε_in = 1` la entrada ya está saturada de ruido, de modo que el ataque no tiene
> señal que el output shield deba proteger.

### 6.5 Por qué aquí el `[diag]` se dejó encendido

En el Exp 1 el `[diag]` se identificó como candidato a eliminar: gasta `R·2·K = 6.400`
consultas por cada uno de los 3 targets que inspecciona, unas **19.200 por corrida**, y una
vez validado el sistema no aportaba nada al barrido.

En el Exp 3 se dejó encendido **a propósito**, y con razón: es literalmente la evidencia de
§6.3, sin la cual el resultado principal se queda sin defensa frente a la objeción del
promediado. Costó ~19.200 consultas de un total de ~235.000, un 8%. Barato para lo que
compra.

Esa es también la razón por la que en la corrida pendiente de §5.7 hay que dejarlo
encendido: es el objetivo de esa corrida, no un residuo de la depuración.

---

## 7. La síntesis: por qué las dos capas se comportan de forma distinta

Esta sección es el aporte conceptual del conjunto, y es donde hay que tener más cuidado con
lo que se afirma.

### 7.1 El hecho medido

En los 5 valores de ε que comparten los dos barridos —y recordando que la configuración es
simétrica, con la capa no barrida en 10000 en ambos casos—:

| ε nominal | Exp 1 (barre `ε_in`) | Exp 2 (barre `ε_out`) | ¿cuál filtra más? |
|---:|---:|---:|:---|
| 1 | 1.1389 | 1.1308 | salida |
| 10 | 1.1389 | 1.0318 | salida |
| 50 | 1.0700 | 0.7026 | salida |
| 100 | 0.7292 | 0.4210 | salida |
| 500 | 0.2654 | 0.0941 | salida |

La ordenación es consistente en los 5 puntos, sin una sola excepción. Y va **en contra del
handicap** del Exp 2, que corrió con `K = 200` frente a `K = 400`, es decir con la mitad de
promediado. Que aun así filtre más en todos los puntos refuerza la dirección del efecto.

Hay además una confirmación independiente en la pendiente misma (§5.8): a ε = 500, la capa
de entrada ensucia la pendiente ~4.7× más que la de salida (corrigiendo por K), mientras que
la pendiente media es la misma en ambos casos.

**Lo que se puede afirmar:** *a igual ε nominal, la capa de salida deja pasar más
información sobre el secreto.* Repito el caveat de §1.3: convertir esto en «la capa de
entrada protege más» requiere replotear ambos ejes contra `b = Δ/ε` con las sensibilidades
reales de cada capa, y ese paso no se ha dado.

### 7.2 La explicación mecanicista, y hasta dónde llega

**Hipótesis.** Las dos capas atacan partes distintas del cociente que el atacante calcula:

```
                    |slope_mean|
        SNR  =  ───────────────────
                  slope_std / √R
```

- **El output shield ataca el denominador.** Añade ruido de media cero al score. La curva de
  respuesta subyacente —cómo varía la salida esperada del modelo en función de V4— **no
  cambia**. Solo se observa con más varianza. El promediado sobre `K` consultas ataca
  directamente esa varianza a razón de `1/√K`.
- **El input shield ataca el numerador.** El ruido no está en la observación sino en la
  *cantidad que se observa*. La respuesta esperada deja de ser `σ(f(V4))` y pasa a ser el
  promedio de `σ(f(V4 + η))` sobre el ruido `η`: una versión **suavizada y más plana** de la
  curva original. El atacante que promedia infinitas consultas no converge a la curva
  verdadera: converge a la curva aplanada.

La evidencia que apoya esto está en las pendientes: a `ε_in = 1` la `slope_mean` cae a ±0.04
frente a los +0.55 de referencia (§6.3), mientras que a `ε_out = 500` la pendiente sigue
valiendo +0.55–0.62 y lo único que cambia es su dispersión (§5.8).

**Hasta dónde llega la hipótesis, y dónde hay que frenar.** Es tentador concluir «la
protección del input shield no es erosionable por más consultas y la del output shield sí».
**Esa conclusión es más fuerte de lo que los datos sostienen**, y conviene ver por qué:

Aunque la curva se aplane, si sigue siendo **monótona** en V4, el atacante que la mide con
suficiente precisión sigue pudiendo invertirla. Una pendiente de 0.005 en vez de 0.55 no es
información cero: es información que cuesta `(0.55/0.005)² = 12.100` veces más consultas
recuperar. Es decir, aplanar la señal encarece el ataque de forma **cuadrática**, que es
brutal pero finito.

La pregunta decisiva es si a `ε_in = 1` la pendiente es **muy pequeña** o **exactamente
cero**, y los datos actuales no la distinguen: la medición del `[diag]` es compatible con
ambas (media ±0.04 con signos aleatorios es compatible con 0.0 y también con 0.004).

- Si es **muy pequeña**: la protección tiene un precio de consultas finito, calculable, y se
  puede extrapolar cuántas consultas necesitaría un atacante paciente.
- Si es **exactamente cero**: la protección es de otra naturaleza. Un mecanismo plausible
  para que lo sea es el **recorte del servidor**: si el input shield recorta el valor ruidoso
  de vuelta al rango válido antes de pasarlo al modelo, con un ruido cuyo ancho supera
  ampliamente ese rango el valor entregado al modelo satura contra los límites casi siempre,
  y la dependencia respecto al valor verdadero puede anularse de verdad.

**Cuál de las dos es no se puede saber desde el lado del atacante.** Requiere leer la
implementación del shield en el servidor (no la he revisado; solo he leído el cliente de
ataque) y confirmar si hay recorte post-ruido y con qué límites.

### 7.3 La afirmación calibrada

Juntando todo, esto es lo que se puede sostener sin sobrepasarse:

> Las dos capas encarecen el ataque, pero por vías distintas. El output shield añade varianza
> de observación sobre una señal intacta, y esa varianza se combate con promediado a razón
> de `1/√K`: su protección tiene un precio en consultas conocido y moderado. El input shield
> atenúa la señal misma, lo que encarece el ataque de forma cuadrática en el factor de
> atenuación: a `ε_in = 1` la atenuación medida lleva la pendiente por debajo del umbral de
> detección del instrumento, de modo que **el precio en consultas queda acotado por abajo
> pero no por arriba desde estos datos**. Determinar si es finito o infinito requiere medir
> la pendiente con más precisión de la que estos experimentos alcanzan.

Es una afirmación menos vistosa que «una capa es erosionable y la otra no», pero es la que
se sostiene, y es más difícil de tumbar en una defensa.

### 7.4 El punto formal que un tribunal va a plantear

Hay una tensión entre lo que garantiza la privacidad diferencial en el papel y lo que miden
estos experimentos, y conviene anticiparla.

Por **composición secuencial**, si cada request tiene garantía ε, entonces `K` requests
tienen garantía `K·ε`. Con `K = 400` y `ε_in = 1`, el presupuesto compuesto de una sola
estimación es **400**, no 1. Formalmente, el atacante del Exp 3 opera contra una garantía
compuesta enorme, y sin embargo fracasa por completo.

Eso no es una contradicción: es la diferencia entre una **cota del peor caso** y el
**rendimiento empírico de un ataque concreto**. La cota de DP vale sobre todos los
adversarios posibles y todas las funciones que puedan querer calcular; este ataque concreto
es mucho menos eficiente que el peor adversario imaginable. Los experimentos miden **fuga
empírica**, no la cota formal, y las dos divergen mucho.

La consecuencia práctica es la parte importante y hay que decirla claramente: **la
configuración desplegada no ofrece una garantía formal de ε = 1 frente a un atacante que
consulta repetidamente, porque no hay contabilidad de presupuesto por cliente.** Lo que
ofrece, y es lo que estos experimentos demuestran, es que *este* ataque de inversión no
consigue nada con 400 repeticiones por sondeo.

De ahí sale la recomendación de diseño más valiosa de todo el trabajo, que además es
gratuita en términos de utilidad del modelo:

> **La defensa que falta no es más ruido, es contabilidad de presupuesto.** Un acumulador
> por cliente que sume el ε consumido y corte, o simplemente un rate limit sobre requests
> casi idénticos, elimina de raíz la posibilidad de promediar. Y el promediado es el único
> mecanismo mediante el cual el atacante derrota al output shield.

---

## 8. Lo que NO se puede afirmar con estos datos

Vale la pena tener esta lista explícita, porque cada punto es una pregunta que puede caer en
la defensa.

1. **Que el input shield proteja «más» que el output shield.** Solo que a igual ε nominal
   filtra menos. Las unidades no son comparables sin normalizar por sensibilidad (§1.3).
2. **Que la protección del input shield sea inerosionable.** Solo que su precio en consultas
   está fuera del alcance del instrumento actual (§7.2).
3. **Que la protección del output shield a `ε_out = 2` sea real.** Solo que a `K = 200` no
   se detecta ventaja. Falta la corrida de §5.7.
4. **Que el output shield sea redundante en general.** Solo que lo es *frente a este ataque
   de inversión*, y dado que el input shield ya está en 1 (§5.4).
5. **Que estos resultados generalicen a otras features.** Todo está medido sobre `V4`. Hay
   indicios de que `V14`, con una distribución más compacta, se comporta distinto en cuanto
   al SNR (§4.9).
6. **Que generalicen a otros ataques.** Un ataque de pertenencia, o uno que use varias
   features simultáneamente, o uno que ajuste un modelo sustituto en vez de una pendiente de
   dos puntos, podrían comportarse de otra forma.
7. **Que el span p25/p75 sea el que maximiza la potencia del ataque.** No está calibrado, y
   si un span mejor existiera, la protección medida sería optimista (§4.9).
8. **Que los resultados sean reproducibles bit a bit.** No lo son, por diseño (§3.7).

---

## 9. Código nuevo del commit que aún no se ha explotado

El commit `0542d54` añadió, además del eje `out`, dos piezas que todavía no han producido
resultados:

**`run_floor_experiment()` / `--floor`.** Mide, en un punto DP fijo (por defecto
`ε_in = 500`, `ε_out = 10000`), el **piso de protección** descomponiendo el error en sesgo y
varianza. Repite la estimación `floor_runs = 20` veces **sobre el mismo target** y calcula,
por target, el sesgo (`estimate_mean − valor_verdadero`) y la varianza entre repeticiones,
agregándolos después con `build_bias_variance_curve()`.

Es exactamente el instrumento que responde la pregunta abierta de §7.2, pero desde el otro
lado: dice **hasta dónde podría llegar un atacante con infinitas consultas en ese punto**,
porque la varianza se promedia y el sesgo no. Si `piso_sesgo_abs` resulta ser grande, hay un
límite irreducible; si resulta ser pequeño, todo el error es varianza y por tanto erosionable
con más `K`. Correrlo en `ε_in = 1` (no en 500) sería la medición más informativa que falta
en todo el proyecto.

Detalle de implementación que conviene arreglar antes: el agregador usa correctamente
`|bias(t)|` y el RMS en vez del sesgo con signo —el propio comentario del código explica por
qué: los sesgos con signo se cancelan entre targets y darían un piso artificialmente bajo—.
Bien visto.

**Una inconsistencia menor a corregir.** La firma de `run_floor_experiment()` tiene
`snr_threshold: float = 2.0` por defecto, mientras que el umbral validado es 5.0. En las
corridas por CLI no afecta, porque `main()` lo sobrescribe con `args.snr_threshold`; pero es
una trampa si alguna vez se llama la función desde código. Y el Exp 1 documentó que con
umbral 2.0 el error familiar es del 79% y el MAE queda *por encima* de la baseline (§2.5).
Alinear el default a 5.0.

**Otra menor.** La columna clave del CSV se llama `epsilon` en los dos ejes —el propio código
lo avisa con un comentario `# OJO`—, y solo el nombre del directorio (`exp2_*`) distingue si
contiene `ε_in` o `ε_out`. Añadir una columna `axis` al CSV haría los artefactos
autodescriptivos y evitaría un error de interpretación dentro de seis meses.

---

## 10. Reconciliación: por qué `RESUMEN_curva_proteccion.md` contradice al Exp 1

Este apartado existe porque hay cuatro documentos en circulación con números distintos para
lo que parece lo mismo, y eso es una fuente segura de confusión.

`RESUMEN_curva_proteccion.md` es **anterior** a la introducción del flag `--seed` y reporta
**una sola corrida**. Está superado por `RESULTADOS_EXPERIMENTO_1.md` (n=5) en todo lo que se
solapa. Comparando:

| ε_in | RESUMEN (1 corrida) | Exp 1 (media de 5) |
|---:|---:|---:|
| 30 | 1.1530 | 1.1316 ± 0.0209 |
| 50 | 1.0395 | 1.0700 ± 0.0375 |
| 100 | 0.7548 | 0.7292 ± 0.0537 |
| 300 | 0.3999 | 0.3685 ± 0.0450 |
| 500 | 0.2046 | 0.2654 ± 0.0160 |

Las tres discrepancias que importan:

**La rodilla se movió de «30–50» a «50–100».** En parte por los datos (con n=5, `ε_in=30`
resultó no significativo) y en parte por un cambio de definición no declarado: el RESUMEN
usaba «primer punto que gana» y el Exp 1 usa «dónde se derrumba la protección». Ver §4.4:
conviene reportar las dos cosas con nombres distintos.

**El «−82%» se convirtió en «−76.7%».** El −82% salía del 0.2046 de una corrida única; la
media de las cinco es 0.2654. Aquella corrida era un extremo de la distribución. **Usar
−76.7%.** Y si se quiere el número de máxima fuga del proyecto, el correcto es el −91.7% del
Exp 2, que es otro eje y otra pregunta.

**El «±35% de varianza corrida a corrida», marcado en rojo en el RESUMEN, no existe.** Era
una extrapolación desde dos observaciones (§4.5). El CV real es del 5–12%. Toda la sección 3
del RESUMEN, incluidas sus tres consecuencias, quedó resuelta por la propia acción que
recomendaba: repetir con varias semillas.

**Recomendación práctica:** marcar `RESUMEN_curva_proteccion.md` como obsoleto en su
encabezado, o archivarlo, en vez de dejarlo circulando junto a los otros tres. Su valor
ahora es histórico: documenta bien por qué hacían falta las barras de error.

---

## 11. Qué falta, ordenado por prioridad

### Bloqueantes antes de escribir el capítulo

1. **La corrida de `ε_out ∈ {1,2}` con `K = 800` y lectura del `[diag]`** (§5.7). Sin ella,
   la afirmación central del Exp 2 está condicionada a `K = 200` y es tumbable.
2. **Consolidar `success_rate` y `mean_error` de todas las corridas** de los tres
   experimentos. Ahora mismo, el Exp 1 los reporta solo de la corrida 5 y el Exp 2 solo de la
   corrida 1. En particular, verificar si el pico de sesgo de `ε_out = 50` (+0.267) aguanta:
   si aguanta, es un hallazgo por derecho propio (§5.6).
3. **Extraer `success_rate` y `mean_error` del Exp 3** de
   `results/exp3_run_*/inversion_curve.json`, que no se leían del stdout. Se espera
   `success_rate` en el nivel de azar (6.7%) y `mean_error` cerca de +0.159, lo que
   confirmaría la abstención por una tercera vía.

### Importantes

4. **Correr `--floor` en `ε_in = 1`** (§9). Es la medición que decide si la protección del
   input shield tiene precio finito o infinito, que es la pregunta abierta de §7.2. El código
   ya existe.
5. **Un barrido de `[diag]` a lo largo de la rodilla** (`ε_in` = 30, 50, 100, 300) para
   obtener la curva de **atenuación de la pendiente** frente a ε. Ahora mismo solo hay dos
   puntos, `ε_in = 1` (≈0) y `ε_in = 500` (≈0.55), y nada en medio: justo donde ocurre todo.
   Esa curva **explicaría** la curva de MAE en vez de solo describirla, y es probablemente la
   figura más informativa que se puede añadir a la tesis.
6. **Replotear ambos ejes contra `b = Δ/ε`** (§1.3), extrayendo las sensibilidades reales del
   código del servidor.
7. **Recalibrar el span** (§4.9): p5/p95 vs p25/p75 vs p40/p60, con varias semillas, y
   considerar el span en unidades de σ.
8. **Añadir `ε_out ∈ {20, 30}`** para cerrar la rodilla del eje de salida a un bracket 2× e
   igualarla con la del eje de entrada (§5.3).
9. **Repetir el Exp 2 con `K = 400`** para eliminar la única asimetría de configuración
   frente al Exp 1. Puede quedar parcialmente cubierto por el punto 1.

### Menores

10. Alinear el `snr_threshold` por defecto de `run_floor_experiment()` a 5.0 (§9).
11. Añadir una columna `axis` a los CSV (§9).
12. Marcar `RESUMEN_curva_proteccion.md` como obsoleto (§10).
13. ¿Subir Exp 2 y Exp 3 a n=5? Los CV son del 0.11% (Exp 3) y del 1–2% en la zona baja del
    Exp 2. Estadísticamente n=3 basta de sobra. Solo homogeneizaría las tablas. Es una
    decisión de presentación, no de rigor.

---

## 12. Preguntas previsibles de un tribunal, y cómo están respondidas

**«¿Cómo sabes que el ataque es bueno? Quizá falla por incompetencia.»**
Respondido. El mismo ataque, con las capas transparentes, reconstruye 29 de 30 targets con
un MAE del 8% de la baseline (Exp 2, `ε_out = 500`). El instrumento está calibrado en el
mismo montaje, con los mismos targets y la misma tolerancia.

**«El ataque se abstiene. ¿No será que no promediaste suficiente?»**
Respondido **para el Exp 3** por el `[diag]` (§6.3): pendiente en ±0.04, signos aleatorios,
SNR entre 0.14 y 2.38. No hay pendiente que promediar. **No respondido todavía para el
Exp 2** — es la corrida pendiente de §5.7.

**«¿Elegiste el umbral del guard para que saliera bien?»**
Respondido. Bonferroni sobre 30 tests con t de df=7, valor exacto 4.94 (§2.5). El criterio es
*a priori*. Además el `[diag]` del Exp 1 muestra SNR de 8.7 a 25.5 donde hay señal real, con
lo que el umbral no está sacrificando señal legítima.

**«Un solo run no demuestra nada.»**
Respondido. n=5 en el Exp 1, n=3 en los Exps 2 y 3, con IC de t de Student. Y hay un ejemplo
documentado de por qué hacía falta: el bulto de `ε_in = 30` desapareció al pasar a n=5
(§4.3), igual que la supuesta varianza del ±35% (§4.5).

**«¿Y el presupuesto de privacidad? Con 400 consultas el ε compuesto es 400, no 1.»**
Respondido en §7.4, y es la pregunta más incisiva de la lista. Hay que reconocer sin rodeos
que la configuración desplegada **no ofrece la garantía formal de ε = 1** frente a un
atacante que repite consultas, porque no hay contabilidad de presupuesto. Lo que se demuestra
es fuga empírica nula frente a este ataque. Y la conclusión de diseño que sale de ahí —añadir
un acumulador de presupuesto o un rate limit sobre requests casi idénticos— es una aportación
del trabajo, no una debilidad.

**«Solo has probado una feature.»**
No respondido. Es la limitación de alcance de §8.5 y hay que declararla. Hay indicios de que
`V14`, con distribución más compacta, interactúa distinto con el span.

**«El atacante necesita conocer todas las demás features de la víctima. ¿Es realista?»**
No respondido experimentalmente, y hay que declararlo como supuesto del modelo de amenaza
(§2.1). El argumento defendible es que se trata deliberadamente de un **adversario fuerte**:
si la defensa aguanta contra alguien que lo sabe casi todo, aguanta con más razón contra
alguien que sabe menos. Pero conviene decirlo como decisión de diseño del experimento, no
darlo por supuesto.
