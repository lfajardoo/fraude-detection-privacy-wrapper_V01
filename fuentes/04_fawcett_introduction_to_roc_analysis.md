# An Introduction to ROC Analysis

**Autor:** Tom Fawcett — Institute for the Study of Learning and Expertise, Palo Alto, CA
**Publicación:** Pattern Recognition Letters 27 (2006) 861–874, doi:10.1016/j.patrec.2005.10.010

## Resumen

Los gráficos ROC (*Receiver Operating Characteristics*) son útiles para organizar clasificadores y visualizar su desempeño. Se usan comúnmente en decisiones médicas y, cada vez más, en machine learning y minería de datos. Este artículo es una introducción a los gráficos ROC y una guía para su uso en investigación, cubriendo también errores comunes.

---

## 1. Introducción

Uso histórico: teoría de detección de señales (Egan, 1975; Swets et al., 2000), sistemas de diagnóstico médico (Swets, 1988). En ML: Spackman (1989) fue de los primeros en aplicar curvas ROC; su uso se incrementó al reconocerse que la accuracy simple es una métrica pobre en dominios con distribución de clases sesgada y costos de error desiguales (Provost y Fawcett, 1997, 1998).

---

## 2. Desempeño de clasificadores

Instancia $I$ mapeada a $\{p,n\}$ (positivo/negativo real). Un **clasificador** mapea instancias a clases predichas $\{Y,N\}$.

**Matriz de confusión** (tabla de contingencia 2×2):

| | Predicho Y | Predicho N |
|---|---|---|
| Real p | True Positives (TP) | False Negatives (FN) |
| Real n | False Positives (FP) | True Negatives (TN) |

$$
\text{tp rate} = \frac{\text{Positivos correctamente clasificados}}{\text{Total positivos}}
$$

$$
\text{fp rate} = \frac{\text{Negativos incorrectamente clasificados}}{\text{Total negativos}}
$$

**Términos asociados:**

$$
\text{sensitivity} = \text{recall}, \qquad \text{specificity} = \frac{TN}{FP+TN} = 1 - \text{fp rate}, \qquad \text{positive predictive value} = \text{precision}
$$

---

## 3. Espacio ROC

Gráfico 2D: $tp\ rate$ en eje Y, $fp\ rate$ en eje X. Un **clasificador discreto** (produce solo etiqueta de clase) genera un único punto $(fp\ rate, tp\ rate)$.

**Puntos notables:**
- $(0,0)$: nunca predecir positivo — sin FP pero sin TP.
- $(1,1)$: siempre predecir positivo.
- $(0,1)$: clasificación perfecta.

**Orientación:** un punto es "mejor" que otro si está al **noroeste** (mayor tp rate, menor fp rate, o ambos). Clasificadores cerca del eje X izquierdo: "conservadores" (pocos FP, pero también bajo TP). Clasificadores en la esquina superior derecha: "liberales" (alto TP pero también alto FP).

### 3.1 Desempeño aleatorio

La diagonal $y=x$ representa la estrategia de adivinar aleatoriamente. Si un clasificador adivina positivo el 50% del tiempo → punto $(0.5,0.5)$; si adivina el 90% → $(0.9,0.9)$. La diagonal es el "punto de no información".

Cualquier punto en el triángulo inferior derecho (peor que azar) puede **negarse** (invertir las decisiones) para producir un punto en el triángulo superior izquierdo (mejor que azar). Un clasificador **debajo** de la diagonal tiene información útil pero la aplica incorrectamente (Flach y Wu, 2003).

---

## 4. Curvas en espacio ROC

Clasificadores que producen una **puntuación o probabilidad** (Naive Bayes, redes neuronales) pueden convertirse en clasificadores discretos aplicando un **umbral**: si la puntuación supera el umbral → Y, si no → N. Variar el umbral de $+\infty$ a $-\infty$ traza una curva en espacio ROC.

**Método eficiente:** ordenar instancias por puntaje decreciente, procesar una por una, actualizando TP y FP, generando un punto tras cada instancia.

### 4.1 Puntuaciones relativas vs. absolutas

**Punto crítico:** las curvas ROC miden la capacidad del clasificador de producir puntuaciones **relativas** correctas (ranking), no probabilidades calibradas. Un clasificador puede tener AUC=1 (ranking perfecto) pero mala accuracy con un umbral fijo de 0.5 si sus puntuaciones no están calibradas. La solución: calibrar las puntuaciones (Zadrozny y Elkan, 2001) o usar métodos ROC que eligen puntos operativos basados en desempeño relativo (Provost y Fawcett, 1998, 2001).

**Consecuencia:** no se deben comparar puntuaciones de clasificadores entre distintas clases de modelo (rangos distintos, ej. $[0,1]$ vs $[-1,1]$).

### 4.2 Sesgo de clase (Class Skew)

**Propiedad clave de las curvas ROC:** son **insensibles a cambios en la distribución de clases**. Esto se debe a que $tp\ rate$ y $fp\ rate$ son razones puramente columnares (cada una usa solo una columna de la matriz de confusión: positivos reales o negativos reales). En cambio, métricas como accuracy, precision, lift y F-score usan valores de **ambas** columnas → son sensibles al sesgo de clase.

Sesgos de clase de $10^1$–$10^2$ son comunes en el mundo real; se han observado hasta $10^6$. Ejemplos: epidemias (medicina), fraude (varía mes a mes), defectos de manufactura.

**Precision-Recall vs ROC:** las curvas Precision-Recall **sí** cambian con la distribución de clases (a diferencia de ROC), lo cual puede alterar la conclusión sobre qué clasificador es superior si la distribución cambia.

### 4.3 Creación de clasificadores con puntuación

- **Árboles de decisión:** usar la proporción de clases en el nodo hoja como score.
- **Reglas de aprendizaje:** usar la confianza de la regla.
- **MetaCost** (Domingos, 1999): bagging para generar un ensamble; el conjunto de votos puede usarse como score (aunque MetaCost fue diseñado en la dirección opuesta, para producir un clasificador discreto).

---

## 5. Generación eficiente de curvas ROC

Se explota la **monotonicidad**: toda instancia clasificada positiva con un umbral dado lo será también con umbrales menores. Basta ordenar por score decreciente y recorrer linealmente.

**Algoritmo 1 (esquema):**

```
Entrada: L (instancias de test), f(i) (score), P, N (conteos reales)
Salida: R (lista de puntos ROC, creciente en fp rate)

Lsorted ← L ordenado decreciente por f
FP ← TP ← 0
R ← lista vacía
fprev ← -∞
para i = 1 hasta |Lsorted|:
    si f(i) ≠ fprev:
        push (FP/N, TP/P) a R
        fprev ← f(i)
    si Lsorted[i] es positivo: TP ← TP+1
    si no: FP ← FP+1
push (FP/N, TP/P) a R   # (1,1)
```

Complejidad: $O(n\log n)$ por el ordenamiento + $O(n)$ del recorrido.

**Manejo de empates:** cuando varias instancias tienen el mismo score, no se debe emitir un punto ROC hasta procesar **todas** las instancias con ese score empatado — de lo contrario el orden de procesamiento (todo positivo primero vs. todo negativo primero) genera curvas "optimista" o "pesimista" muy distintas. La curva correcta es el **promedio** (la diagonal del rectángulo formado por ambos extremos).

---

## 6. La envolvente convexa ROC (ROC Convex Hull)

Dos puntos $(FP_1,TP_1)$ y $(FP_2,TP_2)$ tienen el mismo desempeño esperado si:

$$
\frac{TP_2-TP_1}{FP_2-FP_1} = \frac{c(Y,n)\, p(n)}{c(N,p)\, p(p)} = m \tag{1}
$$

Esta ecuación define la pendiente de una **línea de iso-desempeño (iso-performance)**. Todos los clasificadores sobre una línea de pendiente $m$ tienen el mismo costo esperado. Un clasificador es **potencialmente óptimo** si y solo si está en la **envolvente convexa** (ROCCH) del conjunto de puntos.

**Aplicación práctica:** dado un escenario de costos/skew, se calcula $m$ vía (1), y se busca la línea "más al noroeste" de esa pendiente tangente a la envolvente convexa — el punto de tangencia es el clasificador óptimo para esas condiciones.

Clasificadores intermedios en la envolvente pueden generarse **interpolando** entre dos clasificadores adyacentes (ver Sección 10).

---

## 7. Área bajo la curva ROC (AUC)

Reduce el desempeño 2D a un escalar. Como es una porción del área del cuadrado unitario: $\text{AUC} \in [0,1]$. Un clasificador aleatorio tiene AUC=0.5 (diagonal); ningún clasificador razonable debería tener AUC < 0.5.

**Propiedad estadística clave:** el AUC es equivalente a la probabilidad de que el clasificador rankee una instancia positiva elegida al azar por encima de una instancia negativa elegida al azar. Equivalente al test de rangos de **Wilcoxon** (Hanley y McNeil, 1982). Relación con el **coeficiente de Gini**: $\text{Gini}+1 = 2\cdot\text{AUC}$ (Hand y Till, 2001).

**Advertencia:** un clasificador con AUC alto puede tener peor desempeño que uno con AUC bajo en una región **específica** del espacio ROC.

**Algoritmo 2 (esquema, cálculo de AUC):** análogo al Algoritmo 1, pero acumulando áreas de trapezoides en lugar de rectángulos (para promediar correctamente el efecto de empates, como en la Sección 5):

```
A ← 0
... (mismo recorrido que Algoritmo 1)
si f(i) ≠ fprev:
    A ← A + TRAPEZOID_AREA(FP, FPprev, TP, TPprev)
    ...
A ← A / (P·N)   # escalar al cuadrado unitario

función TRAPEZOID_AREA(X1,X2,Y1,Y2):
    Base ← |X1-X2|
    Heightavg ← (Y1+Y2)/2
    retornar Base · Heightavg
```

---

## 8. Promediado de curvas ROC

Comparar clasificadores solo por dominancia visual en espacio ROC es engañoso (análogo a comparar el máximo de un conjunto de cifras de accuracy sin medida de varianza). Se necesitan medidas de varianza.

**Dos métodos de promediado:**

### 8.1 Promediado vertical

Fija $fp\ rate$ y promedia los $tp\ rate$ correspondientes de varias curvas (tratando cada curva como función $tp\ rate = R_i(fp\ rate)$, interpolando si es necesario):

$$
\hat R(fp\ rate) = \text{mean}[R_i(fp\ rate)]
$$

Intervalos de confianza de la media de $tp\ rate$: asumiendo distribución binomial.

### 8.2 Promediado por umbral

Alternativa cuando $fp\ rate$ no está bajo control directo del investigador (Holte, 2002). Se muestrean umbrales (scores del clasificador) y para cada uno se promedian los puntos correspondientes de cada curva ROC en ambos ejes (X e Y), con barras de confianza en ambas direcciones.

**Limitación:** requiere el score asignado a cada punto; y ROC de distintas clases de modelo pueden no ser comparables por scores inconmensurables (ver 4.1).

Macskassy y Provost (2004) investigan bandas de confianza adicionales (regiones de confianza conjuntas simultáneas, bandas de Working–Hotelling, bandas de ancho fijo).

---

## 9. Problemas con más de dos clases

### 9.1 Gráficos ROC multiclase

Con $n$ clases, la matriz de confusión es $n\times n$: $n$ aciertos (diagonal) y $n^2-n$ posibles errores. Con solo 3 clases, la superficie relevante tiene $3^2-3=6$ dimensiones.

**Formulación de referencia de clase (class reference):** para cada clase $c_i$, se genera un gráfico ROC tratando $c_i$ como positivo y la unión del resto como negativo:

$$
P_i = c_i, \qquad N_i = \bigcup_{j\ne i} c_j \in C \tag{2, 3}
$$

**Advertencia:** esto compromete la insensibilidad al sesgo de clase, porque $N_i$ es la unión de $n-1$ clases, y cambios en la prevalencia relativa *dentro* de esas clases pueden alterar la curva ROC de $c_i$ incluso si el desempeño real del clasificador respecto a $c_i$ no cambió.

### 9.2 AUC multiclase

**Provost y Domingos (2001):**

$$
\text{AUC}_{\text{total}} = \sum_{c_i \in C} \text{AUC}(c_i) \cdot p(c_i)
$$

Complejidad: $O(|C| \cdot n\log n)$. Ventaja: se genera directamente de curvas de referencia de clase (visualizables). Desventaja: sensible a distribución de clases y costos de error (por la razón anterior).

**Hand y Till (2001):** medida $M$, insensible a distribución de clases, basada en la interpretación probabilística del AUC:

$$
\text{AUC}_{\text{total}} = \frac{2}{|C|(|C|-1)} \sum_{\{c_i,c_j\}\in C} \text{AUC}(c_i,c_j)
$$

Suma sobre todos los pares distintos de clases ($|C|(|C|-1)/2$ pares). Complejidad: $O(|C|^2\, n\log n)$. Bien justificada e insensible a distribución de clases, pero sin interpretación geométrica sencilla de la superficie cuya área se calcula.

---

## 10. Interpolación de clasificadores

Cuando el desempeño deseado no lo produce exactamente ningún clasificador disponible, pero se encuentra **entre** dos clasificadores disponibles, se puede lograr **muestreando** las decisiones de cada uno con cierta proporción.

**Ejemplo (CoIL Challenge 2000):** 4000 clientes, presupuesto para contactar solo 800, prevalencia esperada de respondedores 6% (240 positivos, 3760 negativos). Clasificadores $A=(0.1,0.2)$ y $B=(0.25,0.6)$ en espacio ROC.

Restricción: $fp\ rate \cdot 3760 + tp\ rate \cdot 240 = 800$.

- Con $A$: $0.1\cdot3760+0.2\cdot240 = 424$ candidatos (muy pocos).
- Con $B$: $0.25\cdot3760+0.6\cdot240 = 1084$ candidatos (demasiados).

Se busca el punto $C \approx (0.18, 0.42)$ en la línea entre $A$ y $B$ que satisface la restricción. Calcular $k$ (distancia proporcional):

$$
k = \frac{0.18-0.1}{0.25-0.1} \approx 0.53
$$

Se muestrean las decisiones de $B$ con probabilidad $k$ y las de $A$ con probabilidad $1-k$ (para cada instancia, generar un número aleatorio uniforme en $[0,1]$; si supera $k$, aplicar $A$, si no, aplicar $B$).

---

## 11. Conclusión

Los gráficos ROC son una herramienta muy útil para visualizar y evaluar clasificadores, ofreciendo una medida más rica que escalares como accuracy, error rate o costo de error. Al desacoplar el desempeño del sesgo de clase y los costos de error, tienen ventajas sobre gráficos precision-recall y curvas lift. Como con cualquier métrica de evaluación, su uso correcto requiere conocer sus características y limitaciones.

---

## Referencias clave citadas

- Provost, F., Fawcett, T. (1997, 1998, 2001). Robustez de clasificadores bajo entornos imprecisos.
- Hand, D. J., Till, R. J. (2001). Generalización de AUC multiclase.
- Hanley, J. A., McNeil, B. J. (1982). AUC y test de Wilcoxon.
- Provost, F., Domingos, P. (2001). Well-trained PETs.
- Flach, P., Wu, S. (2003). Repairing concavities in ROC curves.
- Macskassy, S., Provost, F. (2004). Bandas de confianza para curvas ROC.

## Fórmulas esenciales para implementación (resumen rápido)

```
tp_rate (Recall/Sensitivity) = TP / (TP+FN)
fp_rate (Fallout)            = FP / (FP+TN)
Specificity                  = TN / (TN+FP) = 1 - fp_rate
Precision (PPV)               = TP / (TP+FP)

AUC (trapezoidal)  — ver Algoritmo 2 arriba
AUC = P(score(instancia positiva aleatoria) > score(instancia negativa aleatoria))

Iso-performance slope: m = [c(Y,n)*p(n)] / [c(N,p)*p(p)]
```
