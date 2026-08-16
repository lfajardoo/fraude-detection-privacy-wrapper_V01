# On Nonparametric Ordinal Classification with Monotonicity Constraints

**Autores:** Wojciech Kotłowski, Roman Słowiński
**Publicación:** IEEE Transactions on Knowledge and Data Engineering, Vol. 25, No. 11, Noviembre 2013 (pp. 2576–2589)
**DOI:** 10.1109/TKDE.2012.204

## Resumen

Se aborda el problema de clasificación ordinal con restricciones de monotonía: la clase (variable de salida) no debe decrecer cuando los valores de los atributos (variables de entrada) aumentan. El enfoque más general para tratar este problema es el **no paramétrico**, donde la única suposición es la monotonía. Se analizan dos métodos:

- **Plug-in**: se estima primero la distribución condicional de clase.
- **Directo**: se minimiza el riesgo empírico.

Se demuestra que ambos métodos están estrechamente relacionados (coinciden para funciones de pérdida lineales), se analizan sus propiedades estadísticas y computacionales, y se valida todo mediante experimentos.

**Palabras clave:** aprendizaje automático, restricciones de monotonía, clasificación ordinal, regresión ordinal, aprendizaje de preferencias, métodos no paramétricos, regresión isotónica, clasificación isotónica, funciones monótonas.

---

## 1. Introducción

El **principio de dominancia**: cuanto mejor la evaluación de un objeto en los atributos considerados (variables de entrada), mejor su asignación de clase (variable de salida). Ejemplos:

- "cuanto mayor el ratio de deuda de una empresa, mayor su riesgo de bancarrota"
- "cuanto mejor la educación/experiencia de un candidato, más alta su posición en el proceso de selección"

Este principio aparece en: análisis de satisfacción del cliente, valoración de viviendas, riesgo de bancarrota, valoración de opciones, diagnóstico médico, aprobación de crédito, datos de encuestas, etc.

Enfoques previos: DRSA (Dominance-based Rough Set Approach), inducción de reglas de decisión, ensembles de reglas, árboles de clasificación monótonos, redes monótonas, métodos basados en instancias, separación isotónica.

El enfoque **no paramétrico** trabaja eficientemente con la clase de *todas* las funciones monótonas (sin otra restricción de forma), y puede usarse también como preprocesamiento para "monotonizar" (corregir) un conjunto de datos antes de aplicar cualquier otro algoritmo.

### Contribución principal

1. Marco estadístico formal para el aprendizaje con restricciones de monotonía; condiciones necesarias y suficientes sobre la función de pérdida para que el clasificador de Bayes óptimo sea monótono.
2. Análisis de dos métodos no paramétricos: *plug-in* (regresión isotónica múltiple) y *directo* (clasificación isotónica, vía programación lineal). Se muestra su relación, consistencia asintótica y validación experimental.

---

## 2. Planteamiento del problema

Sea $(x,y) \in \mathcal{X} \times \mathcal{Y}$ un par objeto-etiqueta generado según una distribución desconocida $P(x,y)$, donde $y$ es una etiqueta de clase de un conjunto finito ordenado $\mathcal{Y} = \{1,\dots,K\}$, $x$ es la descripción del objeto y $\mathcal{X}$ es el espacio de entrada.

El objetivo: hallar un clasificador $h: \mathcal{X} \to \mathcal{Y}$ que prediga $y$ con precisión, medida mediante una función de pérdida $L(y,\hat y)$.

**Riesgo esperado:**
$$
L(h) = \mathbb{E}[L(y, h(x))] \tag{1}
$$

El **clasificador de Bayes** minimiza el riesgo esperado: $h^* = \arg\min_h L(h)$, y el riesgo mínimo $L^* = L(h^*)$ se llama **riesgo de Bayes**. Se cumple:

$$
h^*(x) = \arg\min_{k \in \mathcal{Y}} \sum_{y \in \mathcal{Y}} L(y,k)\, P(y|x) \tag{2}
$$

Dado que $P(x,y)$ es desconocida, se aproxima $h^*$ usando una muestra i.i.d. $D = \{(x_1,y_1),\dots,(x_n,y_n)\}$.

### Dos enfoques

**Método plug-in:** se estiman $K$ funciones $\hat p_1(x),\dots,\hat p_K(x)$ (estimadores de $P(y=k|x)$), y luego:

$$
\hat h(x) = \arg\min_{k \in \mathcal{Y}} \sum_{y \in \mathcal{Y}} L(y,k)\, \hat p_y(x) \tag{3}
$$

**Método directo:** minimización del riesgo empírico

$$
L_D(h) = \frac{1}{n}\sum_{i=1}^n L(y_i, h(x_i))
$$

dentro de una familia de funciones $\mathcal{H}$:

$$
\hat h = \arg\min_{h \in \mathcal{H}} L_D(h) \tag{4}
$$

### Supuestos sobre la pérdida

Se asume $L(k,k)=0$ y $L(y,k)>0$ si $y \ne k$. La pérdida debe ser consistente con el orden de las clases: forma **V** — para $k \le y$: $L(y,k-1) \ge L(y,k)$; para $k \ge y$: $L(y,k) \le L(y,k+1)$.

### Relación de dominancia

Se asume $\mathcal{X} \subseteq \mathbb{R}^m$, $x = (x_1,\dots,x_m)$. Se dice que $x$ domina a $x'$, $x \succeq x'$, si $x_s \ge x'_s$ para todo $s=1,\dots,m$. Esta relación es un orden parcial. Las restricciones de monotonía exigen que $x \succeq x' \Rightarrow$ etiqueta de $x \ge$ etiqueta de $x'$.

Función $h: \mathcal{X} \to \mathcal{Y}$ es **monótona** si $\forall x,x'$: $x \succeq x' \Rightarrow h(x) \ge h(x')$. Análogamente para vectores $v \in \mathcal{Y}^n$ definidos sobre el conjunto de entrenamiento.

---

## 3. Restricciones de monotonía

### 3.1 Dominancia estocástica

Para $x \succeq x'$ y cada $k \in \{1,\dots,K\}$:

$$
P(y \ge k \mid x) \ge P(y \ge k \mid x') \tag{5}
$$

Esto es la relación de **dominancia estocástica (de primer orden)**:

$$
x \succeq x' \implies P(y|x) \succeq P(y|x') \tag{6}
$$

Una distribución que satisface (6) se llama **monotónicamente restringida**.

### 3.2 Clasificador de Bayes monótono

**Contraejemplo (3 clases):** $x \succeq x'$, $P(y|x) = (0.1, 0.5, 0.4)$, $P(y|x') = (0.3, 0.3, 0.4)$. Aunque $P(y|x) \succeq P(y|x')$, la moda de $P(y|x')$ es 3 y la de $P(y|x)$ es 2 → el clasificador de Bayes bajo pérdida 0-1 **no** es monótono en general.

**Teorema 1.** Sea $L(y,k)$ una pérdida en forma V. El clasificador de Bayes es monótono para toda distribución monotónicamente restringida si y solo si, para todo $y,k \in \{1,\dots,K-1\}$:

$$
L(y,k+1) - L(y,k) \le L(y+1,k+1) - L(y+1,k) \tag{7}
$$

Una pérdida que satisface (7) se llama **pérdida monótona**.

### 3.3 Clasificador de Bayes monótono y pérdida convexa

Para pérdidas de la forma $L(y,k) = c(y-k)$, con $c(0)=0$, $c(k)>0$ para $k\ne 0$ (ej.: pérdida 0-1, error absoluto $c(k)=|k|$, error cuadrático $c(k)=k^2$).

Sea $\mathcal{Y}' = \{-(K-1),\dots,-1,0,1,\dots,K-1\}$. Una función $c: \mathcal{Y}' \to \mathbb{R}$ es **convexa** si para todo $k$ con $-(K-1) < k < K-1$:

$$
c(k) \le \frac{c(k-1) + c(k+1)}{2} \tag{8}
$$

**Teorema 2.** Sea $L(y,k)=c(y-k)$ pérdida en forma V. El clasificador de Bayes $h^*(x)$ es monótono si y solo si $c(k)$ es convexa.

**Corolario 1.** Sea $L(y,k) = |y-k|^p$, $p \ge 0$, $K \ge 3$. El clasificador de Bayes es monótono si y solo si $p \ge 1$.

Esto implica que la pérdida 0-1 ($p \to 0$) **no** da un clasificador de Bayes monótono para $K\ge 3$, mientras que el error absoluto ($p=1$) y el error cuadrático ($p=2$) sí garantizan monotonía.

### 3.4 Función de pérdida lineal

$$
L(y,k) = \begin{cases} \tau (k-y) & \text{si } k > y \\ (1-\tau)(y-k) & \text{si } k \le y \end{cases}, \quad 0 < \tau < 1 \tag{9}
$$

Para $\tau = 1/2$: pérdida de error absoluto $L=|k-y|$ (hasta constante). Esta pérdida modela asimetría en el costo de clasificación errónea (por ejemplo, en un contexto médico, predecir un estado de salud mejor de lo real puede ser más peligroso que lo contrario).

Es minimizada por el **cuantil $(1-\tau)$** de la distribución condicional: $y_{1-\tau}$ tal que $P(y \le y_{1-\tau}) \ge 1-\tau$ y $P(y \ge y_{1-\tau}) \ge \tau$. Para $\tau=1/2$, se obtiene la **mediana**.

La minimización de la pérdida lineal es **invariante a la codificación** de las etiquetas (invariante bajo transformaciones monótonas del dominio).

---

## 4. El enfoque plug-in

Basado en **regresión isotónica**, propuesto originalmente por Dembczyński et al. y, de forma independiente, por Feelders.

**Corolario 2.** Sea $\hat h$ un clasificador plug-in (3) y $L(y,k)$ monótona. Entonces $\hat h$ es monótono si, para cada $k=2,\dots,K$:

$$
\hat P_k(x) := \sum_{k'=k}^{K} \hat p_{k'}(x) \tag{10}
$$

es monótona.

### 4.1 Problema binario y regresión isotónica

Caso $K=2$, $\mathcal{Y}=\{0,1\}$. Sea $\hat p_i = \hat p_1(x_i)$. Se define $\hat{\mathbf p} = (\hat p_1,\dots,\hat p_n)$ como la **regresión isotónica** de $\mathbf y = (y_1,\dots,y_n)$ si resuelve:

$$
\min \sum_{i=1}^n (y_i - p_i)^2 \quad \text{sujeto a: } x_i \succeq x_j \Rightarrow p_i \ge p_j,\ \forall i,j \tag{11}
$$

Es un problema de optimización cuadrática con restricciones lineales, resoluble eficientemente. Algoritmo heurístico de Burdakov et al.: $O(n^2)$. Algoritmo exacto: $O(n^4)$.

**Conjuntos inferior/superior:** $L \subseteq \{x_1,\dots,x_n\}$ es un *conjunto inferior* si $x_i \in L$ implica que $x_j \preceq x_i \Rightarrow x_j \in L$; análogo para conjunto *superior* $U$. Para un vector $f$, $\text{Av}(f,A) = \frac{1}{|A|}\sum_{i\in A} f_i$.

**Teorema 3.** Sea $\hat{\mathbf p}$ la regresión isotónica de $\mathbf y$. Entonces:

$$
\hat p_i = \min_{L: x_i \in L} \max_{U: x_i \in U} \text{Av}(\mathbf y, L \cap U)
$$

**Objeto consistente:** $x_i$ es *consistente* si para todo $j$: $x_i \succeq x_j \Rightarrow y_i \ge y_j$ y $x_i \preceq x_j \Rightarrow y_i \le y_j$.

**Teorema 4.** $\hat p_i = y_i$ si y solo si $x_i$ es consistente.

Esto permite **reducir el tamaño del problema**: solo hay que optimizar sobre objetos inconsistentes.

### 4.2 Problema multiclase

Se descompone en $K-1$ subproblemas binarios. Para $k=2,\dots,K$, valores dummy $y_{ik} = \mathbb{1}[y_i \ge k]$. Se define $\hat{\mathbf q}_k = (\hat q_{1k},\dots,\hat q_{nk})$ como la regresión isotónica de $\mathbf y_k = (y_{1k},\dots,y_{nk})$:

$$
\min \sum_{i=1}^n (y_{ik}-p_i)^2 \quad \text{sujeto a: } x_i \succeq x_j \Rightarrow p_i \ge p_j \tag{12}
$$

Estimadores de $P(y=k|x_i)$:

$$
\hat p_{ik} = \begin{cases}
\hat q_{ik} & k = K \\
\hat q_{ik} - \hat q_{i,k+1} & 2 \le k < K \\
1 - \hat q_{i,2} & k = 1
\end{cases} \tag{13}
$$

**Teorema 5.** Para cada $i$, $\{\hat p_{i1},\dots,\hat p_{iK}\}$ forma una distribución de probabilidad válida (no negativa, suma 1).

### 4.3 Extensión más allá del conjunto de entrenamiento

Extensiones mínima y máxima (Potharst y Feelders):

$$
\hat p_{\min}(x) = \max\{\hat p_i : x_i \preceq x\}, \qquad \hat p_{\max}(x) = \min\{\hat p_i : x_i \succeq x\}
$$

Toda extensión válida satisface $\hat p_{\min}(x) \le \hat p(x) \le \hat p_{\max}(x)$. Extensión parametrizada:

$$
p_\lambda(x) = \lambda\, \hat p_{\max}(x) + (1-\lambda)\, \hat p_{\min}(x), \quad \lambda \in [0,1] \tag{14}
$$

---

## 5. El método directo

Minimización del riesgo empírico dentro de la clase de todas las funciones monótonas:

$$
\min \sum_{i=1}^n L(y_i,d_i) \quad \text{sujeto a: } x_i \succeq x_j \Rightarrow d_i \ge d_j;\quad d_i \in \{1,\dots,K\} \tag{15}
$$

Interpretación: relabelar objetos para hacer el dataset monótono, minimizando el cambio según la pérdida (**corrección de error no paramétrica**). Se llama **clasificación isotónica**, y su solución óptima $\hat{\mathbf d}$ es la clasificación isotónica de $\mathbf y$.

Resoluble por programación lineal o por flujo máximo en red, $O(n^3)$ (Chandrasekaran et al.).

Variables binarias $d_{ik}$ ("$d_{ik}=1$ ssi la nueva etiqueta de $x_i$ es al menos $k$"), con $d_{ik} \ge d_{i,k+1}$. Nueva etiqueta: $d_i = 1 + \sum_{k=2}^K d_{ik}$.

Con $\delta(y,k) = L(y,k)-L(y,k-1)$:

$$
\min \sum_{i=1}^n \sum_{k=2}^K \delta(y_i,k)\, d_{ik} \quad \text{sujeto a: } x_i \succeq x_j \Rightarrow d_{ik} \ge d_{jk};\ d_{ik} \ge d_{i,k+1};\ d_{ik}\in\{0,1\} \tag{17}
$$

La relajación a $0 \le d_{ik} \le 1$ es exacta gracias a la **unimodularidad total** de la matriz de restricciones.

### 5.1 Reducción del tamaño del problema

Etiquetas límite inferior/superior:

$$
l_i = \min\{y_j : x_j \succeq x_i\}, \qquad u_i = \max\{y_j : x_j \preceq x_i\} \tag{18}
$$

**Lema 1.** (i) $l_i \le y_i \le u_i$; (ii) $u_i=l_i$ ssi $x_i$ es consistente; (iii) $x_i \succeq x_j \Rightarrow l_i \ge l_j$, $u_i \ge u_j$.

**Teorema 6.** Toda clasificación isotónica $\hat{\mathbf d}$ satisface $l_i \le \hat d_i \le u_i$. Por tanto, los objetos consistentes pueden removerse de la optimización.

### 5.2 Clasificación isotónica binaria

$K=2$: $\tau = \frac{L(0,1)}{L(0,1)+L(1,0)}$. El clasificador de Bayes toma una de estas formas:

$$
h^*(x) = \mathbb{1}[P(y=1|x) \ge \tau] \quad \text{o} \quad h^*(x) = \mathbb{1}[P(y=1|x) > \tau] \tag{19}
$$

Con $w_0=\tau$, $w_1=1-\tau$ (20):

$$
\min \sum_{i=1}^n w_{y_i} |y_i - d_i| \quad \text{sujeto a: } x_i \succeq x_j \Rightarrow d_i \ge d_j \tag{21}
$$

**Nivel de conjunto:** $[\hat{\mathbf p}=a] = \{i : \hat p_i = a\}$.

**Teorema 7.** Si $\hat{\mathbf p}$ es la regresión isotónica de $\mathbf y$ y $a$ es tal que $[\hat{\mathbf p}=a] \ne \emptyset$, entonces $a = \text{Av}(\mathbf y, [\hat{\mathbf p}=a])$.

**Teorema 8.** Sea $\hat{\mathbf p}$ la regresión isotónica de $\mathbf y$. Los vectores $\hat d^-_i = \mathbb{1}[\hat p_i > \tau]$ y $\hat d^+_i = \mathbb{1}[\hat p_i \ge \tau]$ son clasificaciones isotónicas de $\mathbf y$. Toda clasificación isotónica $\hat d$ satisface $\hat d^-_i \le \hat d_i \le \hat d^+_i$. Si $\hat d^- = \hat d^+$, la clasificación isotónica es única.

**Conclusión clave:** el método plug-in y el método directo **coinciden** para clasificación binaria.

**Teorema 9.** Si $\tau \ne \frac{r}{r+s}$ para enteros $r,s \le n$, la $\tau$-clasificación isotónica binaria es única. En caso contrario, se obtiene perturbando $\tau$ en $\pm \epsilon \ll n^{-2}$.

### 5.3 Clasificación isotónica lineal

**Teorema 10.** Sean $\hat{\mathbf q}_k$ regresiones isotónicas de $\mathbf y_k$, $k=2,\dots,K$. Los vectores $\hat d^+_i = 1+\sum_{k=2}^K \mathbb{1}[\hat q_{ik} > \tau]$ y $\hat d^-_i = 1+\sum_{k=2}^K \mathbb{1}[\hat q_{ik} \ge \tau]$ son clasificaciones isotónicas lineales de $\mathbf y$, y toda otra solución $\hat d_i$ satisface $\hat d^-_i \le \hat d_i \le \hat d^+_i$.

De aquí se deduce que el problema con $(K-1) \cdot n$ variables se descompone en $K-1$ subproblemas de $n$ variables cada uno — gran ventaja computacional.

Cada $\hat d_i$ intermedio es el cuantil $(1-\tau)$ de la distribución $\{\hat p_{i1},\dots,\hat p_{iK}\}$ obtenida en (13). Como el clasificador de Bayes para pérdida lineal es también el cuantil $(1-\tau)$: **plug-in y método directo coinciden para pérdida lineal**.

### 5.4 Extensión más allá del conjunto de entrenamiento

$$
\hat h_{\min}(x) = \max\{\hat d_i : x_i \preceq x\}, \qquad \hat h_{\max}(x) = \min\{\hat d_i : x_i \succeq x\} \tag{22}
$$

**Teorema 11.** Toda extensión monótona válida $\hat h$ satisface $\hat h_{\min}(x) \le \hat h(x) \le \hat h_{\max}(x)$.

---

## 6. Consistencia asintótica

Se dice que $\hat h_n$ es **fuertemente consistente** si $\lim_{n\to\infty} L(\hat h_n) = L^*$ con probabilidad 1.

**Ejemplo de inconsistencia:** $\mathcal{X}=[0,1]^2$, distribución concentrada en la diagonal $x_1 = 1-x_2$. Con probabilidad 1 ningún objeto domina a otro → los métodos no paramétricos no son consistentes en general.

Sin embargo, basta con que $P(x)$ tenga densidad respecto a la medida de Lebesgue para garantizar consistencia.

**Teorema 12.** Si $P(x,y)$ está monotónicamente restringida y $P(x)$ tiene densidad en $\mathcal{X}=\mathbb{R}^m$, cualquier extensión válida de la clasificación isotónica lineal $\hat h_n$ es fuertemente consistente.

**Lema 2.** Bajo las mismas condiciones, para $K=2$: $\lim_{n\to\infty}\mathbb{E}[(\hat p_n(x)-\eta(x))^2 \mid D_n] = 0$ c.s., donde $\eta(x)=P(y=1|x)$.

**Teorema 13.** El clasificador plug-in basado en regresión isotónica múltiple es fuertemente consistente para **cualquier** función de pérdida (no requiere pérdida lineal, a diferencia del Teorema 12).

Dado que la clasificación isotónica es computacionalmente más simple, se recomienda **preferir el método directo (isotónico) en aplicaciones**.

---

## 7. Resultados experimentales

### 7.1 Datos artificiales

Objetos $x \in [0,1]^m$ generados uniformemente. Función objetivo "Bayes":

$$
f(x) = \sum_{t=1}^T a_t\, r_t(x), \qquad r_t(x) = \prod_{s=1}^{m_s} \mathbb{1}[x_{j_s} \ge b_s] \ \text{ó} \ \prod_{s=1}^{m_s} \mathbb{1}[x_{j_s} \le b_s] \tag{24}
$$

Parámetros: $K=5$, $n=1000$, $m \in \{4,6,8,10\}$, riesgo de Bayes $R \in \{0.1,0.2,0.3,0.4\}$ (medido con pérdida de error absoluto). Se comparan C4.5, AdaBoost+C4.5, regresión logística, RankBoost, con y sin preprocesamiento por clasificación isotónica.

**Resultado principal:** eliminar inconsistencias (monotonización) **nunca empeora** la precisión, y la mejora es mayor cuando el riesgo de Bayes (ruido) es alto y cuando $m$ es bajo (la relación de dominancia se vuelve dispersa en alta dimensión).

Tiempos de cómputo (con solver `lp_solve`): la descomposición propuesta (Teoremas 6 y 10) acelera el cómputo en varios órdenes de magnitud.

### 7.2 Datos reales

8 datasets con monotonía conocida: ESL, SWD, LEV (encuestas), Housing, Breast Cancer Wisconsin, Breast Cancer Ljubljana, Car, CPU (UCI). Comparación con C4.5, Naive Bayes, SVM, usando MAE con validación cruzada 10-fold × 10 repeticiones. La clasificación isotónica, como clasificador *standalone*, **supera** a los algoritmos estándar en la mayoría de los casos.

---

## 8. Conclusiones

- Se presentó una teoría estadística para clasificación ordinal con restricciones de monotonía, basada en dominancia estocástica.
- Las pérdidas **convexas** son las más adecuadas para este tipo de clasificación.
- El método plug-in (regresión isotónica múltiple) y el método directo (programación lineal) coinciden para pérdida lineal.
- Ambos métodos son consistentes asintóticamente bajo condiciones leves sobre $P(x)$.
- Se recomienda el método directo (isotónico) por su menor coste computacional.

---

## Referencias clave citadas en el texto

- Robertson, Wright, Dykstra — *Order Restricted Statistical Inference*, Wiley, 1998. (Teoría de regresión isotónica; Teoremas 3 y 7 provienen de esta referencia.)
- Chandrasekaran, Ryu, Jacob, Hong — "Isotonic Separation", *INFORMS J. Computing*, 2005.
- Devroye, Györfi, Lugosi — *A Probabilistic Theory of Pattern Recognition*, Springer, 1996.
- Potharst, Feelders — "Classification Trees for Problems with Monotonicity Constraints", *SIGKDD Explorations*, 2002.
