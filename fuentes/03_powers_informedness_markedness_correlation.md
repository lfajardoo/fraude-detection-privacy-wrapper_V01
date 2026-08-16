# Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation

**Autor:** David M. W. Powers — AILab, School of Computer Science, Engineering and Mathematics, Flinders University, South Australia
**Contacto:** David.Powers@flinders.edu.au

## Resumen

Las medidas de evaluación de uso común (Recall, Precision, F-Measure, Rand Accuracy) están **sesgadas** y no deberían usarse sin comprender claramente sus sesgos e identificar el nivel de azar/caso base correspondiente. Un sistema objetivamente peor en el sentido de **Informedness** puede parecer mejor bajo cualquiera de esas medidas comunes. Se introducen y discuten **Informedness** y **Markedness** (medida dual), y se demuestran las relaciones elegantes entre Informedness, Markedness, Correlación y Significancia, junto con sus relaciones intuitivas con Recall y Precision, además de la extensión del caso dicotómico al caso multiclase general.

**Palabras clave:** Recall y Precision, F-Measure, Rand Accuracy, Kappa, Informedness y Markedness, DeltaP, Correlación, Significancia.

---

## 1. El caso binario

Tabla de contingencia $2\times2$ con notación sistemática (minúsculas = probabilidades/proporciones; MAYÚSCULAS = conteos).

| | +R | −R | |
|---|---|---|---|
| +P | tp (A) | fp (B) | pp = A+B |
| −P | fn (C) | tn (D) | pn = C+D |
| | rp = A+C | rn = B+D | N |

### Recall / Sensibilidad y Precision / Confianza

$$
\text{Recall} = \text{Sensitivity} = tpr = \frac{tp}{rp} = \frac{TP}{RP} = \frac{A}{A+C} \tag{1}
$$

$$
\text{Precision} = \text{Confidence} = tpa = \frac{tp}{pp} = \frac{TP}{PP} = \frac{A}{A+B} \tag{2}
$$

**Recall** mide la cobertura de casos reales positivos (relevante en Medicina: identificar *todos* los casos positivos reales; "True Positive Rate" en ROC). **Precision** mide cuántas de las predicciones positivas son correctas (foco de Machine Learning / Data Mining / Information Retrieval; ignorado en ROC).

### Recall e Precision inversos (Specificity)

$$
\text{Inverse Recall} = \text{Specificity} = tnr = \frac{tn}{rn} = \frac{TN}{RN} = \frac{D}{B+D} \tag{3}
$$

$$
\text{Inverse Precision} = tna = \frac{tn}{pn} = \frac{TN}{PN} = \frac{D}{C+D} \tag{4}
$$

### Rand Accuracy, Dice, Jaccard

$$
\text{Accuracy} = tca = tcr = tp+tn = rp\cdot tpr + rn\cdot tnr = \frac{TP+TN}{N} = pp\cdot tpa + pn\cdot tna = \frac{A+D}{N} \tag{5}
$$

$$
\text{Dice} = F_1 = \frac{tp}{tp+(fn+fp)/2} = \frac{A}{A+(B+C)/2} = \frac{1}{1+\text{mean}(FN,FP)/TP} \tag{6}
$$

$$
\text{Jaccard} = \frac{tp}{tp+fn+fp} = \frac{TP}{N-TN} = \frac{A}{A+B+C} = \frac{A}{N-D} = \frac{F_1}{2-F_1} \tag{7}
$$

### Fallout (False Positive Rate) y Miss Rate (False Negative Rate)

$$
\text{Fallout} = fpr = \frac{fp}{rp} \text{ (nota: en el original relativo a } rn) = \frac{FP}{RN}=\frac{B}{B+D} \tag{8}
$$

$$
\text{Miss Rate} = fnr = \frac{fn}{rp} = \frac{FN}{RP}=\frac{C}{A+C} \tag{9}
$$

> Nota: $FN$/$FP$ se asocian a errores Tipo I/II, y las tasas $fn$/$fp$ a $\alpha$/$\beta$ — aunque técnicamente esos términos se refieren a un problema meta-nivel distinto (bondad de ajuste vs. hipótesis nula).

### Prevalencia, Sesgo (Bias), Costo y Skew

- **Prevalencia** $rp = RP/N$: propiedad de la población, no controlada por el experimentador.
- **Bias** (sesgo de etiqueta) $pp = PP/N$: tendencia del modelo a predecir positivo; controlada por el experimentador.
- Regla heurística: $rp = pp$ (Prevalencia = Bias) $\Rightarrow$ Recall = Precision (= Dice, pero no Jaccard), Inverse Recall = Inverse Precision, Fallout = Miss Rate.
- **Skew:** $c_s = rn/rp$ (razón de clases). **Cost Ratio:** $c_v = cn/cp$. **Cost factor combinado:** $c = c_v c_s$ (o $c=c_s$ si insensible al costo, $c=1$ si insensible a costo y skew).

La tabla de contingencia normalizada con márgenes libres tiene **3 grados de libertad**; $N$ se necesita adicionalmente para significancia.

---

## 2. ROC y análisis PN

ROC grafica $tpr$ (eje Y) vs. $fpr$ (eje X). Clasificador perfecto: $(0,1)$. Peor caso: $(1,0)$. Clasificador aleatorio: a lo largo de la diagonal $tpr=fpr$. Diagonal negativa $tpr + c\cdot fpr = 1$: corresponde a Bias = Prevalencia ajustada por skew $c$.

**AUC** (para un único punto/modelo, trapezoide):

$$
\text{AUC} = \frac{tpr-fpr+1}{2} = \frac{tpr+tnr}{2} = 1 - \frac{fpr+fnr}{2} \tag{10}
$$

Para $c=1$ (insensible a costo/skew), maximizar AUC $\equiv$ maximizar $tpr-fpr$ $\equiv$ minimizar $fpr+fnr$.

**Fórmulas de medidas en términos de $tpr$, $fpr$, skew $c$, y de Recall/Bias/Prevalencia:**

$$
\text{Accuracy} = \frac{tpr+c(1-fpr)}{1+c} = 2\cdot\text{Recall}\cdot\text{Prev} + 1 - \text{Bias} - \text{Prev} \tag{11}
$$

$$
\text{Precision} = \frac{tpr}{tpr+c\cdot fpr} = \frac{\text{Recall}\cdot\text{Prev}}{\text{Bias}} \tag{12}
$$

$$
F_1 = \frac{2\, tpr}{tpr+c\cdot fpr+1} = \frac{2\cdot\text{Recall}\cdot\text{Prev}}{\text{Bias}+\text{Prev}} \tag{13}
$$

$$
\text{WRacc} = \frac{4c(tpr-fpr)}{(1+c)^2} = 4(\text{Recall}-\text{Bias})\cdot\text{Prev} \tag{14}
$$

**WRAcc** (Weighted Relative Accuracy, Lavrač, Flach, Zupan 1999) es insesgada: maximizar WRAcc $\equiv$ maximizar AUC $\equiv$ maximizar $tpr-fpr = 2\text{AUC}-1$. Las demás medidas (10–13) están sesgadas por el skew (excepto en su forma $c=1$).

---

## 3. DeltaP, Informedness y Markedness

**Bookmaker Informedness** (Powers): "el margen" ("edge") que tiene un apostador informado, cuantificado por sus ganancias con precios justos de un bookmaker imparcial.

> **Definición 1 (Informedness).** Cuantifica cuán informado está un predictor respecto a la condición especificada: la probabilidad de que una predicción esté informada respecto a la condición (vs. azar).

> **Definición 2 (Markedness).** Cuantifica cuán marcada está una condición para el predictor especificado: la probabilidad de que una condición esté marcada por el predictor (vs. azar).

**Fórmulas para el caso binario:**

$$
\text{Informedness} = \text{Recall} + \text{Inverse Recall} - 1 = tpr - fpr = 1 - fnr - fpr \tag{15}
$$

$$
\text{Markedness} = \text{Precision} + \text{Inverse Precision} - 1 = tpa - fna = 1 - fpa - fna
$$

En Psicología, Markedness corresponde a **DeltaP**; Informedness a **DeltaP′**. Ambas son buenos predictores de juicios asociativos humanos (Shanks; Perruchet y Peereman).

### Regresión, correlación y causalidad

Regresión lineal $y = y_0 + r_x x$, con $r_x = \frac{n\sum xy - \sum x\sum y}{n\sum x^2 - (\sum x)^2}$.

**Regresión de +R sobre +P** (predice R minimizando error en R):

$$
r_P = \frac{AD-BC}{(A+B)(C+D)} = \frac{A}{A+B} - \frac{C}{C+D} = \Delta P = \text{Markedness} \tag{17}
$$

**Regresión de +P sobre +R** (predice P minimizando error en P):

$$
r_R = \frac{AD-BC}{(A+C)(B+D)} = \frac{A}{A+C} - \frac{B}{B+D} = \Delta P' = \text{Informedness} \tag{18}
$$

**Correlación de Matthews (= Pearson en tabla de contingencia):**

$$
r_G = \frac{AD-BC}{\sqrt{(A+C)(B+D)(A+B)(C+D)}} = \text{Correlation} = \pm\sqrt{\text{Informedness}\cdot\text{Markedness}} \tag{19}
$$

El signo de la correlación coincide con el de Informedness/Markedness e indica si el uso de la información es correcto o "perverso".

Nota epistemológica: la dirección de predicción más fuerte no implica dirección de causalidad (falacia del razonamiento abductivo: $A\to B$ verdadero no implica $B\to A$).

### Formulación en términos de Evenness

Con probabilidades reducidas: numerador $= dp$ (determinante de la matriz de contingencia, común a las tres medidas), denominador depende solo de Prevalencia o Bias.

$$
M = \frac{dp}{\text{Bias}\cdot(1-\text{Bias})} = \frac{dp}{pp\cdot pn} = \frac{dp}{pg^2} = \frac{dp}{\text{BiasG}^2} = \frac{dp}{\text{EvennessP}} = \frac{\text{Precision}-\text{Prevalence}}{I_{\text{Bias}}} \tag{20}
$$

$$
B = \frac{dp}{\text{Prevalence}(1-\text{Prevalence})} = \frac{dp}{rp\cdot rn} = \frac{dp}{rg^2} = \frac{dp}{\text{PrevG}^2} = \frac{dp}{\text{EvennessR}} = \frac{\text{Recall}-\text{Bias}}{I_{\text{Prev}}}
$$
$$
= \text{Recall} - \text{Fallout} = \text{Recall}+\text{IRecall}-1 = \text{Sensitivity}+\text{Specificity}-1 \tag{21}
$$

También expresable vía razones de verosimilitud: $LR = \text{Sensitivity}/(1-\text{Specificity})$, $NLR=(1-\text{Sensitivity})/\text{Specificity}$:

$$
B = (LR-1)(1-\text{Specificity}) = (1-NLR)\cdot\text{Specificity} = \frac{(LR-1)(1-NLR)}{LR-NLR}
$$

**BookMark geométrico:**

$$
BMG = \frac{dp}{\sqrt{\text{Prev}(1-\text{Prev})\cdot\text{Bias}(1-\text{Bias})}} = \frac{dp}{\text{PrevG}\cdot\text{BiasG}} = \frac{dp}{\text{EvennessG}} \tag{22}
$$

**Definiciones de Evenness** (medias geométricas): $\text{EvennessR}=\text{PrevG}^2$ (evenness de clases reales), $\text{EvennessP}=\text{BiasG}^2$ (evenness de etiquetas predichas), $\text{EvennessG}=\text{PrevG}\cdot\text{BiasG}$ (evenness global).

$Prev\cdot Bias = etp$ (True Positives esperados relativo a $N$ bajo azar); $dp = tp - etp$ es la desviación respecto a lo esperado.

### Forma armónica (deltap, deltap')

$$
etp = rp\cdot pp, \quad etn = rn\cdot pn \tag{23}
$$

$$
dp = tp-etp = -(tn-etn), \quad \text{deltap} = dp - dtn = 2\,dp \tag{24}
$$

$$
rh = \frac{2\,rp\cdot rn}{rp+rn} = \frac{rp^2}{ra^2}, \qquad ph = \frac{2\,pp\cdot pn}{pp+pn} = \frac{pp^2}{pa^2} \tag{25}
$$

$$
B = \Delta P' = \frac{2\,dp}{rh} = \frac{\text{deltap}}{rh} \tag{26}, \qquad M = \Delta P = \frac{2\,dp}{ph} = \frac{\text{deltap}}{ph} \tag{27}
$$

### Recall, Precision, Bias, Prevalencia

$$
\text{Recall} = \text{Bookmaker}\cdot(1-\text{Prevalence}) + \text{Bias} \tag{28a}
$$
$$
\text{Bookmaker} = \frac{\text{Recall}-\text{Bias}}{1-\text{Prevalence}} \tag{28b}
$$
$$
\text{Precision} = \text{Markedness}\cdot(1-\text{Bias}) + \text{Prevalence} \tag{29a}
$$
$$
\text{Markedness} = \frac{\text{Precision}-\text{Prevalence}}{1-\text{Bias}} \tag{29b}
$$

Si Bias = Prevalencia: Recall = Precision = $F_1$ y Bookmaker = Markedness = Correlation.

**Interpretación clave:** Recall refleja el Bias más una estimación descontada de Informedness; Precision refleja la Prevalencia más una estimación descontada de Markedness.

### Relación con Kappa

Kappa de Cohen: renormalización de Accuracy restando la Accuracy esperada (producto punto de Bias y Prevalencia). $\text{Kappa} = \frac{dtp}{dtp+\text{mean}(fp,fn)}$. Propiedades invariantes bajo el problema Inverso (intercambiar +/− en condición y predicción): Informedness, Markedness y Kappa son invariantes; en el problema **Dual** (intercambiar antecedente/consecuente), se intercambian Precision↔Recall, Prevalencia↔Bias, Markedness↔Informedness.

Kappa, aunque más significativo que Recall/Precision/Accuracy, es no lineal y no da cuenta bien del error → se prefiere Correlation como medida estándar de acuerdo.

---

## 4. Significancia e información

**Chi-cuadrado para el caso positivo:**

$$
\chi^2_{+P} = \frac{(TP-ETP)^2}{ETP} + \frac{(FP-EFP)^2}{EFP} = \frac{2N\cdot dp^2}{ehp} = N\cdot B^2\cdot\text{EvennessR}/\text{Bias} \tag{30}
$$

**$G^2$ (Ganancia de información total, relacionado con Información Mutua):**

$$
G^2_{+P}/2 \approx N\cdot dp^2 / \text{PrevG}^2 / \text{Bias} = N\cdot B^2\cdot\text{EvennessR}/\text{Bias} \tag{31}
$$

$\chi^2$ es poco fiable para $N$ y celdas pequeñas; se prefiere $G^2$. Corrección de Yates: restar 0.5 al valor absoluto de $dp$ antes de elevar al cuadrado (celdas $<5$).

**Promedio ponderado (independiente de qué variable se elige):**

$$
\chi^2_{KB} = 2N\cdot\frac{dtp^2}{\text{PrevG}^2} = 2N\cdot B^2\cdot\text{EvennessR} \tag{32}
$$

$$
\chi^2_{KM} = 2N\cdot\frac{dtp^2}{\text{BiasG}^2} = 2N\cdot M^2\cdot\text{EvennessP} \tag{33}
$$

$$
\chi^2_{KBM} = 2N\cdot\frac{dtp^2}{\text{PrevG}\cdot\text{BiasG}} = 2N\cdot B\cdot M\cdot\text{EvennessG} \tag{34}
$$

**Test de independencia sobre la tabla completa** (cancela el factor Evenness, menos conservador):

$$
\chi^2_{BM} = N\cdot r_G^2 = N\rho^2 = N\phi^2 = N\cdot B\cdot M \tag{35}
$$

(Equivale el coeficiente de correlación Phi $\phi$ con Pearson $\rho$; conecta $N\cdot MI(\mathbf{R}\|\mathbf{P})$ con $\chi^2$ vía $G^2$.)

---

## 5. Intervalos de confianza y desviaciones

Bookmaker Informedness $B$, Markedness $M$, Correlation $C$ (media geométrica). Fórmulas de intervalo de confianza (relacionadas con $\gamma=0.05$, multiplicador $X$, típicamente $X=1.96$):

$$
CI_{B2} = X\cdot\frac{1-|B|}{\sqrt{2E(N-1)}} \tag{63}
$$
$$
CI_{M2} = X\cdot\frac{1-|B|}{\sqrt{2E(N-1)}} \tag{64}
$$
$$
CI_{C2} = X\cdot\frac{1-|B|}{\sqrt{2E(N-1)}} \tag{65}
$$

Variante que da cuenta del error de discretización ($N<8K$):

$$
CI_{B1} = X\cdot\frac{1-2|B|+2B^2}{\sqrt{2E(N-1)}} \tag{66}
$$
$$
CI_{M1} = X\cdot\frac{1-2|B|+2B^2}{\sqrt{2E(N-1)}} \tag{67}
$$
$$
CI_{C1} = X\cdot\frac{1-2|B|+2B^2}{\sqrt{2E(N-1)}} \tag{68}
$$

Multiplicador $X$: 1.96 para pruebas de dos colas al 5%; 1.65 para pruebas de una cola.

---

## 6. Ejemplos simples y consideraciones prácticas

**Recomendaciones de uso:**
- Comparación entre dos evaluadores/sistemas sin preferencia a priori → **Correlation** (preferible a Kappa).
- Existe un gold standard confiable → normalización por Prevalencia/Evenness del gold standard → **Informedness**.
- Comparar qué problema resuelve mejor un sistema propuesto entre distintas condiciones → **Markedness**.
- Recall/Informedness: pruebas de efectividad relativa a un conjunto de *condiciones* (ej. Word Alignment en Machine Translation).
- Precision/Markedness: efectividad relativa a un conjunto de *predicciones* (ej. Information Retrieval, sin gold standard completo).

**Ejemplo numérico (Tabla 2, N=100):** En dos tablas de contingencia se muestra que Recall, Precision, Accuracy, F1, G-mean y Kappa **suben** de una tabla a otra, mientras **Bookmaker Informedness cae** — ilustrando el problema central del artículo.

---

## 7. Generalización al caso multiclase (K clases)

**Mutual Information y Entropía Condicional:**

$$
MI(\mathbf{R}\|\mathbf{P}) = \sum_l P_{\mathbf P}(l)\sum_c P_{\mathbf R}(c|l)\left[-\log\frac{P_{\mathbf R}(c|l)}{P_{\mathbf R}(c)}\right] \tag{39}
$$

$$
H(\mathbf{R}|\mathbf{P}) = \sum_l P_{\mathbf P}(l) \sum_c P_{\mathbf R}(c|l)\, [-\log P_{\mathbf R}(c|l)] \tag{40}
$$

**Bookmaker Informedness generalizado** (promedio puntual sobre las celdas):

$$
B(\mathbf{R}|\mathbf{P}) = \sum_l P_{\mathbf P}(l)\sum_c P_{\mathbf R}(c|l)\left[\frac{P_{\mathbf P}(l)}{P_{\mathbf R}(l) - \partial_{|c-l|}}\right] \tag{41}
$$

Definiendo una dicotomía binaria para cada etiqueta $l$ (caso $l$ = Positivo, resto = Negativo), con Prevalencia $\text{Prev}(l)$ e Informedness dicotómica $B(l)$:

$$
B(\mathbf{R}|\mathbf{P}) = \sum_l \text{Prev}(l)\, B(l) \tag{42}
$$

$$
M(\mathbf{P}|\mathbf{R}) = \sum_c \text{Bias}(c)\, M(c) \tag{43}
$$

Correlación multiclase = Media Geométrica de Informedness y Markedness multiclase; su cuadrado da el Coeficiente de Determinación.

### Vía el determinante de la matriz de contingencia

$$
M \approx \left[\frac{\det}{\text{BiasG}^K}\right]^{2/K} = \frac{\det^{2/K}}{\text{EvennessP}+} \tag{44}
$$

$$
B \approx \left[\frac{\det}{\text{PrevG}^K}\right]^{2/K} = \frac{\det^{2/K}}{\text{EvennessR}+} \tag{45}
$$

$$
BMG \approx \frac{\det^{2/K}}{\text{PrevG}\cdot\text{BiasG}} = \frac{\det^{2/K}}{\text{EvennessG}+} \tag{46}
$$

(Empíricamente, esta generalización ajusta bien cerca de $B=0$ o $B=1$, pero no tan bien en valores intermedios — sugiere un exponente heurístico mal calibrado al pasar de $K$ dimensiones a 2; se explora un exponente alternativo $1/(3K-2)$.)

### Generalización de significancia

$$
\chi^2_{KB} = KN\cdot B^2\cdot\text{EvennessR}^- \tag{47}
$$
$$
\chi^2_{KM} = KN\cdot M^2\cdot\text{EvennessP}^- \tag{48}
$$
$$
\chi^2_{KBM} = KN\cdot B\cdot M\cdot\text{EvennessG}^- \tag{49}
$$

Grados de libertad: $r=K-1$ apropiado para $\beta$ (bajo asociación casi completa); $r=(K-1)^2$ apropiado para $\alpha$ (hipótesis nula).

$$
\chi^2_{XB} = K(K-1)\cdot N\cdot B^2\cdot\text{EvennessR}^- \tag{50}
$$
$$
\chi^2_{XBM} = K(K-1)\cdot N\cdot B\cdot M\cdot\text{EvennessG}^- \tag{52}
$$

Forma "naïve" (suma no ponderada sobre toda la tabla; corresponde a Cramer's V):

$$
\chi^2_B = (K-1)\cdot N\cdot B^2 \tag{53}
$$
$$
\chi^2_M = (K-1)\cdot N\cdot M^2 \tag{54}
$$
$$
\chi^2_{BM} = (K-1)\cdot N\cdot B\cdot M \tag{55}
$$

**Cramer's V:** $V = [\chi^2/(N(K-1))]^{1/2}$ — tiende a **sobreestimar** la asociación real medida por Bookmaker/Markedness, especialmente para asociación alta.

### Generalización de Evenness

$$
\chi^2_{KB} = KN\cdot \det^{2/K} / \text{EvennessR}^\# \tag{58}
$$
$$
\chi^2_{KM} = KN\cdot \det^{2/K} / \text{EvennessP}^\# \tag{59}
$$
$$
\chi^2_{KBM} = KN\cdot \det^{2/K} / \text{EvennessG}^\# \tag{60}
$$

con $\text{EvennessR}^- = \text{EvennessR}^+/\text{EvennessR}^\#$ (relación entre las tres formas de Evenness: "+" = media geométrica al cuadrado, "−" ≈ media aritmética, "#" = media armónica).

---

## 8. Conclusiones y trabajo futuro

- **Informedness dicotómica** = Recall + Inverse Recall − 1 (= Sensitivity + Specificity − 1).
- **Markedness dicotómica** = Precision + Inverse Precision − 1.
- Su media geométrica = Correlación de Matthews.
- Evenness = cuadrado de la media geométrica de Prevalencia e Inverse Prevalencia (y/o Bias e Inverse Bias).
- $\chi^2$ es simplemente multiplicación por una constante; los intervalos de confianza conservadores son una raíz cuadrada.
- **Trabajo futuro:** desarrollo de algoritmos de aprendizaje que optimicen directamente una medida corregida por azar (Bookmaker Informedness) en vez de medidas sesgadas; esto ya ha mostrado mejoras sustanciales en contextos de boosting.

---

## Fórmulas esenciales para implementación (resumen rápido)

```
Recall (tpr)        = TP / (TP+FN)
Precision            = TP / (TP+FP)
Specificity (tnr)    = TN / (TN+FP)
Inverse Precision    = TN / (TN+FN)

Informedness (B)     = Recall + Specificity - 1   = tpr - fpr
Markedness (M)       = Precision + InvPrecision - 1
Correlation (MCC)    = sign(B) * sqrt(|B * M|)   (para B,M de igual signo)
                     = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

AUC                  = (tpr + tnr) / 2   [para un solo punto operativo]
```

Multiclase: promediar Informedness/Markedness dicotómicas ponderando por Prevalencia/Bias de cada clase (Ecs. 42–43).

## Referencias clave citadas

- Flach, P. (2003). ICML 2003, "The geometry of ROC space".
- Fürnkranz, J., Flach, P. (2005). *Machine Learning* 58(1):39–77.
- Hand, D. J., Till, R. J. (2001). Generalización de AUC a multiclase.
- Cohen, J. (1960, 1968). Kappa.
- Shanks, D. R. (1995); Perruchet, P., Peereman, R. (2004). DeltaP en Psicología.
