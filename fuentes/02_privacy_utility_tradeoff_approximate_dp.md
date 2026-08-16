# Tight Analysis of Privacy and Utility Tradeoff in Approximate Differential Privacy

**Autores:** Quan Geng, Wei Ding, Ruiqi Guo, Sanjiv Kumar (Google Research)
**Publicación:** Proceedings of AISTATS 2020, PMLR Vol. 108

## Resumen

Se caracteriza la mínima amplitud y potencia de ruido para mecanismos de adición de ruido bajo privacidad diferencial $(\epsilon,\delta)$ para una función de consulta real. Se derivan nuevas cotas inferiores mediante dualidad de programación lineal, y nuevas cotas superiores analizando una nueva clase de mecanismos: los **mecanismos Laplacianos truncados**. Se muestra que la brecha multiplicativa entre las cotas inferior y superior tiende a cero en varios regímenes de alta privacidad, probando su optimalidad (cerrando la brecha constante previa en el caso discreto). Experimentos numéricos muestran mejoras del mecanismo Laplaciano truncado frente al mecanismo Gaussiano óptimo en todos los regímenes de privacidad.

---

## 1. Introducción

La privacidad diferencial (Dwork et al., 2006b) exige la casi-indistinguibilidad de si un individuo está o no en un dataset. La privacidad diferencial clásica, **$\epsilon$-DP**, impone una cota multiplicativa $e^\epsilon$ sobre la razón de probabilidades de salidas para datasets vecinos. El enfoque estándar: ruido Laplaciano.

La **privacidad diferencial aproximada** $(\epsilon,\delta)$-DP (Dwork et al., 2006a) se interpreta como "$\epsilon$-DP excepto con probabilidad $\delta$". El enfoque estándar: mecanismo Gaussiano.

### Contribuciones

1. Análisis de una nueva clase de mecanismos $(\epsilon,\delta)$-DP: **mecanismos Laplacianos truncados**, derivando nuevas cotas superiores alcanzables.
2. Nuevas cotas inferiores vía discretización + dualidad de programación lineal, extendiendo el resultado de Geng y Viswanath (2016a) (caso entero) al caso continuo.
3. Se prueba que la brecha multiplicativa entre cotas superior e inferior tiende a 0 en regímenes de alta privacidad, estableciendo la optimalidad (casi) del mecanismo Laplaciano truncado. Cierra la brecha multiplicativa constante previa del caso discreto.

---

## 2. Formulación del problema

Función de consulta real $q: \mathcal{D} \to \mathbb{R}$. Datasets vecinos $D_1, D_2$: difieren en a lo sumo un elemento.

**Definición 1 ($(\epsilon,\delta)$-privacidad diferencial).** Un mecanismo aleatorizado $K$ satisface $(\epsilon,\delta)$-DP si para todo $D_1,D_2$ vecinos y todo conjunto medible $S \subset \text{Range}(K)$:

$$
\Pr[K(D_1) \in S] \le e^\epsilon \Pr[K(D_2)\in S] + \delta \tag{1}
$$

**Definición 2 (Sensibilidad).** $\Delta := \max_{D_1,D_2} |q(D_1)-q(D_2)|$ para vecinos.

Mecanismo de adición de ruido: $K(D) = q(D) + X$, con $X \sim P$.

**Lema 1.** $P$ preserva $(\epsilon,\delta)$-DP si y solo si

$$
P(S) - e^\epsilon P(S+d) \le \delta, \quad \forall |d|\le\Delta,\ \forall S \subset \mathbb{R} \text{ medible} \tag{2}
$$

*(Extensión al caso continuo de la Ec. (18) en Geng y Viswanath, 2016a.)*

Sea $\mathcal{P}_{\epsilon,\delta}$ el conjunto de distribuciones que satisfacen (2). Se definen:

$$
V_1^* := \inf_{P\in\mathcal{P}_{\epsilon,\delta}} \int_{x\in\mathbb{R}} |x|\, P(dx) \quad \text{(amplitud mínima de ruido)}
$$

$$
V_2^* := \inf_{P\in\mathcal{P}_{\epsilon,\delta}} \int_{x\in\mathbb{R}} x^2\, P(dx) \quad \text{(potencia mínima de ruido)}
$$

El objetivo es caracterizar $V_1^*, V_2^*$ vía cotas ajustadas: $V_1^{low} \le V_1^* \le V_1^{upp}$ y $V_2^{low}\le V_2^*\le V_2^{upp}$.

---

## 3. Cota superior: mecanismo Laplaciano truncado

Densidad Laplaciana estándar: $f(x) = \frac{\epsilon}{2\Delta} e^{-\epsilon|x|/\Delta}$, con razón de decaimiento $f(x)/f(x+\Delta) = e^\epsilon$ (óptima bajo $\epsilon$-DP puro, Geng y Viswanath 2016b). Bajo $(\epsilon,\delta)$-DP, la distribución Laplaciana **no** es óptima por su cola pesada.

**Idea clave:** la densidad debe decaer lo más rápido posible (tasa $e^\epsilon$) mientras el ruido es pequeño, y luego reducirse bruscamente a cero para evitar cola pesada.

**Definición 3 (Distribución Laplaciana truncada).** Dados $0<\delta<\tfrac12$, $\epsilon>0$, $\Delta>0$:

$$
f_{\text{TLap}}(x) := \begin{cases} B e^{-|x|/\lambda}, & x \in [-A,A] \\ 0, & \text{en otro caso} \end{cases} \tag{4}
$$

donde:

$$
\lambda := \frac{\Delta}{\epsilon}, \qquad A := \frac{\Delta}{\epsilon}\log\!\Big(1+\frac{e^\epsilon-1}{2\delta}\Big), \qquad B := \frac{1}{2\lambda(1-e^{-A/\lambda})}
$$

**Propiedades clave:**
- La razón de decaimiento en $[0,A-\Delta]$ es exactamente $e^\epsilon$.
- La masa de probabilidad en $[A-\Delta,A]$ es exactamente $\delta$.
- La razón de decaimiento es $+\infty$ para $x \in (A-\Delta,A]$ (la densidad cae a cero fuera de $[-A,A]$).

**Definición 4 (Mecanismo Laplaciano truncado).** Añade ruido con distribución $\mathcal{P}_{\text{TLap}}$ definida en (4).

**Teorema 1.** El mecanismo Laplaciano truncado preserva $(\epsilon,\delta)$-DP.

*Idea de la demostración:* se maximiza $\mathcal{P}_{\text{TLap}}(S) - e^\epsilon \mathcal{P}_{\text{TLap}}(S+d)$ sobre $S$; por simetría y monotonía de $f_{\text{TLap}}$, el máximo se alcanza en $S=[A-\Delta,+\infty)$, y su valor es a lo sumo $\int_{A-\Delta}^{A} f_{\text{TLap}}(x)\,dx = \delta$.

### Cotas superiores en amplitud y potencia

**Teorema 2 (Cota superior — amplitud).**

$$
V_1^* \le V_1^{upp} := \frac{\Delta}{\epsilon}\left(1 - \frac{\log\!\big(1+\frac{e^\epsilon-1}{2\delta}\big)}{\frac{e^\epsilon-1}{2\delta}}\right) \tag{5}
$$

Interpretación: el primer término $\Delta/\epsilon$ es la amplitud del mecanismo Laplaciano puro ($\epsilon$-DP); el segundo reduce el ruido por la relajación $\delta$.

**Comportamiento asintótico** (regímenes de alta privacidad):

- Fijo $\epsilon$, $\delta \to 0$: $V_1^{upp} \to \Delta/\epsilon$ (se recupera el Laplaciano estándar).
- Fijo $\delta$, $\epsilon \to 0$: $V_1^{upp} \to \Delta/(4\delta)$ (la distribución se aproxima a una uniforme en $[-\Delta/(2\delta), \Delta/(2\delta)]$ con densidad $\delta/\Delta$).
- Régimen $\delta=\epsilon \to 0$: $V_1^{upp} \approx \frac{\Delta}{\epsilon}\big(1 - 2\log\tfrac32\big)$ (Ec. 6). Se demuestra en la Sección 5 que esta constante es **ajustada**.

**Teorema 3 (Cota superior — potencia).**

$$
V_2^* \le V_2^{upp} := \frac{2\Delta^2}{\epsilon^2}\left(1 - \frac{\tfrac12\log^2\!\big(1+\tfrac{e^\epsilon-1}{2\delta}\big) + \log\!\big(1+\tfrac{e^\epsilon-1}{2\delta}\big)}{\tfrac{e^\epsilon-1}{2\delta}}\right) \tag{7}
$$

---

## 4. Cota inferior

Técnica: discretizar la distribución de probabilidad continua y la función de pérdida, transformando el problema funcional continuo en programación lineal; se aplica el resultado discreto de Geng y Viswanath (2016a) vía dualidad de PL.

Sean $a := \frac{\delta + \frac{e^\epsilon-1}{2}}{e^\epsilon}$, $b := e^{-\epsilon}$. (Se asume que existe entero $n$ con $\sum_{k=0}^{n-1} ab^k = \tfrac12$.)

**Lema 2 (Teorema 8 en Geng y Viswanath 2016a).** Para una función de costo simétrica $L: \mathbb{Z}\to\mathbb{R}$, con sensibilidad discreta $\tilde\Delta \in \mathbb{Z}^+$, si una distribución discreta $P$ satisface la restricción discreta de $(\epsilon,\delta)$-DP y $L$ satisface una condición de crecimiento apropiada, entonces:

$$
\sum_{i\in\mathbb{Z}} L(i) P(i) \ge 2\sum_{k=0}^{n-1} a b^k L(1+k\tilde\Delta) \tag{10}
$$

**Teorema 4 (Cota inferior — amplitud).**

$$
V_1^* \ge V_1^{low} := 2a\left(\frac{b-b^n}{(1-b)^2} - \frac{(n-1)b^n}{1-b}\right)\Delta \tag{11}
$$

*Demostración (esquema):* se discretiza $P$ en bins de ancho $\Delta/N$ y se aplica el Lema 2 con sensibilidad discreta $N$, tomando $N\to\infty$.

**Teorema 5 (Cota inferior — potencia).**

$$
V_2^* \ge V_2^{low} := 2\sum_{k=0}^{n-1} a b^k k^2\Delta^2 \tag{12}
$$

(fórmula cerrada análoga, ver artículo original).

---

## 5. Ajuste (tightness) de las cotas

**Teorema 6 (Ajuste — amplitud).**

$$
\lim_{\epsilon\to0} \frac{V_1^{low}}{V_1^{upp}} \ge 1-2\delta, \qquad
\lim_{\delta\to0} \frac{V_1^{low}}{V_1^{upp}} \ge \frac{\epsilon}{e^\epsilon-1} = 1-\frac{\epsilon}{2}+O(\epsilon^2), \qquad
\lim_{\epsilon=\delta\to0} \frac{V_1^{low}}{V_1^{upp}} = 1
$$

Esto **cierra la brecha multiplicativa constante previa** en el escenario discreto (Ecs. 67, 69 de Geng y Viswanath, 2016a).

**Teorema 7 (Ajuste — potencia).**

$$
\lim_{\epsilon\to0} \frac{V_2^{low}}{V_2^{upp}} \ge 1-3\delta+2\delta^2, \qquad
\lim_{\delta\to0} \frac{V_2^{low}}{V_2^{upp}} \ge \frac{\epsilon^2(1+e^\epsilon)}{2(e^\epsilon-1)^2} = 1-\frac{\epsilon}{2}+O(\epsilon^2), \qquad
\lim_{\epsilon=\delta\to0} \frac{V_2^{low}}{V_2^{upp}} = 1
$$

---

## 6. Comparación con el mecanismo Gaussiano óptimo

Resultado clásico: ruido Gaussiano con $\sigma = \frac{\sqrt{2\log(1.25/\delta)}}{\epsilon}\Delta$ preserva $(\epsilon,\delta)$-DP (Dwork y Roth, 2014). Balle y Wang (2018) desarrollaron el **mecanismo Gaussiano óptimo**, calibrado directamente con la CDF Gaussiana (no una aproximación por cota de cola).

**Resultados numéricos** ($\epsilon \in [10^{-4},10]$, $\delta \in [10^{-6},0.1]$): el mecanismo Laplaciano truncado reduce significativamente tanto la amplitud como la potencia del ruido respecto al Gaussiano óptimo, en **todos** los regímenes de privacidad. Esto se debe a que el Laplaciano truncado mejora universalmente la razón de decaimiento de la densidad (tanto para ruidos pequeños como grandes).

---

## 7. Conclusión y discusión

- Se caracterizan cotas ajustadas para amplitud y potencia mínima de ruido bajo $(\epsilon,\delta)$-DP.
- El mecanismo Laplaciano truncado es (casi) óptimo en todos los regímenes de alta privacidad.
- **Limitación:** el soporte del ruido Laplaciano truncado está acotado en $[-A,A]$; para dos datasets vecinos, los rangos de salida tendrán un conjunto no solapado. Con probabilidad hasta $\delta$, un adversario podría distinguir los dos datasets vecinos. Posible mejora: imponer una cola ligera arbitraria sobre $[A,+\infty)$ para igualar el espacio de salidas entre datasets.

---

## Notación y fórmulas de referencia rápida

| Símbolo | Significado |
|---|---|
| $\Delta$ | Sensibilidad de la consulta |
| $\epsilon, \delta$ | Parámetros de privacidad |
| $\lambda = \Delta/\epsilon$ | Escala del Laplaciano |
| $A$ | Punto de truncamiento |
| $B$ | Constante de normalización |
| $V_1^*, V_2^*$ | Amplitud / potencia mínima de ruido (óptimos) |
| $V_1^{upp}, V_2^{upp}$ | Cotas superiores (Laplaciano truncado) |
| $V_1^{low}, V_2^{low}$ | Cotas inferiores (dualidad LP, vía discretización) |

## Referencias clave citadas

- Dwork, McSherry, Nissim, Smith (2006b) — "Calibrating noise to sensitivity in private data analysis"
- Dwork, Kenthapadi, McSherry, Mironov, Naor (2006a) — mecanismo de privacidad aproximada
- Balle, Wang (2018) — "Improving the Gaussian mechanism for differential privacy"
- Geng, Viswanath (2016a, 2016b) — mecanismo óptimo de ruido bajo DP
- Dwork, Roth (2014) — *The Algorithmic Foundations of Differential Privacy*
