# Root Mean Square Error (RMSE) or Mean Absolute Error (MAE)? — Arguments Against Avoiding RMSE in the Literature

**Autores:** T. Chai (NOAA ARL / CICS, Univ. of Maryland), R. R. Draxler (NOAA ARL)
**Publicación:** Geoscientific Model Development, 7, 1247–1250, 2014. doi:10.5194/gmd-7-1247-2014

## Resumen

Tanto el RMSE como el MAE se emplean regularmente en estudios de evaluación de modelos. Willmott y Matsuura (2005) sugirieron que el RMSE no es un buen indicador del desempeño promedio del modelo y puede ser engañoso, proponiendo el MAE como mejor métrica. Este artículo argumenta que, si bien algunas preocupaciones sobre el RMSE son válidas, **evitarlo completamente en favor del MAE no es la solución**. Se demuestra que:

1. El RMSE **no es ambiguo** en su significado (contrario a lo afirmado por Willmott et al., 2009).
2. El RMSE es **más apropiado** que el MAE cuando se espera que la distribución de errores sea Gaussiana.
3. El RMSE **sí satisface** la desigualdad triangular requerida para ser una métrica de distancia (contrario a lo indicado por Willmott et al., 2009, quienes afirmaron que las estadísticas basadas en suma de cuadrados no la satisfacen).

Los autores no sostienen que el RMSE sea superior al MAE — una **combinación de métricas** suele ser necesaria.

---

## 1. Introducción

El RMSE es un estándar en meteorología, calidad del aire e investigación climática. El MAE es otra medida ampliamente usada. No hay consenso sobre cuál es más apropiada.

- El MAE da el mismo peso a todos los errores.
- El RMSE penaliza la varianza: da más peso a errores de mayor valor absoluto.
- Por definición, **RMSE $\ge$ MAE siempre**.
- Ejemplo (Chai et al., 2009): razón RMSE/MAE entre 1.63 y 2.29 en predicciones de columna de NO2 vs. observaciones satelitales SCIAMACHY.

**Argumento de Willmott y Matsuura (2005):** con un conjunto hipotético de 4 errores, manteniendo el MAE constante en 2.0, el RMSE varía entre 2.0 y 4.0. Concluyen que el RMSE varía con la variabilidad de las magnitudes de error, la magnitud total/promedio de error (MAE) y el tamaño de muestra $n$. El RMSE tiende a crecer más que el MAE con $n^{1/2}$ (su límite inferior es el MAE; su límite superior es $n^{1/2}\cdot\text{MAE}$).

**Willmott et al. (2009):** concluyen que las estadísticas de error basadas en suma de cuadrados (RMSE, error estándar) tienen ambigüedades inherentes y recomiendan alternativas como el MAE.

**Postura de este artículo:** toda estadística condensa muchos datos en un solo valor, ofreciendo solo una proyección del error, enfatizando cierto aspecto. Willmott y Matsuura solo probaron que RMSE $\ne$ MAE (no son intercambiables), lo cual es esperable dado que se definen de forma diferente — esto no implica ambigüedad ni invalidez del RMSE.

---

## 2. Interpretación del RMSE y del MAE

Sea un conjunto de $n$ muestras de error del modelo $e_i$, $i=1,\dots,n$ (no se consideran errores de observación; se asume el conjunto de errores insesgado):

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |e_i| \tag{1}
$$

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n e_i^2} \tag{2}
$$

**Supuesto subyacente al presentar el RMSE:** los errores son insesgados y siguen una distribución **normal**. Bajo ese supuesto, el RMSE (o el error estándar, SE) ayuda a reconstruir la distribución completa del error.

> **Nota (pie de página del original):** Para distribuciones de error insesgadas, el SE es equivalente al RMSE (media muestral asumida cero). Para distribución desconocida:
> $$
> SE = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (e_i-\bar e)^2}, \qquad \bar e = \frac{1}{n}\sum_{i=1}^n e_i
> $$

### Tabla 1 — Simulación: errores Gaussianos de media cero y varianza unitaria

RMSE y MAE de 5 conjuntos de errores pseudo-aleatorios generados con distintas semillas, para distintos tamaños de muestra $n$:

| $n$ | RMSEs (5 corridas) | MAEs (5 corridas) |
|---|---|---|
| 4 | 0.92, 0.65, 1.48, 1.02, 0.79 | 0.70, 0.57, 1.33, 1.16, 0.76 |
| 10 | 0.81, 1.10, 0.83, 0.95, 1.01 | 0.65, 0.89, 0.72, 0.84, 0.78 |
| 100 | 1.05, 1.03, 1.03, 1.00, 1.04 | 0.82, 0.81, 0.79, 0.78, 0.78 |
| 1000 | 1.04, 0.98, 1.01, 1.00, 1.00 | 0.82, 0.78, 0.80, 0.80, 0.81 |
| 10 000 | 1.00, 0.98, 1.01, 1.00, 1.00 | 0.79, 0.79, 0.79, 0.81, 0.80 |
| 100 000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.80, 0.80, 0.80, 0.80, 0.80 |
| 1 000 000 | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.80, 0.80, 0.80, 0.80, 0.80 |

**Observaciones clave:**
- Con $n \ge 100$, el RMSE reconstruye la desviación estándar "verdadera" (SE=1) con error $\le5\%$.
- El MAE converge a $\approx 0.8$, aproximando $\sqrt{2/\pi}$ (media de la distribución semi-normal, i.e. el promedio del subconjunto positivo de errores normales de media cero).
- Con muestras muy pequeñas ($n=4$ o $n=10$), **ni RMSE ni MAE son robustos** — en esos casos es preferible presentar los valores de error individuales en tablas.

**Nota importante:** el MAE es adecuado para describir errores **uniformemente distribuidos**. Como los errores de modelo suelen seguir una distribución normal (no uniforme), el **RMSE es la métrica más apropiada** para ese tipo de dato.

---

## 3. Desigualdad triangular de una métrica

Tanto Willmott y Matsuura (2005) como Willmott et al. (2009) afirmaron que las estadísticas basadas en suma de cuadrados **no** satisfacen la desigualdad triangular. Ejemplo dado (nota al pie de Willmott et al., 2009): $d(a,c)=4$, $d(a,b)=2$, $d(b,c)=3$. Una "métrica" $d(x,y)$ debe satisfacer $d(a,c) \le d(a,b)+d(b,c)$. Los autores argumentan que la suma de errores al cuadrado no satisface esto porque $4^2 \not\le 2^2+3^2$.

**Refutación de Chai y Draxler:** ese ejemplo en realidad representa el **MSE** (error cuadrático medio, sin raíz), que efectivamente **no** es una métrica de distancia válida — pero **no es el RMSE**.

### Demostración de que el RMSE sí satisface la desigualdad triangular

Sea $\mathbf{e} = (e_1,\dots,e_n)$ un vector n-dimensional de errores. Normas:

$$
\|\mathbf{e}\|_1 = \sum_{i=1}^n |e_i| = n\cdot\text{MAE} \tag{3}
$$

$$
\|\mathbf{e}\|_2 = \sqrt{\sum_{i=1}^n e_i^2} = \sqrt{n}\cdot\text{RMSE} \tag{4}
$$

Toda norma vectorial satisface $\|X+Y\| \le \|X\|+\|Y\|$ y $\|-X\|=\|X\|$ (Horn y Johnson, 1990). Es trivial que la distancia entre dos vectores medida por norma-$L_p$ satisface $\|X-Y\|_p \le \|X\|_p + \|Y\|_p$. Con tres vectores n-dimensionales $X,Y,Z$:

$$
\|X-Y\|_p = \|(X-Z)-(Y-Z)\|_p \le \|X-Z\|_p + \|Y-Z\|_p \tag{5}
$$

Para la norma $L_2$ y vectores n-dimensionales, (5) se escribe:

$$
\sqrt{\sum_{i=1}^n (x_i-y_i)^2} \le \sqrt{\sum_{i=1}^n (x_i-z_i)^2} + \sqrt{\sum_{i=1}^n (y_i-z_i)^2} \tag{6}
$$

lo cual es equivalente a:

$$
\sqrt{\frac{1}{n}\sum_{i=1}^n (x_i-y_i)^2} \le \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i-z_i)^2} + \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i-z_i)^2} \tag{7}
$$

**Conclusión: el RMSE satisface la desigualdad triangular requerida para una métrica de distancia.** (La confusión de Willmott et al. surge de aplicar el argumento al MSE —sin raíz cuadrada— y no al RMSE.)

---

## 4. Resumen y discusión

### Puntos principales

1. **El RMSE no es ambiguo** en su significado, y es más apropiado que el MAE cuando se espera una distribución de error normal con muestra suficiente.
2. **El RMSE satisface la desigualdad triangular** (métrica de distancia válida), a diferencia de lo indicado en la literatura previa citada.

### Sensibilidad a outliers

Preocupación común: sensibilidad del RMSE a valores atípicos. Según los autores, la existencia de outliers y su probabilidad de ocurrencia está bien descrita por la distribución normal subyacente al uso del RMSE. Con $n \ge 100$, incluyendo outliers, se puede reconstruir bien la distribución de error (ver Tabla 1). En la práctica, puede justificarse remover outliers varios órdenes de magnitud mayores que el resto (especialmente con muestras limitadas), y remover sesgos sistemáticos antes de calcular el RMSE si el modelo tiene sesgos severos.

### Ventaja matemática del RMSE

El RMSE **evita el uso de valor absoluto**, indeseable en muchos cálculos matemáticos (ej. dificultad para calcular gradiente/sensibilidad del MAE respecto a parámetros del modelo). En **asimilación de datos**, la suma de errores al cuadrado se usa a menudo como función de costo a minimizar — penalizar errores grandes mediante términos de mínimos cuadrados resulta muy efectivo para mejorar el desempeño del modelo. En estos contextos (sensibilidad de errores del modelo, asimilación de datos), **el MAE definitivamente no es preferible al RMSE**.

### Capacidad discriminativa

Una medida de error más discriminativa (mayor variación entre distintos conjuntos de resultados) es generalmente más deseable. El MAE puede verse afectado por una gran cantidad de errores promedio pequeños sin reflejar adecuadamente algunos errores grandes. Dando mayor peso a condiciones desfavorables, el **RMSE suele ser mejor para revelar diferencias de desempeño del modelo**.

### Cuándo cada uno es preferible

- En estudios de sensibilidad que usan **solo RMSE**: interpretación detallada no es crítica si el mismo modelo (variaciones) tiene distribuciones de error similares.
- Al evaluar **distintos modelos** con una sola métrica: las diferencias en distribución de error se vuelven más importantes.
- El supuesto subyacente al RMSE: errores insesgados y con distribución normal. Para otras distribuciones, se necesitan más momentos estadísticos (media, varianza, asimetría, curtosis) para una imagen completa.
- Enfoques robustos a outliers o insensibles a distribuciones no normales: Tukey (1977), Huber y Ronchetti (2009).

### Conclusión final

> "Cualquier métrica individual provee solo una proyección de los errores del modelo y, por tanto, enfatiza solo cierto aspecto de sus características. **Una combinación de métricas, incluyendo pero no limitándose a RMSE y MAE, suele ser necesaria para evaluar el desempeño del modelo.**"

---

## Fórmulas esenciales para implementación (resumen rápido)

```
MAE  = (1/n) * Σ|e_i|
RMSE = sqrt( (1/n) * Σ(e_i^2) )

Relación con normas vectoriales:
||e||_1 = n * MAE
||e||_2 = sqrt(n) * RMSE

Propiedad: RMSE >= MAE siempre (por Cauchy-Schwarz / desigualdad de potencias medias)
Propiedad: RMSE satisface desigualdad triangular (es una métrica de distancia L2 válida); MSE (sin raíz) NO la satisface.

Guía de uso:
- Distribución de error ~ Normal, n grande        -> preferir RMSE (o reportar ambos)
- Distribución de error ~ Uniforme                -> MAE más representativo
- Cálculo de gradientes / asimilación de datos     -> RMSE (evita valor absoluto, diferenciable)
- Se desea igual peso a todos los errores          -> MAE
- Se desea penalizar más los errores grandes       -> RMSE
```

## Referencias clave citadas

- Willmott, C., Matsuura, K. (2005). "Advantages of the MAE over the RMSE in assessing average model performance." *Climate Research* 30, 79–82.
- Willmott, C. J., Matsuura, K., Robeson, S. M. (2009). "Ambiguities inherent in sums-of-squares-based error statistics." *Atmospheric Environment* 43, 749–752.
- Horn, R. A., Johnson, C. R. (1990). *Matrix Analysis*, Cambridge University Press.
- Chai, T. et al. (2009). Inversión de emisiones de NOx regional vía enfoque variacional 4D, *Atmos. Environ.* 43, 5046–5055.
- Tukey, J. W. (1977). *Exploratory Data Analysis*, Addison-Wesley.
- Huber, P., Ronchetti, E. (2009). *Robust Statistics*, Wiley.
