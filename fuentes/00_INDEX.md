# Índice — Papers convertidos a Markdown

Referencia rápida de la base teórica para construir indicadores de evaluación (clasificación ordinal, privacidad diferencial, métricas de clasificación/regresión).

| Archivo | Tema | Útil para |
|---|---|---|
| `01_nonparametric_ordinal_classification_monotonicity.md` | Clasificación ordinal no paramétrica con restricciones de monotonía (Kotłowski & Słowiński) | Regresión/clasificación isotónica, restricciones de monotonía en modelos |
| `02_privacy_utility_tradeoff_approximate_dp.md` | Privacidad diferencial (ε,δ), mecanismo Laplaciano truncado (Geng et al.) | Añadir ruido calibrado, mecanismos de privacidad |
| `03_powers_informedness_markedness_correlation.md` | Informedness, Markedness, Correlación de Matthews, más allá de Precision/Recall/F1 (Powers) | Métricas de evaluación de clasificadores no sesgadas por prevalencia |
| `04_fawcett_introduction_to_roc_analysis.md` | Introducción a análisis ROC, AUC, envolvente convexa (Fawcett) | Curvas ROC, AUC, comparación de clasificadores |
| `05_chai_draxler_rmse_vs_mae.md` | RMSE vs. MAE, desigualdad triangular (Chai & Draxler) | Métricas de error para modelos de regresión |

Cada archivo conserva la estructura del paper (resumen, secciones, teoremas) con las fórmulas en **LaTeX** (`$...$` y `$$...$$`), en lugar del texto plano roto que produce la extracción directa de un PDF (fracciones, subíndices y sumatorias mal formateadas). Al final de cada archivo hay un bloque "Fórmulas esenciales para implementación" pensado como referencia rápida al codear.

---

## Nota sobre alternativas para que Copilot use esta información

Markdown con LaTeX es una opción razonable, pero según cómo trabajes hay alternativas que pueden darte mejores resultados:

1. **Adjuntar los PDFs directamente como contexto.** Las versiones recientes de GitHub Copilot Chat (en VS Code) permiten adjuntar PDFs/imágenes como contexto de la conversación. Si tu IDE lo soporta, esto evita cualquier pérdida de fidelidad en la conversión — aunque para fórmulas matemáticas la extracción de texto de un PDF suele ser peor que este Markdown ya limpiado.

2. **Un solo archivo "knowledge base" consolidado.** Si vas a referenciar estos papers repetidamente en muchos prompts, te puede convenir concatenar los 5 `.md` en un único archivo (o usar `# Instrucciones personalizadas` / archivo de contexto del repo, como `.github/copilot-instructions.md` en Copilot) para que el modelo tenga siempre ese contexto disponible sin que tengas que adjuntar archivos sueltos cada vez.

3. **Extraer solo las fórmulas a un módulo de referencia por indicador.** Si vas a implementar funciones concretas (Informedness, RMSE, AUC, regresión isotónica, mecanismo Laplaciano truncado), puede ser más eficiente crear un `.md` corto por indicador con la fórmula, su derivación mínima y pseudocódigo — reduce ruido y mejora la precisión de las sugerencias de Copilot frente a pegar el paper completo.

4. **Notebooks (.ipynb) con celdas Markdown + código.** Si vas a validar las implementaciones numéricamente, un notebook con la teoría en celdas Markdown junto al código de la métrica (y tests contra los ejemplos numéricos de los propios papers, como la Tabla 1 de Chai & Draxler o la Tabla 2 de Powers) te da verificación directa, no solo documentación.

Para tu caso (Copilot leyendo desde el IDE para construir indicadores), lo más simple y efectivo suele ser: mantener estos `.md` en el repo (por ejemplo en `docs/references/`) y referenciarlos por nombre en tus prompts a Copilot, o pegarlos como contexto en el chat cuando implementes cada indicador específico.
