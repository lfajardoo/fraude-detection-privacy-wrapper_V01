# Bitácora de Evolución de Arquitectura - Privacy API

**Proyecto:** Sistema de Detección de Fraude con Privacy Engineering  
**Última actualización:** 2026-08-15

---

## Índice

1. [Visión General](#visión-general)
2. [Arquitectura Actual (Fase 1)](#arquitectura-actual-fase-1)
3. [Métricas de Evaluación (Fase 1.5 — implementada)](#métricas-de-evaluación-fase-15--implementada)
4. [Propuesta: Árbol de Decisión (Fase 2)](#propuesta-árbol-de-decisión-fase-2)
5. [Evaluación del Diseño](#evaluación-del-diseño)
6. [Roadmap de Evolución](#roadmap-de-evolución)
7. [Implementación Técnica](#implementación-técnica)
8. [Notas de Desarrollo](#notas-de-desarrollo)

---

## Visión General

### Objetivo del Sistema

Sistema de detección de fraude que aplica **Privacidad Diferencial (DP)** sobre las predicciones de un modelo de Machine Learning para proteger la privacidad de las transacciones sin comprometer completamente la utilidad de las predicciones.

### Componentes Principales

```
┌─────────────────────────────────────────────────────┐
│                    Cliente                          │
└─────────────────┬───────────────────────────────────┘
                  │ POST /predict
                  ▼
┌─────────────────────────────────────────────────────┐
│              privacy-api (puerto 8001)              │
│  ┌──────────────────────────────────────────────┐  │
│  │ app.py (orquestación)                        │  │
│  └──────────┬───────────────────────────────────┘  │
│             │                                       │
│             ▼                                       │
│  ┌──────────────────────────────────────────────┐  │
│  │ governor.py (toma decisiones)                │  │
│  │  ↓                                            │  │
│  │ noise_selector.py (lógica de selección)      │  │
│  └──────────┬───────────────────────────────────┘  │
│             │                                       │
│             ▼                                       │
│  ┌──────────────────────────────────────────────┐  │
│  │ privacy_layer.py (aplica ruido)              │  │
│  └──────────────────────────────────────────────┘  │
└────────┬────────────────────────────────────────────┘
         │ POST /predict
         ▼
┌─────────────────────────────────────────────────────┐
│           modelo-python (puerto 8000)               │
│         (Regresión Logística - Fraude)              │
└─────────────────────────────────────────────────────┘
```

---

## Arquitectura Actual (Fase 1)

### Estado: Hardcodeado

**Fecha de implementación:** Inicial  
**Versión:** 1.0

#### Características

- **Governor:** Retorna siempre los mismos parámetros fijos
  - `mechanism = "laplace"`
  - `epsilon = 1.0`
  - `delta = 1e-6`

- **Noise repetitions:** Soporta múltiples aplicaciones de ruido por request
  - Campo: `noise_repetitions` (1-20)
  - Predicción del modelo: **1 sola vez** (eficiente)
  - Aplicación de ruido: **N veces** (una por repetición)

#### Archivos Actuales

```
privacy-api/
├── app.py                 # Orquestación, endpoints, loop de repeticiones
├── governor.py            # Hardcodeado: decide() → valores fijos
├── privacy_layer.py       # Implementa Laplace/Gaussian, aplica ruido
├── client.py              # Cliente HTTP → modelo-python
├── config.py              # Parámetros globales (epsilon, delta, bounds)
├── requirements.txt
└── tests.py
```

#### Flujo de una Predicción

```python
# En app.py, endpoint /predict
for i in range(repetitions):
    decision = governor.decide()  # Sin contexto, siempre igual
    result_with_dp = dp_layer.apply_dp_to_response(
        model_prediction.copy(), decision=decision
    )
```

#### Limitaciones

- ❌ No considera el contexto de la transacción
- ❌ Mismo nivel de ruido para todas las situaciones
- ❌ No aprende de datos históricos
- ✅ Funcional, simple, predecible

---

## Métricas de Evaluación (Fase 1.5 — implementada)

**Fecha:** 2026-08-15  
**Versión:** 1.5

### Motivación

Antes de diseñar la gobernanza dinámica (Fase 2) se necesita un marco de
evaluación que responda cuantitativamente: ¿cuánta utilidad pierde el sistema
al añadir ruido DP? ¿cuántos fraudes reales quedan sin detectar? ¿el ruido es
independiente del nivel de riesgo?

### Base Teórica

| Dimensión | Métrica(s) | Referencia |
|---|---|---|
| Comportamiento del ruido | MAE, RMSE, ratio RMSE/MAE | Chai & Draxler, GMD 2014 |
| Utilidad (clasificación) | Accuracy, F1 | Powers 2011; Fawcett 2006 |
| Utilidad (libre de sesgo) | Informedness (BM), MCC | Powers 2011 |
| Sensibilidad | FNR (fraudes que escapan) | Fawcett 2006 |
| Tradeoff privacidad-utilidad | Utility Retention, Risk-Noise Correlation | Geng et al., AISTATS 2020 |
| Adaptatividad (Fase 2 prep) | Monotonicity constraint | Kotłowski & Słowiński, TKDE 2013 |

### Estructura de Archivos Añadidos

```
privacy-api/
├── metrics/
│   ├── __init__.py               # re-exports limpios de todo el paquete
│   ├── noise_metrics.py          # mae(), rmse(), noise_summary()
│   ├── classification_metrics.py # accuracy(), fnr(), f1(),
│   │                             #   informedness(), markedness(), mcc(),
│   │                             #   classification_report()
│   └── privacy_metrics.py        # utility_retention(),
│                                 #   risk_noise_correlation(),
│                                 #   privacy_utility_summary()
└── evaluator.py                  # BatchEvaluator — orquesta métricas
```

### Nuevo Endpoint: `POST /evaluate`

Recibe una lista de transacciones etiquetadas con ground-truth y devuelve un
informe completo de métricas. Límite: `EVALUATION_MAX_TRANSACTIONS` (config).

**Request:**
```json
{
  "transactions": [
    { "Time": 0, "V1": -1.35, ..., "Amount": 149.62, "label": 1 }
  ],
  "mode": "governed",
  "threshold": 0.5
}
```

**Response (estructura):**
```json
{
  "n_transactions": 100,
  "noise_metrics": {
    "fraud_probability": { "mae": 0.12, "rmse": 0.18, "rmse_mae_ratio": 1.5, "n_samples": 100 }
  },
  "classification_original": {
    "accuracy": 0.97, "fnr": 0.04, "f1": 0.83,
    "informedness": 0.76, "markedness": 0.71, "mcc": 0.73
  },
  "classification_with_dp": { "..." },
  "utility_retention": {
    "accuracy": 0.98, "f1": 0.91, "informedness": 0.88, "mcc": 0.89
  },
  "privacy_utility": {
    "utility_retention": 0.91, "risk_noise_correlation": 0.03
  }
}
```

### Contrato de Monotonía (Fase 2 — governor.py)

`governor.py` expone:
- `MONOTONE_EPSILON_SCHEDULE`: mapeo canónico riesgo → ε (decreasing)
- `validate_monotone_schedule(schedule)`: verifica invariante en tiempo de carga
- `epsilon_from_risk(fraud_probability)`: lookup monotóno listo para Fase 2

La Fase 2 reemplazará la lógica hardcodeada de `decide()` por un regresor
isotónico entrenado sobre datos, pero debe continuar pasando
`validate_monotone_schedule` para garantizar la propiedad teórica
[Kotłowski & Słowiński §3].

### Principios de Extensibilidad

- **Añadir una métrica nueva:** añadir función en el módulo correspondiente
  de `metrics/` y re-exportar desde `metrics/__init__.py`. No tocar `app.py`.
- **Añadir un modelo nuevo:** crear nuevo `client_*.py` siguiendo el contrato
  de `ModelAPIClient`; `evaluator.py` acepta cualquier cliente que implemente
  `.predict(features) → dict`.
- **Añadir una dimensión nueva** (p. ej. equidad/fairness): crear
  `metrics/fairness_metrics.py` con sus funciones y añadir la sección
  correspondiente en `evaluator.py`.

---

## Propuesta: Árbol de Decisión (Fase 2)

### Estado: En Diseño

**Fecha propuesta:** 2026-08-02  
**Versión objetivo:** 2.0

### Motivación

Queremos que el sistema **decida dinámicamente** qué nivel de ruido aplicar basándose en:
- Características de la transacción (`Amount`, `Time`, etc.)
- Confianza del modelo (`fraud_probability`, `confidence_score`)
- Predicción del modelo (`is_fraud`)

**Ejemplo de decisión inteligente:**
- Si `fraud_probability > 0.8` → Más privacidad (ε=0.1)
- Si `fraud_probability < 0.2` → Menos privacidad (ε=5.0)
- Transacciones grandes → Más privacidad

### Arquitectura Propuesta

#### Nuevo archivo: `noise_selector.py`

```python
"""
Selector de ruido con árbol de decisión (sklearn)
"""

class DecisionTreeNoiseSelector:
    def __init__(self, model_path="models/noise_decision_tree.pkl"):
        self.model = joblib.load(model_path)
    
    def select(self, model_prediction, transaction) -> NoiseSelection:
        X = self._extract_features(model_prediction, transaction)
        epsilon_predicted = self.model.predict(X)[0]
        
        return NoiseSelection(
            mechanism="laplace",
            epsilon=float(epsilon_predicted),
            delta=1e-6,
            reason=f"Árbol de decisión (ε={epsilon_predicted:.2f})"
        )
```

#### Modificación: `governor.py`

```python
from noise_selector import noise_selector

class PrivacyGovernor:
    def decide(self, model_prediction=None, transaction=None):
        if model_prediction is None or transaction is None:
            return GovernanceDecision(...)  # Default
        
        # Delegar al árbol de decisión
        selection = noise_selector.select(model_prediction, transaction)
        
        return GovernanceDecision(
            mechanism=selection.mechanism,
            epsilon=selection.epsilon,
            delta=selection.delta,
            budget_allowed=True
        )
```

#### Modificación: `app.py`

```python
# En el loop de repeticiones, pasar contexto al governor
for i in range(repetitions):
    decision = governor.decide(
        model_prediction=model_prediction,  # ← agregar
        transaction=transaction_dict         # ← agregar
    )
    result_with_dp = dp_layer.apply_dp_to_response(...)
```

### Features del Árbol de Decisión

#### Variables de Entrada (X)

| Feature | Fuente | Tipo | Rango |
|---------|--------|------|-------|
| `fraud_probability` | Predicción del modelo | float | [0, 1] |
| `confidence_score` | Predicción del modelo | float | [0, 1] |
| `Amount` | Transacción | float | [0, ∞] |
| `Time` | Transacción | float | [0, ∞] |
| `is_fraud` | Predicción del modelo | int | {0, 1} |

**Opcionales:** `V1`, `V2`, ..., `V14` si son relevantes.

#### Variable de Salida (y)

**Opción 1: Clasificación (clases discretas)**
- `HIGH_PRIVACY` → ε = 0.1
- `MEDIUM_PRIVACY` → ε = 1.0
- `LOW_PRIVACY` → ε = 5.0

**Opción 2: Regresión (valor continuo)**
- Predecir `epsilon` directamente en el rango [0.1, 10.0]

### Entrenamiento del Modelo

Crear script: `train_noise_selector.py`

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 1. Crear dataset de entrenamiento
# Este dataset debe contener transacciones históricas con el epsilon "ideal" asignado
data = {
    'fraud_probability': [...],
    'confidence_score': [...],
    'Amount': [...],
    'Time': [...],
    'is_fraud': [...],
    'target_epsilon': [...]  # Etiquetas: qué epsilon queremos para cada caso
}
df = pd.DataFrame(data)

# 2. Dividir en train/test
X = df[['fraud_probability', 'confidence_score', 'Amount', 'Time', 'is_fraud']]
y = df['target_epsilon']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar árbol de decisión
tree = DecisionTreeRegressor(
    max_depth=5,           # Evitar overfitting
    min_samples_leaf=10,   # Mínimo de muestras por hoja
    random_state=42
)
tree.fit(X_train, y_train)

# 4. Evaluar
score = tree.score(X_test, y_test)
print(f"R² Score: {score:.4f}")

# 5. Guardar modelo
joblib.dump(tree, 'models/noise_decision_tree.pkl')
print("✓ Árbol guardado en models/noise_decision_tree.pkl")
```

### Estructura de Archivos Final

```
privacy-api/
├── app.py                          # (modificación menor)
├── governor.py                     # (modificación: llamar a noise_selector)
├── noise_selector.py               # (NUEVO: árbol de decisión)
├── privacy_layer.py                # (sin cambios)
├── client.py                       # (sin cambios)
├── config.py                       # (agregar: NOISE_MODEL_PATH)
├── requirements.txt                # (agregar: scikit-learn)
├── tests.py
├── train_noise_selector.py         # (NUEVO: script de entrenamiento)
└── models/
    └── noise_decision_tree.pkl     # (NUEVO: modelo entrenado)
```

---

## Evaluación del Diseño

### ✅ Adecuación del Diseño Actual

| Criterio | Evaluación | Comentario |
|----------|------------|------------|
| **¿Se adecua el diseño?** | ✅ Sí | El governor está diseñado para ser reemplazado (comentarios en el código) |
| **¿Cambios grandes?** | ✅ No | Solo 2 archivos nuevos, 2 modificaciones menores |
| **¿Extensible a futuros modelos?** | ✅ Sí | Solo reemplazar `noise_selector.py` |
| **¿Afecta a privacy_layer?** | ✅ No | Sigue recibiendo `GovernanceDecision` |
| **¿Afecta al modelo de fraude?** | ✅ No | Totalmente independiente |
| **¿Soporta múltiples repeticiones?** | ✅ Sí | El loop ya invoca `governor.decide()` N veces |
| **¿Retrocompatible?** | ✅ Sí | Fallback a valores por defecto si no hay contexto |

### Separación de Responsabilidades

```
┌────────────────────────────────────────────────────┐
│ app.py                                             │
│ Responsabilidad: Orquestar el flujo                │
│ - Recibir request                                  │
│ - Llamar al modelo-python                          │
│ - Iterar N veces para aplicar ruido                │
│ - Construir respuesta                              │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│ governor.py                                        │
│ Responsabilidad: Tomar la decisión de privacidad  │
│ - Recibir contexto (predicción + transacción)     │
│ - Delegar al noise_selector                       │
│ - Retornar GovernanceDecision                     │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│ noise_selector.py                                  │
│ Responsabilidad: Lógica de selección de ruido     │
│ - Extraer features del contexto                   │
│ - Predecir con el árbol de decisión               │
│ - Retornar NoiseSelection                         │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│ privacy_layer.py                                   │
│ Responsabilidad: Aplicar el ruido calibrado       │
│ - Recibir GovernanceDecision                      │
│ - Generar ruido Laplace/Gaussiano                 │
│ - Aplicar ruido a fraud_probability y confidence  │
│ - Retornar predicción con ruido                   │
└────────────────────────────────────────────────────┘
```

**Ventaja:** Cada módulo tiene **una sola responsabilidad**, lo que facilita:
- Testing independiente
- Evolución sin romper otros componentes
- Debugging más sencillo
- Sustitución de componentes (ej: cambiar el árbol por una red neuronal)

---

## Roadmap de Evolución

### Fase 1: Hardcodeado ✅ (Completada)

**Estado:** Implementado  
**Fecha:** Inicial

- Governor con valores fijos
- Soporte para múltiples repeticiones de ruido
- Mecanismos Laplace y Gaussian implementados

### Fase 2: Árbol de Decisión 🔄 (En Diseño)

**Estado:** Planificado  
**Fecha objetivo:** Por definir

**Tareas:**
- [ ] Crear dataset de entrenamiento para el árbol
- [ ] Implementar `train_noise_selector.py`
- [ ] Entrenar árbol de decisión inicial
- [ ] Implementar `noise_selector.py`
- [ ] Modificar `governor.py` para usar el selector
- [ ] Modificar `app.py` para pasar contexto
- [ ] Actualizar `config.py` con rutas de modelos
- [ ] Actualizar `requirements.txt` (agregar scikit-learn)
- [ ] Testing: comparar decisiones con valores hardcodeados
- [ ] Documentar en README

**Dependencias:**
```
scikit-learn>=1.3.0
joblib>=1.3.0
```

### Fase 3: Random Forest 📋 (Planificado)

**Motivación:** Mejorar precisión y reducir overfitting

**Cambios:**
- Sustituir `DecisionTreeRegressor` por `RandomForestRegressor` en `train_noise_selector.py`
- Sin cambios en `noise_selector.py` (misma interfaz)
- Evaluar mejora en métricas

**Ventajas:**
- Mayor robustez
- Menor varianza
- Feature importance más estable

### Fase 4: Red Neuronal 📋 (Futuro)

**Motivación:** Capturar relaciones no lineales complejas

**Tecnología:** TensorFlow/PyTorch

**Cambios:**
- Nuevo modelo en `models/noise_neural_net.h5`
- Modificar `noise_selector.py` para cargar red neuronal
- Feature engineering más sofisticado

### Fase 5: Modelo Online 📋 (Futuro Avanzado)

**Motivación:** Aprendizaje continuo basado en feedback

**Características:**
- Actualización del modelo en tiempo real
- Presupuesto de privacidad acumulado
- Adaptación a patrones cambiantes de fraude

---

## Implementación Técnica

### Consideraciones de Diseño

#### 1. Fallback Strategy

El `noise_selector` debe tener un **fallback** si el modelo no está disponible:

```python
class DecisionTreeNoiseSelector:
    def __init__(self, model_path):
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.trained = True
        else:
            self.model = None
            self.trained = False
            logger.warning("⚠️  Árbol no encontrado, usando lógica por defecto")
    
    def select(self, model_prediction, transaction):
        if self.trained:
            return self._select_with_model(...)
        else:
            return self._select_fallback(...)  # Lógica simple
```

#### 2. Feature Normalization

Si el árbol requiere normalización:

```python
from sklearn.preprocessing import StandardScaler

class DecisionTreeNoiseSelector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    
    def _extract_features(self, model_prediction, transaction):
        features = [...]
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        return X_scaled
```

#### 3. Validación de Salidas

El árbol puede predecir valores fuera de rango válido:

```python
def select(self, model_prediction, transaction):
    epsilon_predicted = self.model.predict(X)[0]
    
    # Clip a rango válido
    epsilon_clipped = np.clip(epsilon_predicted, 0.1, 10.0)
    
    return NoiseSelection(epsilon=epsilon_clipped, ...)
```

#### 4. Logging y Observabilidad

Registrar las decisiones del árbol para análisis posterior:

```python
def select(self, model_prediction, transaction):
    epsilon = self.model.predict(X)[0]
    
    logger.info(
        f"Árbol decidió ε={epsilon:.2f} para "
        f"fraud_prob={model_prediction['fraud_probability']:.3f}, "
        f"amount={transaction['Amount']:.2f}"
    )
    
    return NoiseSelection(epsilon=epsilon, ...)
```

### Testing

#### Unit Tests

```python
# tests/test_noise_selector.py
import pytest
from noise_selector import DecisionTreeNoiseSelector

def test_select_high_fraud_probability():
    selector = DecisionTreeNoiseSelector()
    
    model_pred = {
        'fraud_probability': 0.95,
        'confidence_score': 0.95,
        'is_fraud': 1
    }
    transaction = {'Amount': 10000, 'Time': 0}
    
    selection = selector.select(model_pred, transaction)
    
    # Esperamos más privacidad (epsilon bajo)
    assert selection.epsilon < 1.0

def test_select_low_fraud_probability():
    selector = DecisionTreeNoiseSelector()
    
    model_pred = {
        'fraud_probability': 0.05,
        'confidence_score': 0.95,
        'is_fraud': 0
    }
    transaction = {'Amount': 100, 'Time': 0}
    
    selection = selector.select(model_pred, transaction)
    
    # Esperamos menos privacidad (epsilon alto)
    assert selection.epsilon > 1.0
```

#### Integration Tests

```python
# tests/test_integration.py
def test_full_flow_with_tree():
    response = client.post("/predict", json={
        "Time": 0,
        "V1": -1.35,
        ...,
        "Amount": 149.62,
        "noise_repetitions": 3
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificar que se aplicaron 3 repeticiones
    assert data['noise_repetitions'] == 3
    assert len(data['results']) == 3
    
    # Verificar que cada resultado tiene privacy_info
    for result in data['results']:
        assert 'privacy_info' in result
        assert 'epsilon' in result['privacy_info']
```

---

## Notas de Desarrollo

### Dataset de Entrenamiento

**Desafío:** Crear el dataset inicial con etiquetas (`target_epsilon`) sin tener datos históricos.

**Soluciones:**

#### Opción 1: Etiquetado Manual (Inicial)
Definir reglas de negocio basadas en experiencia del dominio:

```python
def assign_target_epsilon(row):
    """Asignar epsilon objetivo basado en reglas de negocio"""
    if row['fraud_probability'] > 0.8:
        return 0.1  # Alta confianza en fraude → mucha privacidad
    elif row['fraud_probability'] < 0.2:
        return 5.0  # Baja confianza en fraude → poca privacidad
    elif row['Amount'] > 5000:
        return 0.5  # Transacción grande → más privacidad
    else:
        return 1.0  # Caso normal
```

#### Opción 2: Simulación
Generar transacciones sintéticas con variedad de escenarios:

```python
import numpy as np
import pandas as pd

n_samples = 1000
data = {
    'fraud_probability': np.random.random(n_samples),
    'confidence_score': np.random.random(n_samples),
    'Amount': np.random.exponential(1000, n_samples),
    'Time': np.random.uniform(0, 86400, n_samples),
    'is_fraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
}
df = pd.DataFrame(data)
df['target_epsilon'] = df.apply(assign_target_epsilon, axis=1)
```

#### Opción 3: Active Learning (Fase posterior)
- Empezar con reglas simples
- Recopilar feedback de usuarios expertos
- Re-entrenar periódicamente

### Hiperparámetros del Árbol

Valores iniciales recomendados:

```python
DecisionTreeRegressor(
    max_depth=5,              # Evitar overfitting
    min_samples_split=20,     # Mínimo para dividir nodo
    min_samples_leaf=10,      # Mínimo por hoja
    max_features='sqrt',      # Subset de features
    random_state=42           # Reproducibilidad
)
```

**Ajustar mediante:** GridSearchCV o RandomizedSearchCV

### Métricas de Evaluación

Para el árbol de decisión (regresión):

- **R² Score:** Proporción de varianza explicada
- **MAE (Mean Absolute Error):** Error promedio absoluto
- **RMSE (Root Mean Squared Error):** Penaliza errores grandes

Para evaluar el impacto en privacidad:

- **Privacy Budget Consumption:** ε acumulado en ventana de tiempo
- **Utility Loss:** Diferencia entre predicción original y con ruido
- **User Satisfaction:** Feedback cualitativo

### Versionado de Modelos

```
models/
├── noise_decision_tree_v1.0.pkl      # Primera versión
├── noise_decision_tree_v1.1.pkl      # Mejora iterativa
├── noise_decision_tree_current.pkl   # Symlink al actual
└── training_logs/
    ├── v1.0_training.log
    └── v1.1_training.log
```

### Monitoreo en Producción

Métricas a trackear:

```python
# Logging de decisiones del árbol
logger.info({
    'timestamp': datetime.now().isoformat(),
    'fraud_probability': model_prediction['fraud_probability'],
    'amount': transaction['Amount'],
    'epsilon_decided': selection.epsilon,
    'model_version': 'v1.0'
})
```

Alertas:
- Si `epsilon` predicho está siempre en los extremos (0.1 o 10.0)
- Si el modelo no se puede cargar (fallback activo)
- Si el tiempo de inferencia es > 100ms

---

## Referencias

### Papers y Recursos

- **Differential Privacy:** Dwork, C. (2006). "Differential Privacy"
- **Privacy Budget Management:** Dwork, C., Rothblum, G. N. (2016). "Concentrated Differential Privacy"
- **Decision Trees:** Breiman, L. et al. (1984). "Classification and Regression Trees"

### Documentación Técnica

- [Scikit-learn DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [IBM Diffprivlib](https://github.com/IBM/differential-privacy-library)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

## Changelog

### 2026-08-02
- ✨ Documento inicial creado
- 📝 Análisis de arquitectura actual
- 📋 Propuesta de integración de árbol de decisión
- 🗺️ Roadmap de evolución definido

### Próximas Actualizaciones
- [ ] Implementación de Fase 2 (árbol de decisión)
- [ ] Resultados de entrenamiento inicial
- [ ] Métricas de performance comparativa
- [ ] Decisiones de hiperparámetros finales

---

**Fin del documento**
