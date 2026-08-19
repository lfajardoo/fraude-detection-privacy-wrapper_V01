# Explicación de Resultados: `/predict` vs `/evaluate`

---

## � RESUMEN DE POLÍTICAS DE PRIVACIDAD

### Tabla Comparativa de las 4 Políticas

| Política | Condición | epsilon_in | epsilon_out | budget_allowed | Propósito | Acción |
|----------|-----------|-----------|-----------|---|----------|--------|
| **P1** | `distancia_umbral < 0.02` | 3.0 | 5.0 | ✅ TRUE | Zona gris (decisión en el borde) | Máximo ruido DP |
| **P2** | `distancia_umbral ≥ 0.02` + `conf < 0.6` | 1.5 | 2.0 | ✅ TRUE | Baja confianza | Ruido medio DP |
| **P3** | `distancia_umbral ≥ 0.02` + `conf ≥ 0.6` | 0.5 | 0.8 | ✅ TRUE | Predicción clara (CASO NORMAL) | Ruido bajo DP |
| **P4** | `iteracion ≥ 3` | 0.0 | 0.0 | ❌ FALSE | **Presupuesto agotado** | **Bloquea - Requiere revisión** |

### Desglose de Condiciones

```
if distancia_umbral < 0.02:
    → P1 (zona gris - máxima protección)
elif confidence_score < 0.6:
    → P2 (modelo inseguro - protección media)
else:
    → P3 (predicción clara - protección estándar)

if iteracion >= 3:
    → P4 SIEMPRE (OVERRIDE - bloquea todas)
```

---

## 💰 ¿POR QUÉ P4 EN VEZ DE IR A LA POLÍTICA MÁS RESTRICTIVA?

### La Lógica Detrás de P4

**Pregunta correcta**: Si el presupuesto se agota, ¿por qué no simplemente aplicar P1 (la más restrictiva)?

**Respuesta**: P4 **NO es una política de privacidad**, es una **política de control de acceso**:

```
P1, P2, P3 = CONSUME presupuesto (aplica ruido DP)
P4         = DETIENE el consumo (no aplica ruido)
```

### Comparación Visual

```
PRESUPUESTO DISPONIBLE (repeticiones 0-2):
├─ Iteración 0: P1/P2/P3 → Consume presupuesto → budget_allowed=TRUE
├─ Iteración 1: P1/P2/P3 → Consume presupuesto → budget_allowed=TRUE
├─ Iteración 2: P1/P2/P3 → Consume presupuesto → budget_allowed=TRUE

PRESUPUESTO AGOTADO (iteración ≥ 3):
└─ Iteración 3+: P4 → NO CONSUME → budget_allowed=FALSE
                      (epsilon=0, sin ruido, requiere revisión manual)
```

### ¿Qué hace realmente P4?

| Aspecto | P1/P2/P3 | P4 |
|---------|----------|-----|
| **Aplica ruido DP?** | ✅ Sí | ❌ No |
| **epsilon > 0?** | ✅ Sí (3.0, 1.5, 0.5) | ❌ No (0.0) |
| **Consume presupuesto?** | ✅ Sí | ❌ No |
| **Respuesta** | Predicción ruidosa | Predicción original bloqueada |
| **review_required** | false | **true** |
| **Destino** | Cliente directo | Cola de revisión manual |

### La Razón Técnica

En **Differential Privacy**, el presupuesto es un recurso finito:

```
Presupuesto Global = suma de todos los epsilons consumidos
```

**Escenario real**: Una aplicación tiene presupuesto `$\epsilon_{total} = 5.0$` para TODO el mes:

```
Usuario A solicita: /predict + noise_repetitions=10
├─ Iteración 0-2: Consume 3 × 0.5 = 1.5 (P3)
└─ Iteración 3+: P4 (BLOQUEADO, presupuesto del usuario agotado)

Usuario B solicita: /predict
├─ Accede a su presupuesto personal
```

**Si en iteración 3 aplicara P1 (ε=3.0)**, consumiría **MÁS presupuesto**, violando el límite privacidad.

---

### Ejemplo Concreto

```
Presupuesto total asignado al usuario: ε = 1.0

Solicitud /predict con noise_repetitions=5:

iter=0: P3 (ε=0.5) → Total acumulado: 0.5 ✅ Dentro del presupuesto
iter=1: P3 (ε=0.5) → Total acumulado: 1.0 ✅ Justo en el límite
iter=2: P3 (ε=0.5) → Total acumulado: 1.5 ❌ EXCEDERÍA presupuesto
        → ACTIVA P4: NO aplica ruido, retorna original, review_required=true
iter=3: P4 (ε=0.0) → No suma al presupuesto
iter=4: P4 (ε=0.0) → No suma al presupuesto
```

**Veredicto**: P4 es un **"circuit breaker"** (disyuntor) que **previene exceder el presupuesto**, no una política más restrictiva.

---

## 🌍 ¿QUÉ SERÍA UN PRESUPUESTO GLOBAL POR USUARIO/PETICIÓN?

### Definición

Un **presupuesto global** es un **límite total de privacidad** asignado a un usuario/aplicación, no por transacción:

```
Presupuesto LOCAL (actual):
├─ Cada /predict call: 3 repeticiones permitidas
├─ Repetición 4 en adelante: P4 bloquea
└─ Presupuesto se "resetea" con cada nuevo call

Presupuesto GLOBAL (futuro Fase 2):
├─ Usuario "app_123": ε_total = 10.0 para TODO el mes
├─ Request 1: consume 0.5 → Resto: 9.5
├─ Request 2: consume 2.0 → Resto: 7.5
├─ Request N: cuando quede < umbral → P4 bloquea
└─ Presupuesto NO se resetea hasta mes siguiente
```

### Tabla Comparativa: Local vs Global

| Aspecto | Presupuesto LOCAL (Actual) | Presupuesto GLOBAL (Fase 2) |
|---------|---------------------------|---------------------------|
| **Alcance** | Por transacción (call a `/predict`) | Por usuario/app (mes/período) |
| **Reset** | Se resetea con cada nuevo call | Se resetea al cambiar período |
| **Límite** | 3 repeticiones (iteraciones 0-2) | Ej: 10.0 epsilon/mes |
| **Rastreo** | En memoria (temporal) | Base de datos (persistente) |
| **Cuando agota** | Iteración 3 → P4 | Suma acumulada > límite → P4 |
| **Regulación** | **Por solicitud** | **Por usuario a largo plazo** |

### Implementación Conceptual

```python
# PRESUPUESTO LOCAL (ACTUAL - Fase 1)
class PrivacyGovernor:
    def decide(self, mechanism=None):
        # Solo cuenta iteraciones en call actual
        if iteracion >= 3:
            return P4  # Bloquea
        return P1/P2/P3

# PRESUPUESTO GLOBAL (PROPUESTO - Fase 2)
class PrivacyGovernorPhase2:
    def decide(self, user_id, mechanism=None):
        # Consulta BD: ¿cuánto epsilon ha gastado este usuario?
        consumed = db.query(f"SELECT sum(epsilon) FROM log WHERE user_id={user_id} AND month=current_month")
        budget = BUDGET_PER_USER[user_id]  # Ej: 10.0
        
        if consumed >= budget:
            return P4  # Presupuesto GLOBAL agotado
        
        # Evalúa política normal
        return P1/P2/P3  # Consume más presupuesto

    def log_consumption(self, user_id, epsilon_used):
        # Guarda: "Usuario X consumió ε=0.5 en timestamp Y"
        db.insert("epsilon_log", user_id, epsilon_used, timestamp)
```

### Ejemplo Práctico de Presupuesto Global

**Aplicación**: Banco con API de detección de fraude

```
Presupuesto asignado a cliente "MercadoPago":
├─ ε_mes = 50.0 (epsilon total permitido para junio)

Transacciones de junio:
├─ 1000 requests a /predict
├─ Cada uno con DP
├─ Suma total: ε_consumido = 47.3 ✅ Dentro del límite
├─ Requests restantes: presupuesto 50.0 - 47.3 = 2.7 disponible
└─ Request 1001: si requiere ε > 2.7 → P4 bloquea

Julio:
└─ Presupuesto se RESETEA a 50.0 nuevamente
```

### Ventajas del Presupuesto Global

| Ventaja | Beneficio |
|---------|-----------|
| **Control real de privacidad** | Garantiza límite a largo plazo, no solo por transacción |
| **Justo entre usuarios** | Cada cliente sabe exactamente cuánta privacidad consume |
| **Transparencia** | Auditoría: "Cliente X gastó ε=47.3 de 50.0 en junio" |
| **Regulatorio** | GDPR/CCPA: prueba de cumplimiento de privacidad |
| **Negocio** | Posibilidad de vender "paquetes de privacidad" |

### Challenges de Implementar Presupuesto Global

```python
# Desafío 1: Rastreo Persistente
# Solución: Base de datos con log de cada epsilon consumido
db_table = "epsilon_consumption_log"
columns = ["user_id", "timestamp", "epsilon_consumed", "request_id", "transaction_id"]

# Desafío 2: Contabilidad Exacta
# Solución: Usar mecanismos deterministas donde sea posible
# ¿Cuánto epsilon se usó en esta predicción?
# Respuesta: epsilon_usado = epsilon_in de la política aplicada

# Desafío 3: Sincronización Distribuida
# Solución: Cache + DB con transacciones ACID
# Riesgo: dos requests concurrentes podrían exceder presupuesto
# Mitigación: Usar LOCK en table de usuarios

# Desafío 4: Reset de Período
# Solución: Job diario que reseta contadores a inicio de mes
```

---

## 🔄 FLUJO CON PRESUPUESTO GLOBAL (PROPUESTA FASE 2)

```
Usuario: "BankApp"
Presupuesto mes: ε = 100.0

DÍA 1:
├─ Request 1: /predict + noise_repetitions=5
│  ├─ iter 0: P3 (ε=0.5) → Acumulado: 0.5
│  ├─ iter 1: P3 (ε=0.5) → Acumulado: 1.0
│  ├─ iter 2: P3 (ε=0.5) → Acumulado: 1.5
│  ├─ iter 3: Verifica DB: "1.5 < 100.0" ✅
│  │           → P3 (ε=0.5) → Acumulado: 2.0
│  └─ iter 4: Verifica DB: "2.0 < 100.0" ✅
│             → P3 (ε=0.5) → Acumulado: 2.5
├─ Guarda: epsilon_log["BankApp", 2.5, timestamp]
└─ Presupuesto restante: 100.0 - 2.5 = 97.5

DÍA 30:
├─ Acumulado consumido: 97.8
├─ Request N: /predict
│  └─ Verifica DB: "97.8 < 100.0" ✅ Aún hay 2.2
│     → P3 (ε=0.5) ✅ Aprobado
└─ Presupuesto restante: 100.0 - 98.3 = 1.7

DÍA 30 (solicitud posterior):
├─ Acumulado consumido: 99.1
├─ Request M: /predict
│  └─ Intenta P3 (ε=0.5)
│     Verifica DB: "99.1 + 0.5 = 99.6 < 100.0" ✅ Marginal
│     → Aprobado, presupuesto: 0.4 restante
└─ Cliente: "Solo 0.4 epsilon disponible este mes"

DÍA 30 (AGOTADO):
├─ Acumulado consumido: 99.9
├─ Request Z: /predict
│  └─ Intenta P3 (ε=0.5)
│     Verifica DB: "99.9 + 0.5 = 100.4 > 100.0" ❌
│     → ACTIVA P4: review_required=true
│     → Cliente recibe: "Presupuesto mensual agotado"
└─ Presupuesto: 0 disponible

JULIO 1:
└─ Sistema resetea contador: presupuesto = 100.0 nuevamente
```

---

## 📊 RESUMEN: LOCAL vs GLOBAL

```
FASE 1 (ACTUAL - Presupuesto LOCAL):
┌─────────────────────────────────────┐
│ /predict request individual         │
├─────────────────────────────────────┤
│ iter 0-2: P1/P2/P3 ✅ (permite)    │
│ iter 3+:  P4 ❌ (bloquea)           │
└─────────────────────────────────────┘
Presupuesto se "resetea" con cada nuevo call

FASE 2 (PROPUESTO - Presupuesto GLOBAL):
┌─────────────────────────────────────┐
│  Año / Mes / Semana (período)       │
├─────────────────────────────────────┤
│ User: "app_A"                       │
│ Presupuesto: ε = 100.0              │
│ Consumido: ε = 73.5                 │
│ Disponible: ε = 26.5 ◀─ Restante    │
│                                     │
│ /predict request 1: P3 (ε=0.5) ✅   │
│ /predict request 2: P3 (ε=0.5) ✅   │
│ ...                                 │
│ /predict request N: P4 (cuando      │
│                    consumido>límite) │
└─────────────────────────────────────┘
Presupuesto persiste y se acumula
```

---



```json
{
  "noise_repetitions": 2,
  "original": {
    "is_fraud_original": 0,
    "fraud_probability_original": 0.03096818311665202,
    "confidence_score_original": 0.9690318168833479
  },
```

### Explicación:
- **`noise_repetitions: 2`** → Se solicitaron 2 repeticiones de ruido DP. Cada una aplica ruido diferente para ver variabilidad.
- **`is_fraud_original: 0`** → El modelo original predice: **LEGÍTIMA** (0 = no es fraude)
- **`fraud_probability_original: 0.0310`** → Probabilidad original de fraude muy baja (3.1%), confirma que es legítima
- **`confidence_score_original: 0.9690`** → Confianza del modelo muy alta (96.9%), está muy seguro de su predicción

---

```json
  "results": [
    {
      "index": 1,
      "with_privacy": {
        "is_fraud": 0,
        "fraud_probability": 0.0,
        "confidence_score": 0.7813060945477232
      },
      "message": "Transacción legítima",
      "privacy_info": {
        "mode": "governed",
        "mechanism": "laplace",
        "epsilon": 0.5,
        "delta": 1e-6,
        "budget_allowed": true,
        "applied_to_fields": ["fraud_probability", "confidence_score"],
        "policy": {
          "modo": "governed",
          "politica": "P3",
          "epsilon_in": 0.5,
          "epsilon_out": 0.8,
          "distancia_umbral": 0.469031816883348
        },
        "review_required": false
      }
    },
```

### Explicación - REPETICIÓN 1:
- **`index: 1`** → Primera repetición de ruido
- **`is_fraud: 0`** → Con privacidad diferencial sigue siendo legítima
- **`fraud_probability: 0.0`** → **Ruido aplicado**: la probabilidad bajó de 0.0310 a 0.0 (fue a cero)
- **`confidence_score: 0.7813`** → **Ruido aplicado**: la confianza bajó de 0.9690 a 0.7813 (pérdida de ~19% de confianza)

- **`mode: "governed"`** → Usa la política de governance (no raw, no legacy)
- **`mechanism: "laplace"`** → Ruido añadido con distribución Laplace
- **`epsilon: 0.5`** → Parámetro de privacidad bastante restrictivo (más privacidad = menos precisión)
- **`delta: 1e-6`** → Probabilidad de fallo de privacidad = 0.000001 (muy pequeña)
- **`politica: "P3"`** → Política 3 fue seleccionada por el governance
- **`budget_allowed: true`** → El sistema **tiene presupuesto disponible** para consumir privacidad. Se vuelve `false` en la política P4 (iteración ≥ 3), que se activa cuando el presupuesto de consultas se agota; en ese caso el ruido NO se aplica y la respuesta queda bloqueada para revisión manual
- **`epsilon_in: 0.5, epsilon_out: 0.8`** → Parámetros de calibración del mecanismo DP para dos fases del pipeline. `epsilon_in` es el epsilon con que se aplica ruido; `epsilon_out` es el epsilon máximo permitido en la salida. **No son presupuesto acumulado ni restante**: ambos son constantes de la política P3
- **`distancia_umbral: 0.469`** → Distancia de la probabilidad de fraude al umbral de decisión (0.5). Se calcula como `|fraud_probability - 0.5|` = `|0.031 - 0.5|` = **0.469**. Determina qué política se aplica:
  - `< 0.02` → **P1** (transacción en zona gris, muy cerca del umbral → epsilon=3.0, ruido máximo para máxima protección)
  - `≥ 0.02` + baja confianza → **P2** (epsilon=1.5)
  - `≥ 0.02` + confianza normal → **P3** (epsilon=0.5, caso actual)
  - En este caso 0.469 >> 0.02 y confianza=0.969 > 0.6, la transacción está **muy lejos del umbral** (claramente legítima), lo que activa P3 con menor epsilon

---

```json
    {
      "index": 2,
      "with_privacy": {
        "is_fraud": 0,
        "fraud_probability": 0.0,
        "confidence_score": 0.0
      },
      "message": "Transacción legítima",
      "privacy_info": {
        "mode": "governed",
        "mechanism": "laplace",
        "epsilon": 0.5,
        "delta": 1e-6,
        "budget_allowed": true,
        "applied_to_fields": ["fraud_probability", "confidence_score"],
        "policy": {
          "modo": "governed",
          "politica": "P3",
          "epsilon_in": 0.5,
          "epsilon_out": 0.8,
          "distancia_umbral": 0.469031816883348
        },
        "review_required": false
      }
    }
```

### Explicación - REPETICIÓN 2:
- **`index: 2`** → Segunda repetición (ruido diferente)
- **`fraud_probability: 0.0`** → Nuevamente 0.0 (igual a repetición 1)
- **`confidence_score: 0.0`** → **Ruido mayor**: bajó a 0 (comparado con 0.7813 en repetición 1)
- **`budget_allowed: true`** → Presupuesto disponible (iteración=1, aún no llega a P4 que se activa en iteración ≥ 3)
- **`distancia_umbral: 0.469`** → Idéntica a la repetición 1: el modelo base es determinista, por lo que la distancia al umbral no cambia entre repeticiones. Solo el ruido Laplace aplicado es diferente
- **Diferencia**: El ruido Laplace es aleatorio, por eso produce valores diferentes en cada repetición
- **Ambas siguen prediciendo**: `is_fraud: 0` (legítima), pero con confianzas muy distintas

---

```json
  "metrics": {
    "fraud_probability": {
      "mae": 0.03096818311665202,
      "rmse": 0.03096818311665202,
      "rmse_mae_ratio": 1.0,
      "noise_std": 0.0,
      "n_samples": 2
    },
```

### Explicación - MÉTRICAS DE RUIDO EN `fraud_probability`:
- **`mae: 0.0310`** (Mean Absolute Error) → Error promedio del ruido = 0.031
  - El valor original era 0.0310, el promedio con ruido fue 0.0
  - Error = |0.0310 - 0.0| = 0.0310
- **`rmse: 0.0310`** (Root Mean Square Error) → Raíz del error cuadrático medio
  - Mismo que MAE porque ambas repeticiones produjeron 0.0
- **`rmse_mae_ratio: 1.0`** → RMSE/MAE = 1, indica distribución uniforme del ruido (sin picos grandes)
- **`noise_std: 0.0`** → Desviación estándar del ruido = 0
  - Ambas repeticiones produjeron exactamente el mismo valor (0.0)
  - **Esto es inusual**: el ruido Laplace debería generar valores diferentes
- **`n_samples: 2`** → Se usaron 2 repeticiones

---

```json
    "confidence_score": {
      "mae": 0.5783787696094864,
      "rmse": 0.6979482820949825,
      "rmse_mae_ratio": 1.206732194831819,
      "noise_std": 0.3906530472738616,
      "n_samples": 2
    },
```

### Explicación - MÉTRICAS DE RUIDO EN `confidence_score`:
- **`mae: 0.5784`** → Error promedio absoluto = 0.5784
  - Promedio entre: |0.9690 - 0.7813| = 0.1877 y |0.9690 - 0.0| = 0.9690
  - (0.1877 + 0.9690) / 2 = 0.5784
  - **Degradación significativa** de la confianza
- **`rmse: 0.6979`** → Error cuadrático = 0.6979
  - Mayor que MAE, indica que una repetición tuvo error mucho más grande (0.9690)
- **`rmse_mae_ratio: 1.207`** → RMSE > MAE, hay una "punta" de error (una repetición con ruido muy grande)
- **`noise_std: 0.3906`** → Desviación estándar del ruido = 0.3906
  - Las dos repeticiones produjeron valores muy diferentes (0.7813 vs 0.0)
  - Esto SÍ muestra variabilidad esperada del ruido Laplace

---

```json
    "flip_rate": 0.0,
    "flip_count": 0
  }
}
```

### Explicación - TASA DE INVERSIÓN:
- **`flip_count: 0`** → Número de veces que cambió la predicción = 0
  - Ambas repeticiones mantuvieron `is_fraud: 0` (legítima)
  - El ruido NO fue lo suficientemente grande para cambiar la clasificación
- **`flip_rate: 0.0`** → 0 de 2 repeticiones = 0% (0/2)
  - **Baja vulnerabilidad**: el ruido no invierte la decisión
  - La predicción es robusta incluso con privacidad diferencial

---

---

## 📈 RESULTADO 2: ENDPOINT `/evaluate`

```json
{
  "n_transactions": 1,
  "n_errors": 0,
  "errors": [],
  "mode": "governed",
  "threshold": 0.5,
```

### Explicación:
- **`n_transactions: 1`** → Solo se evaluó 1 transacción etiquetada
- **`n_errors: 0`** → Sin errores en la evaluación
- **`mode: "governed"`** → Se usó política de governance
- **`threshold: 0.5`** → Umbral: si `fraud_probability > 0.5` → fraude, else → legítima

---

```json
  "noise_metrics": {
    "fraud_probability": {
      "mae": 0.9936584700616399,
      "rmse": 0.9936584700616399,
      "rmse_mae_ratio": null,
      "noise_std": null,
      "n_samples": 1
    }
  },
```

### Explicación - COMPARACIÓN DE RUIDO:
- **⚠️ DIFERENCIA CON `/predict`**: `/evaluate` solo hace UNA predicción (sin repeticiones)
- **`n_samples: 1`** → Solo hay 1 predicción, por eso no hay desviación estándar ni ratio
- **`mae: 0.9937`** → Error absoluto del ruido = 0.9937
  - Original: 0.03 → Con ruido: no está especificado en este JSON
  - Pero el valor es casi 1, indica que el ruido aplicado fue grande
- **`rmse_mae_ratio: null`** → No se puede calcular con solo 1 muestra

---

```json
  "classification_original": {
    "confusion_matrix": {
      "tp": 0,
      "fp": 0,
      "tn": 0,
      "fn": 1
    },
    "accuracy": 0.0,
    "fnr": 1.0,
    "f1": 0.0,
    "informedness": -1.0,
    "markedness": -1.0,
    "mcc": 1.0,
    "n_samples": 1
  },
```

### Explicación - CLASIFICACIÓN DEL MODELO ORIGINAL (SIN RUIDO):
- **Matriz de Confusión**:
  - `tp: 0` → True Positives = 0 (NO detectó fraudes que sí eran)
  - `fp: 0` → False Positives = 0 (correctamente no marcó como fraude)
  - `tn: 0` → True Negatives = 0 (correctamente identificó legítimas)
  - `fn: 1` → **False Negatives = 1** ⚠️ **FALLÓ**: La transacción ERA fraude pero el modelo dijo legítima
  
- **Etiqueta real**: 1 (ES FRAUDE)
- **Predicción original**: 0 (modelo dice legítima) → **ERROR CRÍTICO**

- **`accuracy: 0.0`** → 0 de 1 transacciones bien clasificadas = 0% (¡FALLÓ!)
- **`fnr: 1.0`** → Tasa de Falsos Negativos = 100% (detectó 0 de 1 fraude real)
  - **Esto es grave**: el modelo no detectó un fraude real
- **`f1: 0.0`** → Score F1 = 0 (balance entre precisión y recall es malo)
- **`informedness: -1.0`** → Métrica de Powers = -1 (peor caso posible)
- **`mcc: 1.0`** → ⚠️ **RESULTADO ENGAÑOSO**. El código calcula MCC = `sqrt(informedness × markedness) = sqrt(-1.0 × -1.0) = sqrt(1.0) = 1.0`. El producto de dos componentes en su peor valor (-1 × -1) da un MCC positivo, lo que **no indica un buen modelo**. Es un artefacto matemático del dataset con una sola clase predicha.
- **VEREDICTO**: El modelo original **falló completamente**, no detectó un fraude real

---

```json
  "classification_with_dp": {
    "confusion_matrix": {
      "tp": 1,
      "fp": 0,
      "tn": 0,
      "fn": 0
    },
    "accuracy": 1.0,
    "fnr": 0.0,
    "f1": 1.0,
    "informedness": 0.0,
    "markedness": 0.0,
    "mcc": 0.0,
    "n_samples": 1
  },
```

### Explicación - CLASIFICACIÓN CON PRIVACIDAD DIFERENCIAL (CON RUIDO):
- **Matriz de Confusión**:
  - `tp: 1` → **Ahora detectó el fraude correctamente** ✅
  - `fp: 0` → Sin falsos positivos
  - `tn: 0` → (no hay legítimas en este dataset)
  - `fn: 0` → **Sin falsos negativos** ✅

- **`accuracy: 1.0`** → **100%** de precisión (¡1 de 1 correcta!)
- **`fnr: 0.0`** → 0% de falsos negativos (detectó todos los fraudes)
- **`f1: 1.0`** → F1 perfecto (1.0)
- **VEREDICTO**: El ruido DP **MEJORÓ la predicción** (pasó de fallar a acertar)
- ⚠️ **Nota**: Con solo 1 transacción, estas métricas no son estadísticamente significativas

---

```json
  "utility_retention": {
    "accuracy": null,
    "f1": null,
    "informedness": -0.0,
    "mcc": 0.0
  },
```

### Explicación - RETENCIÓN DE UTILIDAD (TRADE-OFF PRIVACIDAD-UTILIDAD):
- **`accuracy: null`** → No se puede calcular (sería 1.0/0.0 = indefinido)
- **`f1: null`** → No se puede calcular
- **`informedness: -0.0`** → Retención = `dp_informedness / orig_informedness` = `0.0 / (-1.0)` = **-0.0** (cero negativo en punto flotante). El denominador es negativo (-1.0), resultado técnicamente 0 pero con signo negativo
- **`mcc: 0.0`** → Retención = `dp_mcc / orig_mcc` = `0.0 / 1.0` = **0.0**. El mcc con DP es 0 (partió de valores cero), reteniendo el 0% del MCC original
- **INTERPRETACIÓN**: accuracy y f1 dan `null` porque el modelo original tuvo 0% en ambas (no se puede calcular retención con denominador 0)

---

```json
  "privacy_utility": {
    "utility_original": 0.0,
    "utility_with_dp": 1.0,
    "utility_retention": 0.0,
    "risk_noise_correlation": 0.0,
    "n_samples": 1
  }
}
```

### Explicación - ANÁLISIS PRIVACIDAD-UTILIDAD:
- **`utility_original: 0.0`** → Utilidad base (F1 del modelo original) = 0.0 — el evaluador usa F1 como proxy de utilidad principal
- **`utility_with_dp: 1.0`** → Utilidad con DP (F1 con ruido aplicado) = 1.0
- **`utility_retention: 0.0`** → El código define: cuando `utility_original == 0.0` y `utility_with_dp != 0.0`, la función retorna **0.0** (caso especial: no hay utilidad base de referencia, por lo que la retención es indefinida y se reporta como 0)
  - Interpretación: **No significa pérdida de utilidad**. Significa que no hay base para calcular retención
- **`risk_noise_correlation: 0.0`** → Correlación entre riesgo y ruido = 0
  - No hay relación entre el nivel de ruido y la probabilidad de fraude
- **`n_samples: 1`** → Solo 1 transacción

---

---

## 🎯 RESUMEN COMPARATIVO

| Aspecto | `/predict` | `/evaluate` |
|---------|-----------|-----------|
| **Propósito** | Ver variabilidad del ruido DP en 1 transacción | Medir performance en dataset etiquetado |
| **Repeticiones** | 2 (múltiples ruidos) | 1 (sin repeticiones) |
| **Métrica principal** | Noise metrics (MAE, RMSE, flip_rate) | Classification metrics (Accuracy, F1, FNR) |
| **Output** | Lista de predicciones con ruido | Reporte de métricas de clasificación |
| **Uso** | Análisis de robustez y variabilidad | Validación de accuracy y trade-off privacidad-utilidad |

### En este caso:
- **`/predict`**: Mostró que el ruido DP es **predecible** y **no invierte la clasificación** (flip_rate = 0%)
- **`/evaluate`**: Mostró que el ruido DP **accidentalmente mejoró** la predicción (de error total a acierto total)
  - ⚠️ **Nota**: Con n=1, esto no es estadísticamente válido

