"""
API de detección de fraude con FastAPI
Arquitectura simple para integración con Privacy Engineering
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import uvicorn
from model import FraudModel
import os

# Inicializar FastAPI
app = FastAPI(
    title="API de Detección de Fraude",
    description="API educativa para detección de fraude con modelo de ML ligero",
    version="1.0.0"
)

# Cargar modelo al iniciar
try:
    fraud_model = FraudModel()
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    print("  Ejecuta primero: python train_model.py")
    fraud_model = None

# Modelos de datos
class Transaction(BaseModel):
    """
    Modelo de entrada: características de una transacción
    PUNTO DE INTEGRACIÓN PRIVACY: Aquí se pueden aplicar técnicas de
    anonimización, differential privacy, etc. antes de procesar
    """
    Time: float = Field(..., description="Tiempo desde primera transacción")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    Amount: float = Field(..., description="Monto de la transacción")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.35,
                "V2": -0.07,
                "V3": 2.53,
                "V4": 1.38,
                "V5": -0.33,
                "V6": 0.46,
                "V7": 0.24,
                "V8": 0.10,
                "V9": 0.36,
                "V10": 0.09,
                "V11": -0.55,
                "V12": -0.62,
                "V13": -0.99,
                "V14": -0.31,
                "Amount": 149.62
            }
        }

class PredictionResponse(BaseModel):
    """
    Modelo de salida: resultado de la predicción
    PUNTO DE INTEGRACIÓN PRIVACY: Aquí se pueden aplicar técnicas de
    agregación, ruido, k-anonimato en respuestas, etc.
    """
    is_fraud: int = Field(..., description="0=legítimo, 1=fraude")
    fraud_probability: float = Field(..., description="Probabilidad de fraude [0-1]")
    confidence_score: float = Field(..., description="Confianza en la predicción")
    message: str = Field(..., description="Mensaje interpretativo")

# Endpoints
@app.get("/")
def root():
    """Endpoint raíz con información de la API"""
    return {
        "service": "API de Detección de Fraude",
        "version": "1.0.0",
        "status": "running" if fraud_model else "model not loaded",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "features": "/features (GET)"
        }
    }

@app.get("/health")
def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy" if fraud_model else "model not loaded",
        "model_loaded": fraud_model is not None
    }

@app.get("/features")
def get_features():
    """Obtener lista de características requeridas"""
    if not fraud_model:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return {
        "features": fraud_model.get_feature_names(),
        "total": len(fraud_model.get_feature_names())
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """
    Predice si una transacción es fraudulenta

    ARQUITECTURA PRIVACY ENGINEERING:
    1. PRE-PROCESAMIENTO (antes de esta función):
       - Validación de entrada
       - Anonimización de datos sensibles
       - Aplicación de differential privacy

    2. PROCESAMIENTO (esta función):
       - Predicción del modelo

    3. POST-PROCESAMIENTO (antes de retornar):
       - Agregación de resultados
       - Adición de ruido calibrado
       - Limitación de información expuesta
    """
    if not fraud_model:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta train_model.py primero"
        )

    try:
        # Convertir a diccionario
        transaction_dict = transaction.model_dump()

        # ===== PUNTO DE INTEGRACIÓN: PRE-PROCESAMIENTO PRIVACY =====
        # Aquí se pueden aplicar transformaciones de privacidad
        # Ejemplo: differential_privacy_layer(transaction_dict)

        # Realizar predicción
        result = fraud_model.predict(transaction_dict)

        # ===== PUNTO DE INTEGRACIÓN: POST-PROCESAMIENTO PRIVACY =====
        # Aquí se pueden aplicar transformaciones de privacidad a la salida
        # Ejemplo: add_calibrated_noise(result)

        # Preparar respuesta
        message = "FRAUDE DETECTADO" if result['is_fraud'] == 1 else "Transacción legítima"

        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            confidence_score=result['confidence_score'],
            message=message
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# Ejecutar servidor
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  API de Detección de Fraude - Modelo Python/ML")
    print("="*60)
    print("\nPuntos de integración para Privacy Engineering:")
    print("  1. PRE-procesamiento: Línea ~145 en /predict")
    print("  2. POST-procesamiento: Línea ~151 en /predict")
    print("\nIniciando servidor en http://localhost:8000")
    print("Documentación interactiva: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

