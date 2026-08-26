"""
API de Privacy Engineering que envuelve modelo-python
Consume predicciones del modelo y aplica privacidad diferencial
Arquitectura modular: cambios en privacy_layer, client o config sin afectar este archivo
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import logging
#from __future__ import annotations

# Importar componentes modulares
from client import model_client
from privacy_layer import dp_layer
from input_privacy_layer import InputPrivacyLayer, FEATURE_BOUNDS
from governor import GovernanceDecision, governor
from privacy_governor_simple import gobernar_politica
from evaluator import batch_evaluator
from metrics import noise_summary
from config import PRIVACY_API_HOST, PRIVACY_API_PORT, LOG_LEVEL, EVALUATION_MAX_TRANSACTIONS, EVALUATION_THRESHOLD, EVALUATION_MODE

# Configurar logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Singleton de la capa de privacidad en entrada
input_dp_layer = InputPrivacyLayer()

# ===== INICIALIZAR API =====
app = FastAPI(
    title="API de Detección de Fraude con Privacy Engineering",
    description="API que envuelve modelo-python con privacidad diferencial (diffprivlib)",
    version="2.0.0"
)

# ===== MODELOS DE DATOS =====
class Transaction(BaseModel):
    """Modelo de entrada: características de una transacción"""
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
    noise_repetitions: int = Field(1, ge=1, le=20, description="Cantidad de veces que se aplica ruido DP (una respuesta por repetición)")
    epsilon_in: Optional[float] = Field(None, ge=0.0, description="LAB ONLY: override del epsilon_in del input shield para barridos.")

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

def _prediction_values(response: Dict[str, Any]) -> Dict[str, Any]:
   """Extraer únicamente el contrato común de una predicción."""
   return {
       "is_fraud": int(response.get("is_fraud")),
       "fraud_probability": float(response.get("fraud_probability")),
       "confidence_score": float(response.get("confidence_score")),
   }


# ===== LIFECYCLE EVENTS =====
@app.on_event("startup")
def startup_event():
    """Verificar disponibilidad del modelo-python al iniciar"""
    logger.info("\n" + "="*70)
    logger.info("  API de Privacy Engineering - Startup")
    logger.info("="*70)

    if model_client.health():
        logger.info("Modelo-python: Disponible y sano")
    else:
        logger.warning("⚠ Modelo-python: No disponible (verifica que está corriendo en puerto 8000)")


# ===== ENDPOINTS =====
@app.get("/")
def root():
    """Información de la API"""
    return {
        "service": "API de Privacy Engineering (Fraude Detection v2.0)",
        "description": "Envuelve modelo-python con privacidad diferencial",
        "mode": "privacy-aware-predictions",
        "endpoints": {
            "root": "/ (GET)",
            "health": "/health (GET)",
            "privacy-info": "/privacy-info (GET)",
            "predict": "/predict (POST)",
            "predict-laplace": "/predict-laplace (POST)",
            "predict-gaussian": "/predict-gaussian (POST)",
            "evaluate": "/evaluate (POST)",
        }
    }


@app.get("/health")
def health_check():
    """Verificar salud de la API y disponibilidad del modelo"""
    model_healthy = model_client.health()
    return {
        "status": "healthy" if model_healthy else "degraded",
        "model_available": model_healthy,
        "privacy_layer_ready": True,
    }


@app.get("/privacy-info")
def privacy_info():
    """Obtener información sobre configuración de privacidad diferencial"""
    return {
        "privacy_engineering": "Differential Privacy (diffprivlib)",
        "configuration": dp_layer.get_info(),
        "description": {
            "epsilon": "Parámetro de privacidad. Menor = más privacidad, menos precisión",
            "delta": "Probabilidad de fallo de privacidad (típicamente muy pequeño)",
            "mechanism": "Laplace o Gaussian para añadir ruido calibrado",
        }
    }


@app.post("/predict")
def predict_with_dp(
    transaction: Transaction,
    mode: str = "governed",
    compare_modes: bool = False,
    mechanism: Optional[str] = None,
):
    """
    Predicción con privacidad diferencial aplicada N veces (noise_repetitions).

    FLUJO:
    1. Llamar al modelo UNA sola vez (determinista)
    2. Por cada repetición: governor decide → privacy_layer aplica ruido
    3. Retornar respuesta estructurada por capa conceptual

    El governor se invoca una vez por repetición para que, cuando se integre
    el noise_selector, pueda devolver parámetros distintos en cada iteración.
    """
    try:
        # Excluir noise_repetitions antes de enviar al modelo-python
        transaction_dict = transaction.model_dump(exclude={"noise_repetitions", "epsilon_in"})
        repetitions = transaction.noise_repetitions
        selected_mode = mode.lower()
        if selected_mode not in {"raw", "legacy", "governed"}:
            raise ValueError("mode debe ser raw, legacy o governed")

        # Input DP: el vector de ruido de entrada se sortea UNA vez por request
        # (aquí, antes del bucle de repeticiones) → noise_resampled_per = "request"
        input_decision = governor.decide(mechanism=mechanism)
        epsilon_in_input = (
            transaction.epsilon_in
            if transaction.epsilon_in is not None
            else (getattr(input_decision, "epsilon_in", None) or input_decision.epsilon)
        )
        perturbed_dict, input_detail = input_dp_layer.apply_with_detail(transaction_dict, epsilon_in=epsilon_in_input)
        _n_perturbed = len([f for f in FEATURE_BOUNDS if f in transaction_dict])

        # 1. modelo-python: predice una sola vez (la predicción es determinista)
        model_prediction = model_client.predict(perturbed_dict)

        samples: list = []
        mode_comparison_legacy: list = []
        mode_comparison_governed: list = []
        noise_pairs_prob: list = []
        noise_pairs_conf: list = []
        flip_count: int = 0

        # ground_truth y metadatos de política se capturan de la iteración 0
        ground_truth: Dict[str, Any] = {}
        first_decision = None
        first_policy: Dict[str, Any] = {}
        first_applied_to_fields: list = []

        for i in range(repetitions):
            raw_result = model_prediction.copy()
            legacy_decision = governor.decide(mechanism=mechanism)
            governed_policy = gobernar_politica(
                fraud_probability=float(raw_result["fraud_probability"]),
                confidence_score=float(raw_result["confidence_score"]),
                iteracion=i,
            )
            governed_decision = GovernanceDecision(
                mechanism=mechanism or governor._MECHANISM,
                epsilon=governed_policy["epsilon_in"],
                delta=governor._DELTA,
                budget_allowed=governed_policy["politica"] != "P4",
                epsilon_in=governed_policy["epsilon_in"],
                epsilon_out=governed_policy["epsilon_out"],
            )

            legacy_result = None
            governed_result = None
            if compare_modes or selected_mode == "legacy":
                legacy_result = dp_layer.apply_dp_to_response(
                    model_prediction.copy(), decision=legacy_decision
                )
            if compare_modes or selected_mode == "governed":
                governed_result = dp_layer.apply_dp_to_response(
                    model_prediction.copy(), decision=governed_decision
                )
            result_with_dp = {
                "raw": raw_result,
                "legacy": legacy_result,
                "governed": governed_result,
            }[selected_mode]
            selected_decision = {
                "legacy": legacy_decision,
                "governed": governed_decision,
            }.get(selected_mode)

            orig_prob = float(model_prediction["fraud_probability"])
            orig_conf = float(model_prediction["confidence_score"])
            orig_is_fraud = int(model_prediction["is_fraud"])
            noisy_prob = float(result_with_dp.get("fraud_probability"))
            noisy_conf = float(result_with_dp.get("confidence_score"))
            # Derive is_fraud from the noisy probability (the DP-protected decision)
            noisy_is_fraud = 1 if noisy_prob > 0.5 else 0

            if i == 0:
                ground_truth = {
                    "is_fraud": orig_is_fraud,
                    "fraud_probability": orig_prob,
                    "confidence_score": orig_conf,
                    # laboratory_only: derivado del score VERDADERO, no es metadato de privacidad
                    "distancia_umbral": governed_policy["distancia_umbral"],
                }
                first_decision = selected_decision
                first_policy = governed_policy
                first_applied_to_fields = (
                    []
                    if selected_mode == "raw" or governed_policy["politica"] == "P4"
                    else ["fraud_probability", "confidence_score"]
                )

            if selected_mode != "raw":
                noise_pairs_prob.append((orig_prob, noisy_prob))
                noise_pairs_conf.append((orig_conf, noisy_conf))
                if noisy_is_fraud != orig_is_fraud:
                    flip_count += 1

            samples.append({
                "index": i + 1,
                "is_fraud": noisy_is_fraud,
                "fraud_probability": noisy_prob,
                "confidence_score": noisy_conf,
                "message": result_with_dp.get("message", ""),
            })

            if compare_modes:
                mode_comparison_legacy.append({"index": i + 1, **_prediction_values(legacy_result)})
                mode_comparison_governed.append({"index": i + 1, **_prediction_values(governed_result)})

        # epsilon_out del primer iter (decisión nominal, antes de posible budget exhaustion)
        epsilon_out_per_sample = (
            first_decision.epsilon_out
            if first_decision is not None and first_decision.epsilon_out is not None
            else 0.0
        )

        response: Dict[str, Any] = {
            "request_context": {
                "endpoint_mode": selected_mode,
                "compare_modes": compare_modes,
                "noise_repetitions": repetitions,
                "feature_count": _n_perturbed,
            },
            "ground_truth": ground_truth,
            "input_shield": {
                "mechanism": input_dp_layer.mechanism,
                "epsilon_in_total": epsilon_in_input,
                "epsilon_per_feature": epsilon_in_input / _n_perturbed if _n_perturbed > 0 else 0.0,
                # ruido de entrada sorteado UNA vez por request (antes del loop de repeticiones)
                "noise_resampled_per": "request",
                "perturbation_detail": input_detail if EVALUATION_MODE else None,
            },
            "output_shield": {
                "mechanism": first_decision.mechanism if first_decision is not None else None,
                "epsilon_out": epsilon_out_per_sample,
                # ruido de salida sorteado una vez por repetición, dentro del loop
                "noise_resampled_per": "repetition",
                "applied_to_fields": first_applied_to_fields,
                "samples": samples,
            },
            "privacy_accounting": {
                "epsilon_in_total": epsilon_in_input,
                "epsilon_out_per_sample": epsilon_out_per_sample,
                "noise_repetitions": repetitions,
                # composición secuencial básica (Dwork & Roth 2014, Theorem 3.14)
                "epsilon_out_effective_sequential_basic": repetitions * epsilon_out_per_sample,
                "delta": first_decision.delta if first_decision is not None else 0.0,
                "budget_allowed": first_decision.budget_allowed if first_decision is not None else True,
                # distancia_umbral excluida: pertenece a ground_truth, no a accounting
                "policy": {k: v for k, v in first_policy.items() if k != "distancia_umbral"},
                "review_required": first_policy.get("politica") == "P4",
            },
        }

        if compare_modes:
            response["mode_comparison"] = {
                "legacy": mode_comparison_legacy,
                "governed": mode_comparison_governed,
            }

        if noise_pairs_prob:
            n_reps = len(noise_pairs_prob)
            prob_sum = noise_summary(noise_pairs_prob)
            conf_sum = noise_summary(noise_pairs_conf)
            conf_std = conf_sum["noise_std"]
            response["output_shield_metrics"] = {
                "_note": "Caracteriza la dispersion del ruido de SALIDA sobre p'. NO mide privacidad de la entrada.",
                "fraud_probability": {
                    "mae_vs_ground_truth": prob_sum["mae"],
                    "rmse_vs_ground_truth": prob_sum["rmse"],
                    "rmse_mae_ratio": prob_sum["rmse_mae_ratio"],
                    "sample_std": prob_sum["noise_std"],
                    "n_samples": n_reps,
                },
                "confidence_score": {
                    "mae_vs_ground_truth": conf_sum["mae"],
                    "rmse_vs_ground_truth": conf_sum["rmse"],
                    "rmse_mae_ratio": conf_sum["rmse_mae_ratio"],
                    "sample_std": conf_std,
                    "n_samples": n_reps,
                    "clipping_suspected": bool(
                        conf_std is not None and conf_std == 0.0 and conf_sum["mae"] > 0
                    ),
                },
                "flip_rate": flip_count / n_reps,
                "flip_count": flip_count,
            }
        else:
            response["output_shield_metrics"] = None

        logger.info(f"Predicción completada: {repetitions} repeticion(es) de ruido DP")
        return response

    except ConnectionError as e:
        logger.error(f"Error de conexión: {e}")
        raise HTTPException(
            status_code=503,
            detail="Modelo-python no disponible. ¿Está ejecutándose?"
        )

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/predict-laplace")
def predict_laplace(transaction: Transaction):
    """Alias: fuerza Laplace sin modificar estado global."""
    return predict_with_dp(transaction, mechanism="laplace")


@app.post("/predict-gaussian")
def predict_gaussian(transaction: Transaction):
    """Alias: fuerza Gaussian sin modificar estado global."""
    return predict_with_dp(transaction, mechanism="gaussian")


# ===== EVALUATION ENDPOINT =====

class LabeledTransaction(BaseModel):
    """Single transaction with ground-truth label for batch evaluation."""
    Time: float
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
    Amount: float
    label: int = Field(..., ge=0, le=1, description="Ground-truth class: 0=legitimate, 1=fraud")


class EvaluationRequest(BaseModel):
    """Request body for POST /evaluate."""
    transactions: List[LabeledTransaction] = Field(
        ..., min_length=1, description="Labeled transactions for evaluation"
    )
    mode: str = Field("governed", description="DP mode: raw | legacy | governed")
    threshold: float = Field(
        EVALUATION_THRESHOLD, ge=0.0, le=1.0,
        description="Probability threshold to binarise predictions"
    )
    mechanism: Optional[str] = Field(None, description="Override DP mechanism")


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    """
    Batch evaluation: run all privacy-utility metrics over a labeled dataset.

    Returns noise metrics (MAE, RMSE), classification metrics (Accuracy, FNR,
    F1, Informedness, MCC), utility retention per metric, and risk-noise
    correlation — for both the original model and the DP-protected output.
    """
    if len(request.transactions) > EVALUATION_MAX_TRANSACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many transactions: max {EVALUATION_MAX_TRANSACTIONS}, "
                   f"got {len(request.transactions)}",
        )

    labeled = [t.model_dump() for t in request.transactions]

    try:
        report = batch_evaluator.evaluate(
            labeled_transactions=labeled,
            mode=request.mode,
            threshold=request.threshold,
            mechanism=request.mechanism,
        )
        return report

    except ConnectionError as e:
        logger.error(f"Error de conexión en evaluación: {e}")
        raise HTTPException(
            status_code=503,
            detail="Modelo-python no disponible. ¿Está ejecutándose?"
        )

    except (ValueError, RuntimeError) as e:
        logger.error(f"Error de evaluación: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error inesperado en evaluación: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ===== MAIN =====
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  API de Privacy Engineering - Detección de Fraude v2.0")
    print("="*70)
    print(f"\n Consumiendo modelo desde: {model_client.base_url}")
    print(f"  Privacidad Diferencial: {dp_layer.mechanism.upper()}")
    #print(f"\n Iniciando servidor en http://{PRIVACY_API_HOST}:{PRIVACY_API_PORT}")
    #print(f" Documentación: http://localhost:{PRIVACY_API_PORT}/docs")
    #print("="*70 + "\n")

    uvicorn.run(app, host=PRIVACY_API_HOST, port=PRIVACY_API_PORT)

