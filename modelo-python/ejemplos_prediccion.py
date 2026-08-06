"""
SCRIPT DE PRUEBA: Ejemplos de Predicción - Modelo Python de Detección de Fraude

Este archivo contiene ejemplos de cómo:
1. Llamar a la API desde Python (requests)
2. Usar directamente la clase FraudModel
3. Crear transacciones de prueba

Ejecutar desde la carpeta modelo-python:
    python ejemplos_prediccion.py
"""

import json
import sys
from pathlib import Path

# ============================================================================
# EJEMPLO 1: USO DIRECTO DE LA CLASE FraudModel (sin API)
# ============================================================================

def ejemplo_1_uso_directo():
    """Usar directamente la clase FraudModel sin levantar servidor FastAPI"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Uso Directo de FraudModel (Sin API)")
    print("="*70)

    try:
        from model import FraudModel

        # Cargar modelo
        print("\n[1] Cargando modelo...")
        fraud_model = FraudModel()
        print("    ✓ Modelo cargado exitosamente")

        # Obtener lista de features
        features = fraud_model.get_feature_names()
        print(f"\n[2] Features requeridos ({len(features)} total):")
        for i, f in enumerate(features, 1):
            print(f"    {i:2d}. {f}")

        # Transacción de prueba 1: LEGÍTIMA
        print("\n[3] Predicción - Transacción LEGÍTIMA:")
        transaccion_legitima = {
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

        resultado1 = fraud_model.predict(transaccion_legitima)
        print(f"    Entrada: Monto=${transaccion_legitima['Amount']:.2f}")
        print(f"    Salida: {json.dumps(resultado1, indent=6)}")
        print(f"    → Interpretación: {('✓ LEGÍTIMA' if resultado1['is_fraud']==0 else '⚠️ FRAUDE')} (confianza: {resultado1['confidence_score']*100:.2f}%)")

        # Transacción de prueba 2: SOSPECHOSA
        print("\n[4] Predicción - Transacción SOSPECHOSA:")
        transaccion_sospechosa = {
            "Time": 50000.0,
            "V1": -2.5,
            "V2": -3.2,
            "V3": 1.8,
            "V4": 0.5,
            "V5": -2.0,
            "V6": 1.5,
            "V7": -1.2,
            "V8": 0.8,
            "V9": -0.5,
            "V10": -1.2,
            "V11": -2.5,
            "V12": -3.0,
            "V13": -1.5,
            "V14": -0.8,
            "Amount": 2500.00
        }

        resultado2 = fraud_model.predict(transaccion_sospechosa)
        print(f"    Entrada: Monto=${transaccion_sospechosa['Amount']:.2f}")
        print(f"    Salida: {json.dumps(resultado2, indent=6)}")
        print(f"    → Interpretación: {('✓ LEGÍTIMA' if resultado2['is_fraud']==0 else '⚠️ FRAUDE')} (confianza: {resultado2['confidence_score']*100:.2f}%)")

        # Transacción de prueba 3: MONTO MUY ALTO
        print("\n[5] Predicción - Transacción MONTO EXTREMO:")
        transaccion_extremo = {
            "Time": 85000.0,
            "V1": -3.0,
            "V2": -2.8,
            "V3": 0.5,
            "V4": -1.5,
            "V5": -2.5,
            "V6": 2.5,
            "V7": -2.0,
            "V8": 1.0,
            "V9": -1.0,
            "V10": -2.0,
            "V11": -3.5,
            "V12": -2.5,
            "V13": -1.0,
            "V14": -0.5,
            "Amount": 15000.00
        }

        resultado3 = fraud_model.predict(transaccion_extremo)
        print(f"    Entrada: Monto=${transaccion_extremo['Amount']:.2f}")
        print(f"    Salida: {json.dumps(resultado3, indent=6)}")
        print(f"    → Interpretación: {('✓ LEGÍTIMA' if resultado3['is_fraud']==0 else '⚠️ FRAUDE')} (confianza: {resultado3['confidence_score']*100:.2f}%)")

        print("\n✓ Ejemplo 1 completado exitosamente")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("    Asegúrate de haber ejecutado: python train_model.py")


# ============================================================================
# EJEMPLO 2: LLAMADAS HTTP A LA API (usando requests)
# ============================================================================

def ejemplo_2_api_http():
    """Llamar a la API FastAPI mediante requests HTTP"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Llamadas HTTP a la API FastAPI")
    print("="*70)

    try:
        import requests

        BASE_URL = "http://localhost:8000"

        # Verificar si el servidor está en línea
        print("\n[1] Verificando estado del servidor...")
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            health = response.json()
            print(f"    ✓ Servidor en línea: {health}")
        except requests.exceptions.ConnectionError:
            print(f"    ❌ No se puede conectar a {BASE_URL}")
            print("    ℹ️  Asegúrate de que app.py está en ejecución")
            print("       Ejecuta en otra terminal: python app.py")
            return

        # Obtener lista de features
        print("\n[2] Obteniendo lista de features...")
        response = requests.get(f"{BASE_URL}/features")
        features_data = response.json()
        print(f"    Total de features: {features_data['total']}")
        print(f"    Features: {', '.join(features_data['features'])}")

        # Realizar predicción - Caso 1: LEGÍTIMA
        print("\n[3] POST /predict - Transacción LEGÍTIMA")
        payload1 = {
            "Time": 100.0,
            "V1": -0.5,
            "V2": 0.3,
            "V3": 1.5,
            "V4": 0.8,
            "V5": -0.2,
            "V6": 0.1,
            "V7": 0.5,
            "V8": -0.1,
            "V9": 0.2,
            "V10": -0.3,
            "V11": 0.1,
            "V12": -0.4,
            "V13": -0.2,
            "V14": 0.0,
            "Amount": 50.00
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload1,
            headers={"Content-Type": "application/json"}
        )
        prediction1 = response.json()
        print(f"    Respuesta:")
        print(f"    {json.dumps(prediction1, indent=6)}")

        # Realizar predicción - Caso 2: SOSPECHOSA
        print("\n[4] POST /predict - Transacción SOSPECHOSA")
        payload2 = {
            "Time": 60000.0,
            "V1": -2.0,
            "V2": -2.5,
            "V3": 0.8,
            "V4": -0.5,
            "V5": -1.5,
            "V6": 1.5,
            "V7": -1.0,
            "V8": 0.5,
            "V9": -0.8,
            "V10": -1.5,
            "V11": -2.0,
            "V12": -2.5,
            "V13": -1.2,
            "V14": -0.6,
            "Amount": 3000.00
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload2,
            headers={"Content-Type": "application/json"}
        )
        prediction2 = response.json()
        print(f"    Respuesta:")
        print(f"    {json.dumps(prediction2, indent=6)}")

        print("\n✓ Ejemplo 2 completado exitosamente")

    except ImportError:
        print("\n❌ Librería 'requests' no instalada")
        print("    Ejecuta: python -m pip install requests")
    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================================
# EJEMPLO 3: BATCH DE PREDICCIONES (varias transacciones)
# ============================================================================

def ejemplo_3_batch_predicciones():
    """Realizar predicciones en lote sobre múltiples transacciones"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Batch de Predicciones")
    print("="*70)

    try:
        from model import FraudModel

        print("\n[1] Cargando modelo...")
        fraud_model = FraudModel()

        # Crear lote de transacciones de prueba
        transacciones_batch = [
            {
                "id": "TX001",
                "data": {
                    "Time": 0.0, "V1": -1.0, "V2": -0.5, "V3": 2.0, "V4": 1.0,
                    "V5": -0.3, "V6": 0.4, "V7": 0.2, "V8": 0.1, "V9": 0.3,
                    "V10": 0.1, "V11": -0.5, "V12": -0.6, "V13": -1.0, "V14": -0.3,
                    "Amount": 100.0
                }
            },
            {
                "id": "TX002",
                "data": {
                    "Time": 5000.0, "V1": -1.5, "V2": -1.0, "V3": 1.5, "V4": 0.5,
                    "V5": -0.8, "V6": 0.8, "V7": -0.5, "V8": 0.5, "V9": -0.2,
                    "V10": -0.8, "V11": -1.5, "V12": -1.2, "V13": -0.8, "V14": -0.4,
                    "Amount": 500.0
                }
            },
            {
                "id": "TX003",
                "data": {
                    "Time": 50000.0, "V1": -2.5, "V2": -3.0, "V3": 0.5, "V4": -1.0,
                    "V5": -2.0, "V6": 2.0, "V7": -1.5, "V8": 0.8, "V9": -0.8,
                    "V10": -1.5, "V11": -2.5, "V12": -2.0, "V13": -1.0, "V14": -0.5,
                    "Amount": 5000.0
                }
            },
            {
                "id": "TX004",
                "data": {
                    "Time": 100000.0, "V1": -3.0, "V2": -2.5, "V3": -0.5, "V4": -2.0,
                    "V5": -2.5, "V6": 1.5, "V7": -2.0, "V8": 1.0, "V9": -1.0,
                    "V10": -2.0, "V11": -3.0, "V12": -2.5, "V13": -1.5, "V14": -0.8,
                    "Amount": 20000.0
                }
            }
        ]

        print(f"\n[2] Procesando {len(transacciones_batch)} transacciones...")

        resultados = []
        for tx in transacciones_batch:
            prediccion = fraud_model.predict(tx["data"])
            resultados.append({
                "tx_id": tx["id"],
                "monto": tx["data"]["Amount"],
                **prediccion
            })

        # Mostrar resultados en tabla
        print("\n[3] Resultados:")
        print(f"\n{'ID':10} {'Monto':>10} {'Fraude':>10} {'Probabilidad':>15} {'Confianza':>12}")
        print("-" * 60)

        for r in resultados:
            fraude_str = "⚠️ SÍ" if r["is_fraud"] == 1 else "✓ NO"
            print(f"{r['tx_id']:10} ${r['monto']:>9.2f} {fraude_str:>10} {r['fraud_probability']:>14.2%} {r['confidence_score']:>11.2%}")

        # Estadísticas
        total_fraudulentas = sum(1 for r in resultados if r["is_fraud"] == 1)
        print(f"\n[4] Resumen:")
        print(f"    Total procesadas: {len(resultados)}")
        print(f"    Flaggeadas como fraude: {total_fraudulentas}")
        print(f"    Legítimas: {len(resultados) - total_fraudulentas}")

        print("\n✓ Ejemplo 3 completado exitosamente")

    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================================
# EJEMPLO 4: COMPARACIÓN DE ESCENARIOS
# ============================================================================

def ejemplo_4_comparacion_escenarios():
    """Comparar predicciones en diferentes escenarios"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Comparación de Escenarios")
    print("="*70)

    try:
        from model import FraudModel

        fraud_model = FraudModel()

        escenarios = {
            "Micro-pago (pequeño monto)": {
                "Time": 1000.0,
                "V1": -0.8, "V2": -0.2, "V3": 1.5, "V4": 0.9,
                "V5": -0.1, "V6": 0.3, "V7": 0.2, "V8": 0.05, "V9": 0.2,
                "V10": 0.05, "V11": -0.3, "V12": -0.4, "V13": -0.5, "V14": -0.15,
                "Amount": 5.00
            },
            "Compra normal": {
                "Time": 2000.0,
                "V1": -1.0, "V2": -0.3, "V3": 2.0, "V4": 1.2,
                "V5": -0.2, "V6": 0.4, "V7": 0.25, "V8": 0.1, "V9": 0.25,
                "V10": 0.08, "V11": -0.4, "V12": -0.5, "V13": -0.7, "V14": -0.25,
                "Amount": 150.00
            },
            "Compra alta": {
                "Time": 50000.0,
                "V1": -1.5, "V2": -0.8, "V3": 1.5, "V4": 0.3,
                "V5": -0.6, "V6": 0.9, "V7": -0.3, "V8": 0.4, "V9": -0.1,
                "V10": -0.6, "V11": -1.2, "V12": -1.0, "V13": -0.6, "V14": -0.3,
                "Amount": 2000.00
            },
            "Compra muy alta (sospechosa)": {
                "Time": 80000.0,
                "V1": -2.8, "V2": -2.5, "V3": 0.3, "V4": -1.5,
                "V5": -2.0, "V6": 1.8, "V7": -1.5, "V8": 0.7, "V9": -0.7,
                "V10": -1.5, "V11": -2.5, "V12": -2.0, "V13": -1.0, "V14": -0.5,
                "Amount": 15000.00
            }
        }

        print("\n")
        for escenario, datos in escenarios.items():
            prediccion = fraud_model.predict(datos)
            resultado = "⚠️ FRAUDE" if prediccion["is_fraud"] == 1 else "✓ LEGÍTIMA"

            print(f"Escenario: {escenario}")
            print(f"  Monto: ${datos['Amount']:.2f}")
            print(f"  Predicción: {resultado}")
            print(f"  Probabilidad de fraude: {prediccion['fraud_probability']:.2%}")
            print(f"  Confianza del modelo: {prediccion['confidence_score']:.2%}")
            print()

        print("✓ Ejemplo 4 completado exitosamente")

    except Exception as e:
        print(f"\n❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  EJEMPLOS DE PREDICCIÓN - Modelo de Detección de Fraude".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    # Ejecutar ejemplos
    ejemplo_1_uso_directo()
    ejemplo_2_api_http()
    ejemplo_3_batch_predicciones()
    ejemplo_4_comparacion_escenarios()

    print("\n" + "="*70)
    print("✓ TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*70)
    print("\nNotas:")
    print("  • Ejemplo 1: No requiere servidor en ejecución")
    print("  • Ejemplo 2: Requiere ejecutar 'python app.py' en otra terminal")
    print("  • Ejemplos 3 y 4: No requieren servidor en ejecución")
    print("\n")

