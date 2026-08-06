#!/usr/bin/env python3
"""
VERIFICADOR DE INSTALACIÓN - Detección de Fraude

Este script verifica que todo está correctamente instalado y configurado.
Ejecutar desde la carpeta modelo-python:

    python verificador_instalacion.py

Hará una serie de chequeos y reportará el estado de todo.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Imprimir encabezado"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_ok(text):
    """Imprimir con check"""
    print(f"✅ {text}")

def print_error(text):
    """Imprimir error"""
    print(f"❌ {text}")

def print_warning(text):
    """Imprimir advertencia"""
    print(f"⚠️  {text}")

def print_info(text):
    """Imprimir información"""
    print(f"ℹ️  {text}")

def check_python_version():
    """Verificar versión de Python"""
    print_header("1. Verificación de Python")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        print_ok(f"Python {version_str}")
        return True
    else:
        print_error(f"Python {version_str} - Se requiere 3.10+")
        return False

def check_dependencies():
    """Verificar dependencias instaladas"""
    print_header("2. Verificación de Dependencias")

    required_packages = [
        'fastapi',
        'uvicorn',
        'sklearn',
        'pandas',
        'numpy',
        'joblib',
        'pydantic'
    ]

    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print_ok(f"{package}")
        except ImportError:
            print_error(f"{package} NO INSTALADO")
            all_ok = False

    if not all_ok:
        print_warning("Ejecuta: python -m pip install -r requirements.txt")

    return all_ok

def check_model_files():
    """Verificar artefactos del modelo"""
    print_header("3. Verificación de Artefactos del Modelo")

    models_dir = Path("models")
    required_files = [
        "fraud_model.pkl",
        "scaler.pkl",
        "features.pkl"
    ]

    all_ok = True
    if not models_dir.exists():
        print_error("Carpeta 'models' no existe")
        print_info("Ejecuta: python train_model.py")
        return False

    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print_ok(f"{file} ({size_kb:.1f} KB)")
        else:
            print_error(f"{file} NO EXISTE")
            all_ok = False

    if not all_ok:
        print_info("Ejecuta: python train_model.py")

    return all_ok

def check_source_files():
    """Verificar archivos fuente"""
    print_header("4. Verificación de Archivos Fuente")

    required_files = [
        "app.py",
        "model.py",
        "train_model.py"
    ]

    all_ok = True
    for file in required_files:
        if Path(file).exists():
            size_kb = Path(file).stat().st_size / 1024
            print_ok(f"{file} ({size_kb:.1f} KB)")
        else:
            print_error(f"{file} NO EXISTE")
            all_ok = False

    return all_ok

def check_model_loadable():
    """Verificar que el modelo se puede cargar"""
    print_header("5. Verificación de Carga del Modelo")

    try:
        from model import FraudModel
        print_ok("Clase FraudModel importable")

        try:
            model = FraudModel()
            print_ok("Modelo cargado exitosamente")

            features = model.get_feature_names()
            print_ok(f"Features cargados: {len(features)} características")

            if len(features) == 16:
                print_ok("Número de features correcto (16)")
                return True
            else:
                print_error(f"Esperados 16 features, encontrados {len(features)}")
                return False

        except FileNotFoundError:
            print_error("Artefactos del modelo no encontrados")
            print_info("Ejecuta: python train_model.py")
            return False

    except ImportError as e:
        print_error(f"Error importando FraudModel: {e}")
        return False

def check_prediction_works():
    """Verificar que una predicción funciona"""
    print_header("6. Verificación de Predicción")

    try:
        from model import FraudModel

        model = FraudModel()

        # Transacción de prueba
        test_transaction = {
            "Time": 0.0,
            "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38, "V5": -0.33,
            "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
            "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
            "Amount": 149.62
        }

        result = model.predict(test_transaction)

        print_ok("Predicción realizada exitosamente")
        print(f"  - is_fraud: {result['is_fraud']}")
        print(f"  - fraud_probability: {result['fraud_probability']:.4f}")
        print(f"  - confidence_score: {result['confidence_score']:.4f}")

        # Validar estructura de respuesta
        if all(k in result for k in ['is_fraud', 'fraud_probability', 'confidence_score']):
            print_ok("Estructura de respuesta correcta")
            return True
        else:
            print_error("Estructura de respuesta incorrecta")
            return False

    except Exception as e:
        print_error(f"Error en predicción: {e}")
        return False

def check_documentation():
    """Verificar documentación"""
    print_header("7. Verificación de Documentación")

    # Ir a la carpeta padre para verificar documentación
    parent_dir = Path("..").resolve()

    docs = [
        ("INDICE_DOCUMENTACION.md", parent_dir),
        ("EJECUCION_LOCAL_MODELO_PYTHON.md", parent_dir),
        ("RESUMEN_VISUAL_MODELO_PYTHON.txt", parent_dir),
    ]

    all_ok = True
    for doc, directory in docs:
        doc_path = directory / doc
        if doc_path.exists():
            size_kb = doc_path.stat().st_size / 1024
            print_ok(f"{doc} ({size_kb:.1f} KB)")
        else:
            print_warning(f"{doc} no encontrado en {directory}")
            all_ok = False

    return all_ok

def check_test_files():
    """Verificar archivos de test"""
    print_header("8. Verificación de Archivos de Test")

    test_files = [
        "ejemplos_prediccion.py",
        "tests_api.ps1",
        "tests_curl.bat",
        "postman_collection.json"
    ]

    all_ok = True
    for file in test_files:
        if Path(file).exists():
            size_kb = Path(file).stat().st_size / 1024
            print_ok(f"{file} ({size_kb:.1f} KB)")
        else:
            print_warning(f"{file} no encontrado")
            all_ok = False

    return all_ok

def main():
    """Ejecutar todas las verificaciones"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  VERIFICADOR DE INSTALACIÓN - Detección de Fraude".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencias", check_dependencies),
        ("Archivos Fuente", check_source_files),
        ("Artefactos del Modelo", check_model_files),
        ("Carga del Modelo", check_model_loadable),
        ("Predicción", check_prediction_works),
        ("Documentación", check_documentation),
        ("Archivos de Test", check_test_files),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Error en {name}: {e}")
            results[name] = False

    # Resumen final
    print_header("RESUMEN FINAL")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} verificaciones pasadas")

    if passed == total:
        print("\n" + "🎉 "*10)
        print("\n¡TODAS LAS VERIFICACIONES PASARON!")
        print("\nEstá todo listo para usar. Próximos pasos:")
        print("  1. python app.py          # Arrancar servidor")
        print("  2. .\tests_api.ps1         # Probar predicciones")
        print("\nO lee la documentación:")
        print("  - RESUMEN_VISUAL_MODELO_PYTHON.txt (5 min)")
        print("  - EJECUCION_LOCAL_MODELO_PYTHON.md (30 min)")
        print("\n" + "🎉 "*10)
    else:
        print("\n⚠️  Algunas verificaciones fallaron.")
        print("\nSoluciona los problemas marcados y ejecuta nuevamente:")
        print("  python verificador_instalacion.py")

    print("\n")

if __name__ == "__main__":
    main()

