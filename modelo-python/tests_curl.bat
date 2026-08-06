@echo off
REM Script: tests_curl.bat
REM Descripción: Tests de la API de detección de fraude usando curl
REM Uso: Asegúrate de que app.py está en ejecución en otra terminal
REM      Luego ejecuta este archivo: tests_curl.bat

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   Tests de API - Detección de Fraude
echo ============================================================
echo.

set BASE_URL=http://localhost:8000

REM Test 1: Health Check
echo [1/6] Health Check...
curl -X GET "%BASE_URL%/health"
echo.
echo.

REM Test 2: Features List
echo [2/6] Features List...
curl -X GET "%BASE_URL%/features"
echo.
echo.

REM Test 3: API Info
echo [3/6] API Info...
curl -X GET "%BASE_URL%/"
echo.
echo.

REM Test 4: Predicción - Transacción Legítima
echo [4/6] Predicción - Transacción Legítima...
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{ \"Time\": 0.0, \"V1\": -1.35, \"V2\": -0.07, \"V3\": 2.53, \"V4\": 1.38, \"V5\": -0.33, \"V6\": 0.46, \"V7\": 0.24, \"V8\": 0.10, \"V9\": 0.36, \"V10\": 0.09, \"V11\": -0.55, \"V12\": -0.62, \"V13\": -0.99, \"V14\": -0.31, \"Amount\": 149.62 }"
echo.
echo.

REM Test 5: Predicción - Micro-pago
echo [5/6] Predicción - Micro-pago...
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{ \"Time\": 1000.0, \"V1\": -0.8, \"V2\": -0.2, \"V3\": 1.5, \"V4\": 0.9, \"V5\": -0.1, \"V6\": 0.3, \"V7\": 0.2, \"V8\": 0.05, \"V9\": 0.2, \"V10\": 0.05, \"V11\": -0.3, \"V12\": -0.4, \"V13\": -0.5, \"V14\": -0.15, \"Amount\": 5.00 }"
echo.
echo.

REM Test 6: Predicción - Transacción Sospechosa
echo [6/6] Predicción - Transacción Sospechosa...
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{ \"Time\": 50000.0, \"V1\": -2.5, \"V2\": -3.2, \"V3\": 1.8, \"V4\": 0.5, \"V5\": -2.0, \"V6\": 1.5, \"V7\": -1.2, \"V8\": 0.8, \"V9\": -0.5, \"V10\": -1.2, \"V11\": -2.5, \"V12\": -3.0, \"V13\": -1.5, \"V14\": -0.8, \"Amount\": 2500.00 }"
echo.
echo.

echo ============================================================
echo   Tests completados
echo ============================================================
echo.
pause

