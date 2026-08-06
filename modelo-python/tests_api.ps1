# Script: tests_api.ps1
# Descripción: Tests de la API de detección de fraude usando PowerShell
# Uso: Asegúrate de que app.py está en ejecución en otra terminal
#      Luego ejecuta desde PowerShell: .\tests_api.ps1

Write-Host ""
Write-Host "============================================================"
Write-Host "  Tests de API - Detección de Fraude"
Write-Host "============================================================"
Write-Host ""

$BASE_URL = "http://localhost:8000"
$ProgressPreference = 'SilentlyContinue'

# Test 1: Health Check
Write-Host "[1/6] Health Check..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 2: Features List
Write-Host "[2/6] Features List..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/features" -Method Get
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: API Info
Write-Host "[3/6] API Info..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/" -Method Get
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: Predicción - Transacción Legítima
Write-Host "[4/6] Predicción - Transacción Legítima (USD 149.62)" -ForegroundColor Cyan
$payload = @{
    Time = 0.0
    V1 = -1.35
    V2 = -0.07
    V3 = 2.53
    V4 = 1.38
    V5 = -0.33
    V6 = 0.46
    V7 = 0.24
    V8 = 0.10
    V9 = 0.36
    V10 = 0.09
    V11 = -0.55
    V12 = -0.62
    V13 = -0.99
    V14 = -0.31
    Amount = 149.62
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method Post `
        -ContentType "application/json" -Body $payload
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 5: Predicción - Micro-pago
Write-Host "[5/6] Predicción - Micro-pago (USD 5.00)" -ForegroundColor Cyan
$payload = @{
    Time = 1000.0
    V1 = -0.8
    V2 = -0.2
    V3 = 1.5
    V4 = 0.9
    V5 = -0.1
    V6 = 0.3
    V7 = 0.2
    V8 = 0.05
    V9 = 0.2
    V10 = 0.05
    V11 = -0.3
    V12 = -0.4
    V13 = -0.5
    V14 = -0.15
    Amount = 5.00
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method Post `
        -ContentType "application/json" -Body $payload
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 6: Predicción - Transacción Sospechosa
Write-Host "[6/6] Predicción - Transacción Sospechosa (USD 2,500)" -ForegroundColor Cyan
$payload = @{
    Time = 50000.0
    V1 = -2.5
    V2 = -3.2
    V3 = 1.8
    V4 = 0.5
    V5 = -2.0
    V6 = 1.5
    V7 = -1.2
    V8 = 0.8
    V9 = -0.5
    V10 = -1.2
    V11 = -2.5
    V12 = -3.0
    V13 = -1.5
    V14 = -0.8
    Amount = 2500.00
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method Post `
        -ContentType "application/json" -Body $payload
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 7: Predicción - Monto Extremo
Write-Host "[7/7] Predicción - Monto Extremo (USD 15,000)" -ForegroundColor Cyan
$payload = @{
    Time = 80000.0
    V1 = -2.8
    V2 = -2.5
    V3 = 0.3
    V4 = -1.5
    V5 = -2.0
    V6 = 1.8
    V7 = -1.5
    V8 = 0.7
    V9 = -0.7
    V10 = -1.5
    V11 = -2.5
    V12 = -2.0
    V13 = -1.0
    V14 = -0.5
    Amount = 15000.00
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$BASE_URL/predict" -Method Post `
        -ContentType "application/json" -Body $payload
    Write-Host (ConvertTo-Json $response) -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "============================================================"
Write-Host "  ✓ Tests completados" -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""

