# 🚀 QUICK START - Detección de Fraude (Modelo Python)

## ⚡ 5 Minutos para Empezar

### PASO 1: Instalar Dependencias (3 min)
```powershell
cd modelo-python
python -m pip install -r requirements.txt
```

### PASO 2: Entrenar Modelo (10 seg)
```powershell
python train_model.py
```

Salida esperada:
```
✓ Accuracy en entrenamiento: 0.9854
✓ Accuracy en test: 0.9809
✓ Modelo entrenado y guardado exitosamente!
```

### PASO 3: Arrancar Servidor (1 seg)
```powershell
python app.py
```

Salida esperada:
```
Iniciando servidor en http://localhost:8000
Documentación interactiva: http://localhost:8000/docs
```

### PASO 4: Hacer Predicciones (Abrir OTRA terminal)
```powershell
cd modelo-python
.\tests_api.ps1
```

O en Python:
```powershell
python ejemplos_prediccion.py
```

---

## 📊 Ejemplo: Predicción de Fraude

### Request (POST http://localhost:8000/predict):
```json
{
  "Time": 0.0,
  "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38, "V5": -0.33,
  "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
  "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
  "Amount": 149.62
}
```

### Response:
```json
{
  "is_fraud": 0,
  "fraud_probability": 0.0324,
  "confidence_score": 0.9676,
  "message": "✓ Transacción legítima"
}
```

---

## 📖 Documentación Disponible

| Archivo | Tiempo | Para Quién |
|---------|--------|-----------|
| `RESUMEN_VISUAL_MODELO_PYTHON.txt` | 5 min | Referencia rápida |
| `EJECUCION_LOCAL_MODELO_PYTHON.md` | 30 min | Comprensión profunda |
| `INDICE_DOCUMENTACION.md` | 1 min | Navegación |
| `ejemplos_prediccion.py` | ejecutable | Ejemplos de código |
| `tests_api.ps1` | ejecutable | Tests automáticos |
| `postman_collection.json` | importable | Postman/Insomnia |

---

## ✅ Verificar Instalación

```powershell
cd modelo-python
python verificador_instalacion.py
```

---

## 🛠️ Troubleshooting Rápido

**Q: "Modelo no cargado"**  
A: Ejecuta `python train_model.py` primero

**Q: "FileNotFoundError"**  
A: Asegúrate de estar en la carpeta `modelo-python`

**Q: "pip no reconocido"**  
A: Usa `python -m pip` en su lugar

**Q: "sklearn compilation error"**  
A: requirements.txt ya está actualizado, reinstala: `python -m pip install -r requirements.txt`

---

## 🎯 Próximos Pasos

1. ✅ Lee `INDICE_DOCUMENTACION.md` (punto de entrada)
2. ✅ Elige tu ruta de aprendizaje
3. ✅ Explora los ejemplos y tests

---

## 💡 API Endpoints

```
GET  http://localhost:8000/           # Info del servicio
GET  http://localhost:8000/health     # Estado del servidor
GET  http://localhost:8000/features   # Lista de características
POST http://localhost:8000/predict    # Predicción (endpoint principal)
GET  http://localhost:8000/docs       # Documentación interactiva (Swagger)
```

---

## 🎉 ¡Listo!

**Status**: ✅ Funcional y documentado

**Comienza con**: `INDICE_DOCUMENTACION.md`

---

*Detección de Fraude - Modelo Python v1.0.0*

