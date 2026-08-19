FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias (limpio caché y elimino __pycache__)
RUN pip install --no-cache-dir -r requirements.txt && \
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copiar código
COPY config.py .
COPY client.py .
COPY privacy_layer.py .
COPY governor.py .
COPY privacy_governor_simple.py .
COPY evaluator.py .
COPY metrics/ ./metrics/
COPY app.py .

# Exponer puerto
EXPOSE 8001

# Variable de ambiente para conectar a modelo-python
ENV MODEL_API_BASE_URL=http://modelo-financiero:8000

# Ejecutar app
CMD ["python", "app.py"]

