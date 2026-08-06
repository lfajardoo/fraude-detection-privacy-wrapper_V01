FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrar al directorio original del proyecto
#WORKDIR /app/modelo-python

# Entrenar modelo
RUN python train_model.py

EXPOSE 8000

# Ejecutar exactamente igual que local
CMD ["python", "app.py"]