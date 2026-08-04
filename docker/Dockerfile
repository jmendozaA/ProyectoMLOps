# ============================================
# Dockerfile - Student Performance Prediction
# ============================================
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema (mínimas para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODA la carpeta src para mantener la estructura de imports (src.inference, src.config, etc.)
COPY src/ ./src/

# Copiar el modelo pre-entrenado y el preprocesador
COPY model/ ./model/

# Variables de entorno
ENV MODEL_SOURCE=local
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MODEL_NAME=student-performance-model

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque (nota el "src." al principio)
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]