"""
API FastAPI para inferencia del modelo
Servicio REST adaptado para Kubernetes con tracking de pod_name para balanceo de carga.
"""
import os
import platform
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


# 1. Agregar la raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, PROCESSED_DATA_DIR
from src.inference.model_loader import ModelLoader
from src.inference.schemas import StudentInput, PredictionOutput, HealthCheck

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# NOMBRE DEL POD (inyectado por Kubernetes automáticamente)
# ============================================================
POD_NAME = os.getenv("HOSTNAME") or "local-dev"

# ============================================================
# CARGA DEL PREPROCESADOR (Global)
# ============================================================
preprocessor = None
try:
    docker_path = Path("/app/model/preprocessor.pkl")
    local_path = project_root / "model" / "preprocessor.pkl"
    path_to_use = docker_path if docker_path.exists() else local_path
    
    if not path_to_use.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en {path_to_use}")
        
    preprocessor = joblib.load(path_to_use)
    logger.info(f"✅ Preprocesador cargado desde: {path_to_use}")
except Exception as e:
    logger.error(f"❌ Error cargando preprocesador: {e}")

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
model_loader: ModelLoader = None
start_time: float = None
request_count: int = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader, start_time
    start_time = time.time()
    logger.info(f"🚀 Iniciando servicio en pod: {POD_NAME}")
    
    model_loader = ModelLoader(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MODEL_NAME
    )
    
    try:
        model_loader.load_by_alias("champion")
        logger.info("✅ Modelo 'champion' cargado correctamente")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo cargar modelo 'champion': {e}")
        try:
            model_loader.load_latest_version()
            logger.info("✅ Última versión del modelo cargada como fallback")
        except Exception as e2:
            logger.error(f"❌ No se pudo cargar ningún modelo: {e2}")
            
    yield
    logger.info("🛑 Apagando servicio de inferencia...")

# ============================================================
# CREAR APLICACIÓN FASTAPI
# ============================================================
app = FastAPI(
    title="Student Performance Prediction API",
    description="API para predecir el rendimiento académico de estudiantes",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MIDDLEWARE
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    request_count += 1
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"[{POD_NAME}] [{request_count}] {request.method} {request.url.path} "
        f"- {response.status_code} - {duration:.3f}s"
    )
    return response

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {
        "service": "Student Performance Prediction API",
        "version": "1.0.0",
        "pod_name": POD_NAME,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    model_info = model_loader.get_model_info() if model_loader else {}
    return HealthCheck(
        status="healthy",
        model_loaded=model_info.get("is_loaded", False),
        model_version=model_info.get("model_version"),
        model_name=model_info.get("model_name"),
        pod_name=POD_NAME,
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(student: StudentInput):
    if model_loader is None or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocesador no disponible")
    
    try:
        input_dict = student.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        input_df_transformed = pd.DataFrame(
            preprocessor.transform(input_df),
            columns=preprocessor.get_feature_names_out()
        )
        
        prediction = model_loader.model.predict(input_df_transformed)[0]
        prediction = float(np.clip(prediction, 0, 100))
        
        model_info = model_loader.get_model_info()
        
        # ⭐ AQUÍ ESTÁ LA CLAVE: pasar pod_name=POD_NAME
        return PredictionOutput(
            prediction=round(prediction, 2),
            prediction_rounded=int(round(prediction)),
            model_version=model_info.get("model_version", "unknown"),
            model_name=model_info.get("model_name", MODEL_NAME),
            pod_name=POD_NAME,
            confidence_note="Prediction based on trained model"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(students: list[StudentInput]):
    if model_loader is None or model_loader.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocesador no disponible")
    
    try:
        input_data = [s.model_dump() for s in students]
        input_df = pd.DataFrame(input_data)
        
        input_df_transformed = pd.DataFrame(
            preprocessor.transform(input_df),
            columns=preprocessor.get_feature_names_out()
        )
        
        predictions = model_loader.model.predict(input_df_transformed)
        predictions = np.clip(predictions, 0, 100)
        
        model_info = model_loader.get_model_info()
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "prediction": round(float(pred), 2),
                "prediction_rounded": int(round(pred)),
            })
            
        return {
            "predictions": results,
            "total": len(results),
            "model_version": model_info.get("model_version", "unknown"),
            "pod_name": POD_NAME,  # ⭐ Y AQUÍ TAMBIÉN
        }
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/model/info")
async def model_info():
    if model_loader is None:
        return {"error": "Model loader no inicializado"}
    
    info = model_loader.get_model_info()
    info["uptime_seconds"] = round(time.time() - start_time, 2) if start_time else 0
    info["total_requests"] = request_count
    info["pod_name"] = POD_NAME
    return info

@app.get("/metrics")
async def get_metrics():
    return {
        "total_requests": request_count,
        "uptime_seconds": round(time.time() - start_time, 2) if start_time else 0,
        "model_loaded": model_loader.model is not None if model_loader else False,
        "pod_name": POD_NAME,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)