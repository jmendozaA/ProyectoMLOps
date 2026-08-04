"""
API FastAPI para inferencia del modelo
Servicio REST que consume el modelo desde MLflow Model Registry.
"""
import sys
from pathlib import Path
import joblib  # <-- Agregado para cargar el preprocesador
import mlflow
import pandas as pd
import numpy as np
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path

# 1. Agregar la raíz del proyecto al path de Python para resolver importaciones
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# <-- Agregado PROCESSED_DATA_DIR para encontrar el preprocesador
from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, PROCESSED_DATA_DIR
from src.inference.model_loader import ModelLoader
from src.inference.schemas import StudentInput, PredictionOutput, HealthCheck

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CARGA DEL PREPROCESADOR (Global)
# ============================================================
preprocessor = None
try:
    # En Docker, el archivo está en /app/model/preprocessor.pkl
    # En desarrollo local, está en la carpeta model/ relativa a la raíz del proyecto
    docker_path = Path("/app/model/preprocessor.pkl")
    local_path = Path(__file__).parent.parent.parent / "model" / "preprocessor.pkl"
    
    # Usar la ruta de Docker si existe, si no, la local
    path_to_use = docker_path if docker_path.exists() else local_path
    
    if not path_to_use.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en {path_to_use}. Asegúrate de haber ejecutado scripts/save_model_local.py")
        
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
    """Gestiona el ciclo de vida de la aplicación."""
    global model_loader, start_time
    
    start_time = time.time()
    logger.info("🚀 Iniciando servicio de inferencia...")
    
    # Cargar modelo al inicio
    model_loader = ModelLoader(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MODEL_NAME
    )
    
    try:
        model_loader.load_by_alias("champion")
        logger.info("✅ Modelo 'champion' cargado correctamente")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo cargar modelo 'champion': {e}")
        logger.info("Intentando cargar la última versión como fallback...")
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

# CORS
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
    """Middleware para loguear requests."""
    global request_count
    request_count += 1
    
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    logger.info(
        f"[{request_count}] {request.method} {request.url.path} "
        f"- {response.status_code} - {duration:.3f}s"
    )
    
    return response

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz."""
    return {
        "service": "Student Performance Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check del servicio."""
    model_info = model_loader.get_model_info() if model_loader else {}
    
    return HealthCheck(
        status="healthy",
        model_loaded=model_info.get("is_loaded", False),
        model_version=model_info.get("model_version"),
        model_name=model_info.get("model_name"),
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(student: StudentInput):
    """
    Predice el rendimiento académico de un estudiante.
    """
    global request_count
    
    if model_loader is None or model_loader.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. El servicio no está listo."
        )
    
    if preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Preprocesador no disponible. No se puede transformar la entrada."
        )
    
    try:
        # 1. Convertir input a DataFrame
        input_dict = student.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # 2. ¡IMPORTANTE! Aplicar la misma transformación que en entrenamiento
        input_df_transformed = pd.DataFrame(
            preprocessor.transform(input_df),
            columns=preprocessor.get_feature_names_out()
        )
        
        # 3. Hacer predicción con los datos YA transformados (numéricos)
        prediction = model_loader.model.predict(input_df_transformed)[0]
        prediction = float(np.clip(prediction, 0, 100))  # Clamp entre 0 y 100
        
        # Obtener info del modelo
        model_info = model_loader.get_model_info()
        
        return PredictionOutput(
            prediction=round(prediction, 2),
            prediction_rounded=int(round(prediction)),
            model_version=model_info.get("model_version", "unknown"),
            model_name=model_info.get("model_name", MODEL_NAME),
            confidence_note="Prediction based on trained model"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar la predicción: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(students: list[StudentInput]):
    """Predicción en lote para múltiples estudiantes."""
    if model_loader is None or model_loader.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )
    
    if preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Preprocesador no disponible."
        )
    
    try:
        input_data = [s.model_dump() for s in students]
        input_df = pd.DataFrame(input_data)
        
        # Transformar lote completo
        input_df_transformed = pd.DataFrame(
            preprocessor.transform(input_df),
            columns=preprocessor.get_feature_names_out()
        )
        
        # Predicción en lote
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
        }
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción batch: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Información detallada del modelo cargado."""
    if model_loader is None:
        return {"error": "Model loader no inicializado"}
    
    info = model_loader.get_model_info()
    info["uptime_seconds"] = round(time.time() - start_time, 2) if start_time else 0
    info["total_requests"] = request_count
    
    return info

@app.post("/model/reload")
async def reload_model(alias: str = "champion"):
    """Recarga el modelo (útil tras actualizaciones)."""
    global model_loader
    
    try:
        model_loader.reload(alias)
        return {
            "status": "success",
            "message": f"Modelo recargado desde alias: {alias}",
            "model_info": model_loader.get_model_info()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recargando modelo: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """Métricas del servicio."""
    return {
        "total_requests": request_count,
        "uptime_seconds": round(time.time() - start_time, 2) if start_time else 0,
        "model_loaded": model_loader.model is not None if model_loader else False,
    }

# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )