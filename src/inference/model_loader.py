"""
model_loader.py - Carga de modelos desde MLflow Model Registry o localmente
Gestiona la carga del modelo, priorizando el registro pero con fallback local
para garantizar la alta disponibilidad en el contenedor.
"""
import sys
import os
import joblib
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Gestor de carga de modelos.
    Intenta cargar desde MLflow Registry, y si falla, usa el archivo local empaquetado.
    """

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        model_name: str = MODEL_NAME
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model = None
        self.model_version = "unknown"
        self.model_uri = "unknown"

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def load_by_alias(self, alias: str = "champion") -> object:
        """Intenta cargar el modelo por alias desde MLflow."""
        model_uri = f"models:/{self.model_name}@{alias}"
        logger.info(f"Intentando cargar modelo desde MLflow alias: {model_uri}")

        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_uri = model_uri
            
            # Intentar obtener la versión real para los logs
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            for v in versions:
                if alias in v.aliases:
                    self.model_version = v.version
                    break
                    
            logger.info(f"✅ Modelo cargado exitosamente desde MLflow (alias: {alias}, versión: {self.model_version})")
            return self.model

        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar desde MLflow ({e}). Usando fallback local...")
            return self._load_local_model()

    def _load_local_model(self) -> object:
        """Carga el modelo desde el archivo local empaquetado en la imagen Docker."""
        # Rutas posibles: dentro del contenedor Docker o en desarrollo local
        docker_path = Path("/app/model/model.joblib")
        local_path = project_root / "model" / "model.joblib"
        
        model_path = docker_path if docker_path.exists() else local_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo ni en MLflow ni localmente en {model_path}")
            
        logger.info(f"Cargando modelo desde archivo local empaquetado: {model_path}")
        self.model = joblib.load(model_path)
        self.model_uri = str(model_path)
        self.model_version = "local-packaged"
        
        logger.info("✅ Modelo local cargado exitosamente")
        return self.model

    def get_model_info(self) -> dict:
        """Retorna información del modelo cargado."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "is_loaded": self.model is not None,
        }

    def reload(self, alias: str = "champion") -> object:
        """Recarga el modelo (útil para actualizaciones en caliente)."""
        logger.info("🔄 Recargando modelo...")
        self.model = None
        self.model_version = "unknown"
        self.model_uri = "unknown"
        return self.load_by_alias(alias)