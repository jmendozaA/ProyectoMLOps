"""
model_loader.py - Carga de modelos desde MLflow Model Registry
Gestiona la carga del modelo desde el registry, soportando
versionado por alias (recomendado en MLflow 2.9+) o por versión.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from typing import Optional

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        model_name: str = MODEL_NAME
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model = None
        self.model_version = None
        self.model_uri = None

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def load_by_alias(self, alias: str = "champion") -> object:
        """Carga el modelo por alias (recomendado en MLflow 2.9+)."""
        model_uri = f"models:/{self.model_name}@{alias}"
        logger.info(f"Cargando modelo desde alias: {model_uri}")

        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_uri = model_uri
            
            # Obtener información de la versión para logging
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            for v in versions:
                if alias in v.aliases:
                    self.model_version = v.version
                    break
            
            logger.info(f"✅ Modelo cargado exitosamente (alias: {alias}, versión: {self.model_version})")
            return self.model

        except Exception as e:
            logger.error(f"❌ Error cargando modelo con alias '{alias}': {e}")
            raise

    def load_latest_version(self) -> object:
        """Carga la última versión del modelo (fallback si no hay alias)."""
        logger.info("Cargando la última versión del modelo como fallback...")
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise ValueError(f"No se encontraron versiones del modelo '{self.model_name}'")
            
            latest_version = max(versions, key=lambda v: int(v.version))
            model_uri = f"models:/{self.model_name}/{latest_version.version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_uri = model_uri
            self.model_version = latest_version.version

            logger.info(f"✅ Modelo cargado exitosamente (versión: {self.model_version})")
            return self.model

        except Exception as e:
            logger.error(f"❌ Error cargando última versión: {e}")
            raise

    def get_model_info(self) -> dict:
        """Retorna información del modelo cargado."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "is_loaded": self.model is not None,
        }

    def reload(self, alias: str = "champion") -> object:
        """Recarga el modelo (útil para actualizaciones)."""
        logger.info("🔄 Recargando modelo...")
        self.model = None
        self.model_version = None
        self.model_uri = None
        return self.load_by_alias(alias)