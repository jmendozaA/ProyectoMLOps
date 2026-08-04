"""
Registro y promoción de modelos en MLflow
===============================================================
Gestiona el Model Registry: versiones, stages y aliases.
"""

import mlflow
from mlflow.tracking import MlflowClient
import logging
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Configura la conexión a MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client


def get_best_run_id() -> str:
    """Obtiene el Run ID del mejor modelo (menor RMSE en test)."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        logger.error(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' no encontrado")
        sys.exit(1)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse ASC"],
        max_results=1
    )

    if runs.empty:
        logger.error("No se encontraron runs en el experimento")
        sys.exit(1)

    best_run_id = runs.iloc[0]["run_id"]
    best_rmse = runs.iloc[0]["metrics.test_rmse"]
    best_model_type = runs.iloc[0].get("params.model_type", "N/A")

    logger.info(f"Mejor run: {best_run_id}")
    logger.info(f"  Modelo: {best_model_type}")
    logger.info(f"  Test RMSE: {best_rmse:.4f}")

    return best_run_id


def register_model(run_id: str, model_name: str = MODEL_NAME) -> str:
    """
    Registra un modelo en el Model Registry.
    Retorna la versión registrada.
    """
    client = setup_mlflow()

    model_uri = f"runs:/{run_id}/model"

    logger.info(f"\nRegistrando modelo '{model_name}' desde run: {run_id}")

    # Registrar modelo
    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    logger.info(f"✅ Modelo registrado: {model_name} v{version}")

    return version


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Staging"
) -> None:
    """
    Transiciona una versión del modelo a un stage específico.
    Stages disponibles: None, Staging, Production, Archived
    """
    client = setup_mlflow()

    logger.info(f"\nTransicionando {model_name} v{version} → {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )

    logger.info(f"✅ {model_name} v{version} ahora está en stage: {stage}")


def set_model_alias(
    model_name: str,
    version: str,
    alias: str = "champion"
) -> None:
    """
    Asigna un alias a una versión del modelo.
    Aliases útiles: champion, challenger, production
    """
    client = setup_mlflow()

    logger.info(f"\nAsignando alias '{alias}' a {model_name} v{version}")

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )

    logger.info(f"✅ Alias '{alias}' asignado a {model_name} v{version}")


def add_model_description(
    model_name: str,
    version: str,
    description: str
) -> None:
    """Agrega una descripción a una versión del modelo."""
    client = setup_mlflow()

    client.update_model_version(
        name=model_name,
        version=version,
        description=description
    )

    logger.info(f"✅ Descripción actualizada para {model_name} v{version}")


def list_model_versions(model_name: str = MODEL_NAME) -> None:
    """Lista todas las versiones de un modelo registrado."""
    client = setup_mlflow()

    logger.info(f"\n📋 Versiones de '{model_name}':")
    logger.info("-" * 80)

    versions = client.search_model_versions(f"name='{model_name}'")

    for v in versions:
        logger.info(f"  v{v.version} | Stage: {v.current_stage} | "
                     f"Run ID: {v.run_id} | Created: {v.creation_timestamp}")

    logger.info("-" * 80)
    logger.info(f"  Total versiones: {len(versions)}")


def get_production_model(model_name: str = MODEL_NAME):
    """Obtiene el modelo en stage Production."""
    client = setup_mlflow()

    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ Modelo Production cargado: {model_uri}")
        return model, model_uri
    except Exception as e:
        logger.warning(f"⚠️ No hay modelo en Production: {e}")
        return None, None


def run_registration_pipeline():
    """
    Pipeline completo de registro:
    1. Obtiene el mejor modelo
    2. Lo registra en el Model Registry
    3. Lo transiciona a Staging
    4. Lista todas las versiones
    """
    logger.info("=" * 60)
    logger.info("PIPELINE DE REGISTRO DE MODELO")
    logger.info("=" * 60)

    # 1. Obtener mejor run
    best_run_id = get_best_run_id()

    # 2. Registrar modelo
    version = register_model(best_run_id)

    # 3. Transicionar a Staging
    transition_model_stage(MODEL_NAME, version, "Staging")

    # 4. Asignar alias
    set_model_alias(MODEL_NAME, version, "champion")

    # 5. Agregar descripción
    description = (
        f"Modelo registrado automáticamente desde el mejor run ({best_run_id}). "
        f"Entrenado con seed={42} en el experimento '{MLFLOW_EXPERIMENT_NAME}'."
    )
    add_model_description(MODEL_NAME, version, description)

    # 6. Listar versiones
    list_model_versions()

    logger.info("\n" + "=" * 60)
    logger.info("REGISTRO COMPLETADO EXITOSAMENTE")
    logger.info(f"  Modelo: {MODEL_NAME}")
    logger.info(f"  Versión: {version}")
    logger.info(f"  Stage: Staging")
    logger.info("=" * 60)

    return version


if __name__ == "__main__":
    version = run_registration_pipeline()