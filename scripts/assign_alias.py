"""
assign_alias.py - Asigna el alias 'champion' al modelo registrado
Uso: python scripts/assign_alias.py [versión]
"""
import mlflow
from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME


def assign_champion_alias(version: str = None):
    """
    Asigna el alias 'champion' a una versión específica del modelo.
    Si no se especifica versión, usa la última registrada.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    print(f"🔍 Buscando modelo: {MODEL_NAME}")

    # Obtener todas las versiones del modelo
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        print(f"❌ No se encontró el modelo '{MODEL_NAME}'")
        return False

    # Determinar la versión a usar
    if version is None:
        # Usar la última versión
        target_version = max(versions, key=lambda v: int(v.version))
    else:
        # Buscar la versión específica
        matching = [v for v in versions if str(v.version) == str(version)]
        if not matching:
            print(f"❌ No se encontró la versión {version}")
            return False
        target_version = matching[0]

    print(f"📦 Versión seleccionada: {target_version.version}")
    print(f"   Stage actual: {target_version.current_stage}")
    print(f"   Aliases actuales: {target_version.aliases}")
    print(f"   Run ID: {target_version.run_id}")

    # Asignar el alias "champion"
    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=target_version.version
        )

        print(f"\n✅ Alias 'champion' asignado exitosamente a la versión {target_version.version}")
        return True

    except Exception as e:
        print(f"\n❌ Error asignando alias: {e}")
        return False


def list_all_versions():
    """Muestra todas las versiones del modelo con sus aliases."""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    print(f"\n{'='*60}")
    print(f"📋 VERSIONES DE '{MODEL_NAME}'")
    print(f"{'='*60}")

    for v in versions:
        aliases_str = ", ".join(v.aliases) if v.aliases else "Ninguno"
        print(f"  v{v.version} | Stage: {v.current_stage} | "
              f"Aliases: {aliases_str} | Run ID: {v.run_id}")

    print(f"{'='*60}")
    print(f"  Total versiones: {len(versions)}")


if __name__ == "__main__":
    # Si se pasa una versión como argumento, usarla
    target_version = sys.argv[1] if len(sys.argv) > 1 else None

    # Mostrar estado actual
    list_all_versions()

    # Asignar alias
    success = assign_champion_alias(target_version)

    # Mostrar estado final
    if success:
        list_all_versions()

    sys.exit(0 if success else 1)