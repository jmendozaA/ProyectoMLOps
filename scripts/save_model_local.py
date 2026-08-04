"""
Guarda el mejor modelo entrenado de forma local para empaquetarlo en Docker.
Ejecutar DESPUÉS de train.py y register_model.py
"""
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_NAME, PROCESSED_DATA_DIR

def save_best_model_locally():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        print("❌ Experimento no encontrado. Ejecuta train.py primero.")
        return
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse ASC"],
        max_results=1
    )
    
    if runs.empty:
        print("❌ No se encontraron runs.")
        return
    
    best_run_id = runs.iloc[0]["run_id"]
    print(f"✅ Mejor run: {best_run_id}")
    
    # Cargar modelo desde MLflow
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Crear directorio model/
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)
    
    # Guardar modelo
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Modelo guardado en: {model_path}")
    
    # Copiar preprocesador
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        import shutil
        dest = model_dir / "preprocessor.pkl"
        shutil.copy(preprocessor_path, dest)
        print(f"✅ Preprocesador copiado a: {dest}")
    
    print("\n✅ Modelo listo para empaquetar en Docker.")

if __name__ == "__main__":
    save_best_model_locally()