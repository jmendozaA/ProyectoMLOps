"""
Entrenamiento de modelos con MLflow
Entrena múltiples modelos con diferentes hiperparámetros
y registra experimentos en MLflow.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import tempfile
import joblib
import logging
import warnings
from pathlib import Path
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Agregar el directorio raíz al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    SEED, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    ALL_PARAMS_GRID, MODEL_NAME, PROCESSED_DATA_DIR
)
from src.training.preprocess import set_seed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data():
    """Carga los datos procesados y aplica el preprocesador guardado."""
    logger.info("Cargando datos procesados y aplicando transformaciones...")
    
    # 1. Cargar los DataFrames crudos guardados
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    
    # 2. Cargar el preprocesador guardado por preprocess.py
    preprocessor = joblib.load(PROCESSED_DATA_DIR / "preprocessor.pkl")
    
    # 3. Aplicar la transformación (convierte texto a números y escala)
    feature_names = preprocessor.get_feature_names_out()
    
    X_train_processed = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names)
    X_val_processed = pd.DataFrame(preprocessor.transform(X_val), columns=feature_names)
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)
    
    logger.info(f"  X_train procesado: {X_train_processed.shape}")
    logger.info(f"  X_val procesado:   {X_val_processed.shape}")
    logger.info(f"  X_test procesado:  {X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test


def create_model(params: dict):
    """Crea un modelo basado en los parámetros."""
    model_type = params["model_type"]
    
    if model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            random_state=params["random_state"],
            objective="reg:squarederror"
        )
    elif model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=params["random_state"]
        )
    elif model_type == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"]
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")


def evaluate_model(model, X, y, dataset_name: str = "test") -> dict:
    """Evalúa el modelo y retorna métricas."""
    y_pred = model.predict(X)
    metrics = {
        f"{dataset_name}_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        f"{dataset_name}_mae": float(mean_absolute_error(y, y_pred)),
        f"{dataset_name}_r2": float(r2_score(y, y_pred)),
        f"{dataset_name}_mse": float(mean_squared_error(y, y_pred)),
    }
    return metrics, y_pred


def log_feature_importance(model, feature_names: list, run_id: str):
    """Registra la importancia de features como artifact."""
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Usar directorio temporal compatible con Windows y Linux
        temp_dir = tempfile.gettempdir()
        importance_path = os.path.join(temp_dir, f"feature_importance_{run_id}.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "feature_importance")
        
        # Log top 5 como métricas
        for i, row in importance_df.head(5).iterrows():
            mlflow.log_metric(f"top_feat_{i+1}_importance", float(row["importance"]))


def train_all_models():
    """
    Entrena todos los modelos definidos en el grid de hiperparámetros.
    Cada ejecución se registra en MLflow.
    """
    set_seed(SEED)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    
    logger.info("=" * 70)
    logger.info(f"INICIANDO ENTRENAMIENTO DE {len(ALL_PARAMS_GRID)} MODELOS")
    logger.info("=" * 70)
    
    best_rmse = float("inf")
    best_run_id = None
    best_model = None
    best_params = {"model_type": "Ninguno"}
    
    for idx, params in enumerate(ALL_PARAMS_GRID, 1):
        model_type = params["model_type"]
        run_name = f"{model_type}_run_{idx}"
        
        logger.info(f"\n[{idx}/{len(ALL_PARAMS_GRID)}] Entrenando: {run_name}")
        logger.info(f"  Parámetros: n_estimators={params['n_estimators']}, "
                    f"max_depth={params.get('max_depth', 'N/A')}, "
                    f"lr={params.get('learning_rate', 'N/A')}")
        
        with mlflow.start_run(run_name=run_name) as run:
            try:
                model = create_model(params)
                model.fit(X_train_full, y_train_full)
                
                test_metrics, y_pred_test = evaluate_model(model, X_test, y_test, "test")
                train_metrics, _ = evaluate_model(model, X_train_full, y_train_full, "train")
                
                mlflow.log_params(params)
                mlflow.log_metrics(test_metrics)
                mlflow.log_metrics(train_metrics)
                
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=MODEL_NAME,
                    input_example=X_test.iloc[:5]
                )
                
                log_feature_importance(model, X_train_full.columns.tolist(), run.info.run_id)
                
                # Guardar predicciones de muestra en carpeta temporal segura
                sample_preds = pd.DataFrame({
                    "actual": y_test.values[:20].tolist(),
                    "predicted": y_pred_test[:20].tolist()
                })
                temp_dir = tempfile.gettempdir()
                sample_path = os.path.join(temp_dir, "sample_predictions.csv")
                sample_preds.to_csv(sample_path, index=False)
                mlflow.log_artifact(sample_path, "predictions")
                
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("run_number", idx)
                mlflow.set_tag("seed", SEED)
                
                current_rmse = test_metrics["test_rmse"]
                logger.info(f"  ✅ Test RMSE: {current_rmse:.4f} | "
                            f"MAE: {test_metrics['test_mae']:.4f} | "
                            f"R²: {test_metrics['test_r2']:.4f}")
                
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_run_id = run.info.run_id
                    best_model = model
                    best_params = params
                    logger.info(f"  🏆 ¡Nuevo mejor modelo! RMSE: {best_rmse:.4f}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error en {run_name}: {str(e)}")
                mlflow.set_tag("error", str(e))
    
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN DEL ENTRENAMIENTO")
    logger.info("=" * 70)
    logger.info(f"  Total de modelos entrenados: {len(ALL_PARAMS_GRID)}")
    
    # Validación de seguridad para evitar el error de NoneType
    if best_model is not None:
        logger.info(f"  Mejor modelo: {best_params['model_type']}")
        logger.info(f"  Mejor Run ID: {best_run_id}")
        logger.info(f"  Mejor RMSE:   {best_rmse:.4f}")
        
        final_metrics, _ = evaluate_model(best_model, X_test, y_test)
        logger.info(f"  Mejor MAE:    {final_metrics['test_mae']:.4f}")
        logger.info(f"  Mejor R²:     {final_metrics['test_r2']:.4f}")
        logger.info(f"  MLflow UI:    {MLFLOW_TRACKING_URI}")
    else:
        logger.error("  ⚠️ Ningún modelo se entrenó exitosamente. Revisa los errores anteriores.")
        
    logger.info("=" * 70)
    
    return best_run_id, best_model, best_params


if __name__ == "__main__":
    best_run_id, best_model, best_params = train_all_models()