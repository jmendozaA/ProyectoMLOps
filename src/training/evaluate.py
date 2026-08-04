"""
Evaluación detallada del mejor modelo
=====================================================
Genera reportes de evaluación con métricas, gráficos y análisis.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import logging
import os
import sys
from pathlib import Path
import joblib  

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR, SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data():
    """Carga los datos de test y aplica el preprocesador."""
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    
    # Cargar y aplicar el preprocesador
    preprocessor = joblib.load(PROCESSED_DATA_DIR / "preprocessor.pkl")
    feature_names = preprocessor.get_feature_names_out()
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)
    
    return X_test, y_test


def get_best_model_from_mlflow():
    """Obtiene el mejor modelo desde MLflow (menor RMSE en test)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Buscar el experimento
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' no encontrado")

    # Buscar runs del experimento
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse ASC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("No se encontraron runs en el experimento")

    best_run_id = runs.iloc[0]["run_id"]
    logger.info(f"Mejor run encontrado: {best_run_id}")
    logger.info(f"  Test RMSE: {runs.iloc[0]['metrics.test_rmse']:.4f}")

    # Cargar modelo
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model, best_run_id, runs.iloc[0]


def compute_comprehensive_metrics(y_true, y_pred) -> dict:
    """Calcula métricas completas de evaluación."""
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "max_error": np.max(np.abs(y_true - y_pred)),
        "median_ae": np.median(np.abs(y_true - y_pred)),
        "std_residuals": np.std(y_true - y_pred),
    }
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, save_path: str = None):
    """Gráfico de valores reales vs predichos."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", s=30)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            "r--", linewidth=2, label="Línea perfecta")

    ax.set_xlabel("Valores Reales (Exam_Score)", fontsize=12)
    ax.set_ylabel("Predicciones", fontsize=12)
    ax.set_title("Actual vs Predicho", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_residuals(y_true, y_pred, save_path: str = None):
    """Gráfico de residuos."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuos vs predichos
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors="k", s=30)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicciones", fontsize=12)
    axes[0].set_ylabel("Residuos", fontsize=12)
    axes[0].set_title("Residuos vs Predicciones", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Distribución de residuos
    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residuos", fontsize=12)
    axes[1].set_ylabel("Frecuencia", fontsize=12)
    axes[1].set_title("Distribución de Residuos", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_error_distribution(y_true, y_pred, save_path: str = None):
    """Distribución del error absoluto."""
    abs_errors = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(abs_errors, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=np.mean(abs_errors), color="r", linestyle="--",
               linewidth=2, label=f"Media: {np.mean(abs_errors):.2f}")
    ax.axvline(x=np.median(abs_errors), color="g", linestyle="--",
               linewidth=2, label=f"Mediana: {np.median(abs_errors):.2f}")

    ax.set_xlabel("Error Absoluto", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución del Error Absoluto", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def run_evaluation():
    """Ejecuta la evaluación completa del mejor modelo."""
    logger.info("=" * 60)
    logger.info("EVALUACIÓN DEL MEJOR MODELO")
    logger.info("=" * 60)

    # Cargar datos de test
    X_test, y_test = load_test_data()

    # Obtener mejor modelo desde MLflow
    model, best_run_id, best_run_info = get_best_model_from_mlflow()

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    metrics = compute_comprehensive_metrics(y_test.values, y_pred)

    logger.info("\n📊 MÉTRICAS DE EVALUACIÓN:")
    logger.info("-" * 40)
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name:20s}: {value:.4f}")

    # Generar gráficos
    plots_dir = PROCESSED_DATA_DIR / "evaluation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n📈 Generando gráficos...")

    plot_actual_vs_predicted(
        y_test.values, y_pred,
        save_path=str(plots_dir / "actual_vs_predicted.png")
    )
    logger.info("  ✅ actual_vs_predicted.png")

    plot_residuals(
        y_test.values, y_pred,
        save_path=str(plots_dir / "residuals.png")
    )
    logger.info("  ✅ residuals.png")

    plot_error_distribution(
        y_test.values, y_pred,
        save_path=str(plots_dir / "error_distribution.png")
    )
    logger.info("  ✅ error_distribution.png")

    # Guardar métricas como CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(plots_dir / "metrics.csv", index=False)
    logger.info("  ✅ metrics.csv")

    # Guardar predicciones
    predictions_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "error": y_test.values - y_pred,
        "abs_error": np.abs(y_test.values - y_pred)
    })
    predictions_df.to_csv(plots_dir / "predictions.csv", index=False)
    logger.info("  ✅ predictions.csv")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUACIÓN COMPLETADA EXITOSAMENTE")
    logger.info(f"Resultados guardados en: {plots_dir}")
    logger.info("=" * 60)

    return metrics, best_run_id


if __name__ == "__main__":
    metrics, run_id = run_evaluation()