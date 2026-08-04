"""
Detección de Concept Drift
Detecta cambios en la relación entre variables de entrada
y la variable objetivo (Exam_Score).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional
import logging
from src.config import DRIFT_CONCEPT_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ConceptDriftDetector:
    """
    Detector de Concept Drift.
    Monitorea la degradación del rendimiento del modelo
    a lo largo del tiempo para detectar cambios en el concepto.
    """
    def __init__(
        self,
        baseline_rmse: float,
        baseline_mae: float,
        baseline_r2: float,
        threshold: float = DRIFT_CONCEPT_THRESHOLD
    ):
        self.baseline_rmse = baseline_rmse
        self.baseline_mae = baseline_mae
        self.baseline_r2 = baseline_r2
        self.threshold = threshold
        self.history = []

    def evaluate_batch(
        self,
        model,
        X_batch: pd.DataFrame,
        y_batch: pd.Series,
        batch_name: str = "batch"
    ) -> Dict:
        """
        Evalúa el modelo en un batch de datos y compara
        con las métricas baseline.
        """
        # Predicciones
        y_pred = model.predict(X_batch)
        
        # Métricas actuales
        current_rmse = np.sqrt(mean_squared_error(y_batch, y_pred))
        current_mae = mean_absolute_error(y_batch, y_pred)
        current_r2 = r2_score(y_batch, y_pred)
        
        # Degradación relativa
        rmse_degradation = (current_rmse - self.baseline_rmse) / self.baseline_rmse
        mae_degradation = (current_mae - self.baseline_mae) / self.baseline_mae
        r2_degradation = (self.baseline_r2 - current_r2) / self.baseline_r2
        
        # Detección de drift
        drift_detected = (
            rmse_degradation > self.threshold or
            mae_degradation > self.threshold or
            r2_degradation > self.threshold
        )
        
        result = {
            "batch_name": batch_name,
            "n_samples": len(y_batch),
            "current_rmse": round(current_rmse, 4),
            "current_mae": round(current_mae, 4),
            "current_r2": round(current_r2, 4),
            "baseline_rmse": round(self.baseline_rmse, 4),
            "baseline_mae": round(self.baseline_mae, 4),
            "baseline_r2": round(self.baseline_r2, 4),
            "rmse_degradation": round(rmse_degradation, 4),
            "mae_degradation": round(mae_degradation, 4),
            "r2_degradation": round(r2_degradation, 4),
            "drift_detected": drift_detected,
            "threshold": self.threshold,
        }
        
        self.history.append(result)
        
        status = "⚠️ DRIFT" if drift_detected else "✅ OK"
        logger.info(f"[{batch_name}] {status} | RMSE: {current_rmse:.4f} "
                     f"(deg: {rmse_degradation*100:.1f}%) | "
                     f"R²: {current_r2:.4f}")
        
        return result

    def detect_concept_drift_simulation(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "Exam_Score"
    ) -> Dict:
        """
        Detecta concept drift simulando un cambio en la relación
        entrada-salida. Compara la distribución de residuos.
        """
        X_ref = reference_data.drop(columns=[target_column])
        y_ref = reference_data[target_column]
        X_cur = current_data.drop(columns=[target_column])
        y_cur = current_data[target_column]
        
        # Calcular correlaciones por feature
        ref_correlations = {}
        cur_correlations = {}
        
        for col in X_ref.select_dtypes(include=[np.number]).columns:
            if col in X_cur.columns:
                ref_correlations[col] = X_ref[col].corr(y_ref)
                cur_correlations[col] = X_cur[col].corr(y_cur)
        
        # Detectar cambios en correlaciones
        correlation_changes = {}
        for col in ref_correlations:
            if col in cur_correlations:
                change = abs(cur_correlations[col] - ref_correlations[col])
                correlation_changes[col] = {
                    "ref_corr": round(ref_correlations[col], 4),
                    "cur_corr": round(cur_correlations[col], 4),
                    "change": round(change, 4),
                    "significant_change": change > 0.1,
                }
        
        n_significant = sum(1 for v in correlation_changes.values() 
                           if v["significant_change"])
        
        result = {
            "concept_drift_detected": n_significant > 2,
            "features_with_significant_change": n_significant,
            "total_features_analyzed": len(correlation_changes),
            "correlation_changes": correlation_changes,
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("CONCEPT DRIFT ANALYSIS")
        logger.info(f"{'='*50}")
        logger.info(f"  Concept drift detectado: "
                     f"{'⚠️ SÍ' if result['concept_drift_detected'] else '✅ NO'}")
        logger.info(f"  Features con cambio significativo: "
                     f"{n_significant}/{len(correlation_changes)}")
        logger.info(f"{'='*50}")
        
        return result

    def get_history_df(self) -> pd.DataFrame:
        """Retorna el historial de evaluaciones como DataFrame."""
        return pd.DataFrame(self.history)

    def get_drift_trend(self) -> Dict:
        """Analiza la tendencia de drift en el historial."""
        if not self.history:
            return {"error": "No hay datos en el historial"}
        df = pd.DataFrame(self.history)
        return {
            "total_batches": len(df),
            "batches_with_drift": df["drift_detected"].sum(),
            "avg_rmse_degradation": round(df["rmse_degradation"].mean(), 4),
            "max_rmse_degradation": round(df["rmse_degradation"].max(), 4),
            "avg_r2_degradation": round(df["r2_degradation"].mean(), 4),
            "trend": "worsening" if df["rmse_degradation"].mean() > 0 else "stable",
        }

    # =====================================================================
    # NUEVO MÉTODO AGREGADO PARA CUMPLIR CON EL CRITERIO DE REENTRENAMIENTO
    # =====================================================================
    def check_sustained_drift(self, consecutive_threshold: int = 3) -> dict:
        """
        Verifica si el drift se ha mantenido de forma sostenida.
        Criterio de reentrenamiento: drift sostenido durante N lotes consecutivos.
        """
        if not self.history:
            return {"sustained": False, "consecutive_count": 0}
        
        consecutive_count = 0
        # Recorremos el historial de atrás hacia adelante
        for batch in reversed(self.history):
            if batch.get("drift_detected", False):
                consecutive_count += 1
            else:
                break  # Se rompió la racha
        
        sustained = consecutive_count >= consecutive_threshold
        
        if sustained:
            logger.warning(
                f"🚨 CRITERIO DE REENTRENAMIENTO ALCANZADO: "
                f"Drift sostenido por {consecutive_count} lotes consecutivos "
                f"(Umbral: {consecutive_threshold})."
            )
        
        return {
            "sustained": sustained,
            "consecutive_count": consecutive_count,
            "threshold": consecutive_threshold
        }