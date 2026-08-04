"""
Detección de Data Drift
=========================================
Detecta cambios en la distribución de las variables de entrada
entre el dataset de referencia (entrenamiento) y datos nuevos.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

from src.config import (
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    DRIFT_KS_THRESHOLD, DRIFT_PSI_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detector de Data Drift usando pruebas estadísticas.
    
    Métodos:
    - Kolmogorov-Smirnov para variables numéricas
    - Chi-cuadrado para variables categóricas
    - PSI (Population Stability Index) para ambas
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        ks_threshold: float = DRIFT_KS_THRESHOLD,
        psi_threshold: float = DRIFT_PSI_THRESHOLD
    ):
        self.reference_data = reference_data
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.results = {}

    def detect_ks_test(
        self,
        current_data: pd.DataFrame,
        columns: List[str] = None
    ) -> Dict:
        """
        Prueba de Kolmogorov-Smirnov para variables numéricas.
        
        H0: Las dos distribuciones son iguales.
        Si p-value < threshold → hay drift.
        """
        if columns is None:
            columns = [c for c in NUMERICAL_COLUMNS if c in current_data.columns]

        results = {}

        for col in columns:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue

            ref_values = self.reference_data[col].dropna()
            cur_values = current_data[col].dropna()

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)

            results[col] = {
                "test": "Kolmogorov-Smirnov",
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drift_detected": p_value < self.ks_threshold,
                "threshold": self.ks_threshold,
                "ref_mean": round(ref_values.mean(), 2),
                "cur_mean": round(cur_values.mean(), 2),
                "mean_diff": round(cur_values.mean() - ref_values.mean(), 2),
                "ref_std": round(ref_values.std(), 2),
                "cur_std": round(cur_values.std(), 2),
            }

        self.results["ks_test"] = results
        return results

    def detect_chi_square(
        self,
        current_data: pd.DataFrame,
        columns: List[str] = None
    ) -> Dict:
        """
        Prueba Chi-cuadrado para variables categóricas.
        
        Compara las frecuencias observadas vs esperadas.
        """
        if columns is None:
            columns = [c for c in CATEGORICAL_COLUMNS if c in current_data.columns]

        results = {}

        for col in columns:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue

            try:
                # Obtener categorías de referencia
                ref_counts = self.reference_data[col].value_counts(normalize=True)
                cur_counts = current_data[col].value_counts()

                # Alinear categorías
                all_categories = ref_counts.index.union(cur_counts.index)
                ref_freq = ref_counts.reindex(all_categories, fill_value=0)
                cur_freq = cur_counts.reindex(all_categories, fill_value=0)

                # Chi-cuadrado
                chi2_stat, p_value = stats.chisquare(
                    cur_freq.values,
                    f_exp=ref_freq.values * cur_freq.sum()
                )

                results[col] = {
                    "test": "Chi-Square",
                    "chi2_statistic": round(chi2_stat, 4),
                    "p_value": round(p_value, 6),
                    "drift_detected": p_value < self.ks_threshold,
                    "threshold": self.ks_threshold,
                    "n_categories": len(all_categories),
                }
            except Exception as e:
                logger.warning(f"Error en Chi-Square para '{col}': {e}")
                results[col] = {"test": "Chi-Square", "error": str(e)}

        self.results["chi_square"] = results
        return results

    def calculate_psi(
        self,
        current_data: pd.DataFrame,
        columns: List[str] = None,
        n_bins: int = 10
    ) -> Dict:
        """
        Population Stability Index (PSI).
        
        PSI < 0.1: Sin cambio significativo
        0.1 <= PSI < 0.2: Cambio moderado
        PSI >= 0.2: Cambio significativo (drift)
        """
        if columns is None:
            columns = [c for c in NUMERICAL_COLUMNS if c in current_data.columns]

        results = {}

        for col in columns:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue

            try:
                ref_values = self.reference_data[col].dropna()
                cur_values = current_data[col].dropna()

                # Crear bins basados en referencia
                bins = np.linspace(ref_values.min(), ref_values.max(), n_bins + 1)

                ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
                cur_hist, _ = np.histogram(cur_values, bins=bins, density=True)

                # Evitar división por cero
                ref_hist = np.where(ref_hist == 0, 0.0001, ref_hist)
                cur_hist = np.where(cur_hist == 0, 0.0001, cur_hist)

                # Calcular PSI
                psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

                # Interpretación
                if psi < 0.1:
                    interpretation = "No significant change"
                elif psi < 0.2:
                    interpretation = "Moderate change"
                else:
                    interpretation = "Significant change (DRIFT)"

                results[col] = {
                    "test": "PSI",
                    "psi_value": round(psi, 4),
                    "drift_detected": psi >= self.psi_threshold,
                    "threshold": self.psi_threshold,
                    "interpretation": interpretation,
                }
            except Exception as e:
                logger.warning(f"Error en PSI para '{col}': {e}")
                results[col] = {"test": "PSI", "error": str(e)}

        self.results["psi"] = results
        return results

    def run_full_detection(self, current_data: pd.DataFrame) -> Dict:
        """
        Ejecuta todas las pruebas de detección de drift.
        """
        logger.info("🔍 Ejecutando detección completa de Data Drift...")

        # 1. KS Test (numéricas)
        ks_results = self.detect_ks_test(current_data)
        n_ks_drift = sum(1 for v in ks_results.values() if v.get("drift_detected", False))
        logger.info(f"  KS Test: {n_ks_drift}/{len(ks_results)} variables con drift")

        # 2. Chi-Square (categóricas)
        chi2_results = self.detect_chi_square(current_data)
        n_chi2_drift = sum(1 for v in chi2_results.values() if v.get("drift_detected", False))
        logger.info(f"  Chi-Square: {n_chi2_drift}/{len(chi2_results)} variables con drift")

        # 3. PSI (numéricas)
        psi_results = self.calculate_psi(current_data)
        n_psi_drift = sum(1 for v in psi_results.values() if v.get("drift_detected", False))
        logger.info(f"  PSI: {n_psi_drift}/{len(psi_results)} variables con drift")

        # Resumen
        total_drift = n_ks_drift + n_chi2_drift + n_psi_drift
        total_tests = len(ks_results) + len(chi2_results) + len(psi_results)

        summary = {
            "overall_drift_detected": total_drift > 0,
            "total_tests": total_tests,
            "total_with_drift": total_drift,
            "drift_percentage": round(total_drift / max(total_tests, 1) * 100, 1),
            "ks_drift_count": n_ks_drift,
            "chi2_drift_count": n_chi2_drift,
            "psi_drift_count": n_psi_drift,
        }

        self.results["summary"] = summary

        logger.info(f"\n{'='*50}")
        logger.info(f"RESUMEN DATA DRIFT")
        logger.info(f"{'='*50}")
        logger.info(f"  Drift detectado: {'⚠️ SÍ' if summary['overall_drift_detected'] else '✅ NO'}")
        logger.info(f"  Variables con drift: {total_drift}/{total_tests} ({summary['drift_percentage']}%)")
        logger.info(f"{'='*50}")

        return self.results

    def get_drift_report(self) -> pd.DataFrame:
        """Genera un reporte tabular de drift."""
        rows = []

        # KS Test results
        for col, res in self.results.get("ks_test", {}).items():
            rows.append({
                "variable": col,
                "test": "KS",
                "statistic": res.get("ks_statistic"),
                "p_value": res.get("p_value"),
                "drift_detected": res.get("drift_detected"),
            })

        # Chi-Square results
        for col, res in self.results.get("chi_square", {}).items():
            rows.append({
                "variable": col,
                "test": "Chi2",
                "statistic": res.get("chi2_statistic"),
                "p_value": res.get("p_value"),
                "drift_detected": res.get("drift_detected"),
            })

        # PSI results
        for col, res in self.results.get("psi", {}).items():
            rows.append({
                "variable": col,
                "test": "PSI",
                "statistic": res.get("psi_value"),
                "p_value": None,
                "drift_detected": res.get("drift_detected"),
            })

        return pd.DataFrame(rows)