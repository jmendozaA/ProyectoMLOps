"""
alerts.py - Sistema de alertas para drift
==========================================
Gestiona alertas cuando se detecta drift en datos o concepto.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

from src.config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DriftAlertManager:
    """
    Gestor de alertas de drift.
    Genera, almacena y consulta alertas.
    """

    def __init__(self, alerts_dir: Path = None):
        self.alerts_dir = alerts_dir or PROCESSED_DATA_DIR / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.alerts: List[Dict] = []

    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Dict = None,
        variables_affected: List[str] = None
    ) -> Dict:
        """Crea una nueva alerta de drift."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,  # "data_drift" o "concept_drift"
            "severity": severity,      # "low", "medium", "high", "critical"
            "message": message,
            "details": details or {},
            "variables_affected": variables_affected or [],
            "status": "active",
        }

        self.alerts.append(alert)

        # Log con emoji según severidad
        severity_emoji = {
            "low": "🟡", "medium": "🟠",
            "high": "🔴", "critical": "🚨"
        }
        emoji = severity_emoji.get(severity, "ℹ️")

        logger.info(f"{emoji} ALERTA [{severity.upper()}] {alert_type}: {message}")

        return alert

    def generate_data_drift_alert(self, drift_results: Dict) -> Optional[Dict]:
        """Genera alerta basada en resultados de data drift."""
        summary = drift_results.get("summary", {})

        if not summary.get("overall_drift_detected", False):
            return None

        drift_pct = summary.get("drift_percentage", 0)

        # Determinar severidad
        if drift_pct > 50:
            severity = "critical"
        elif drift_pct > 30:
            severity = "high"
        elif drift_pct > 15:
            severity = "medium"
        else:
            severity = "low"

        # Variables afectadas
        affected = []
        for test_name in ["ks_test", "chi_square", "psi"]:
            test_results = drift_results.get(test_name, {})
            for var, res in test_results.items():
                if res.get("drift_detected", False):
                    affected.append(var)

        message = (
            f"Data drift detectado en {summary.get('total_with_drift', 0)} "
            f"variables ({drift_pct}%). "
            f"Variables afectadas: {', '.join(affected[:5])}"
        )

        return self.create_alert(
            alert_type="data_drift",
            severity=severity,
            message=message,
            details=summary,
            variables_affected=list(set(affected))
        )

    def generate_concept_drift_alert(self, concept_results: Dict) -> Optional[Dict]:
        """Genera alerta basada en resultados de concept drift."""
        if not concept_results.get("concept_drift_detected", False):
            return None

        n_features = concept_results.get("features_with_significant_change", 0)

        severity = "high" if n_features > 5 else "medium"

        message = (
            f"Concept drift detectado. {n_features} features muestran "
            f"cambios significativos en su correlación con el target."
        )

        return self.create_alert(
            alert_type="concept_drift",
            severity=severity,
            message=message,
            details=concept_results,
        )

    def save_alerts(self, filename: str = None) -> str:
        """Guarda las alertas en un archivo JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_alerts_{timestamp}.json"

        filepath = self.alerts_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.alerts, f, indent=2, default=str)

        logger.info(f"💾 {len(self.alerts)} alertas guardadas en: {filepath}")

        return str(filepath)

    def get_active_alerts(self) -> List[Dict]:
        """Retorna alertas activas."""
        return [a for a in self.alerts if a["status"] == "active"]

    def get_alerts_by_type(self, alert_type: str) -> List[Dict]:
        """Retorna alertas filtradas por tipo."""
        return [a for a in self.alerts if a["alert_type"] == alert_type]

    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        """Retorna alertas filtradas por severidad."""
        return [a for a in self.alerts if a["severity"] == severity]

    def summary(self) -> Dict:
        """Resumen de todas las alertas."""
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.get_active_alerts()),
            "by_type": {
                "data_drift": len(self.get_alerts_by_type("data_drift")),
                "concept_drift": len(self.get_alerts_by_type("concept_drift")),
            },
            "by_severity": {
                "critical": len(self.get_alerts_by_severity("critical")),
                "high": len(self.get_alerts_by_severity("high")),
                "medium": len(self.get_alerts_by_severity("medium")),
                "low": len(self.get_alerts_by_severity("low")),
            }
        }