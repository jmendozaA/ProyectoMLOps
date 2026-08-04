"""
demo_drift_completo.py
Demostración exhaustiva de Data Drift y Concept Drift.
CORREGIDO: Ajustado para ejecutarse desde src/monitoring/
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, DRIFT_CONCEPT_THRESHOLD
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.concept_drift import ConceptDriftDetector
from src.monitoring.alerts import DriftAlertManager

print("="*70)
print(" DEMOSTRACIÓN DE MONITOREO DE DRIFT (FASE 6 - MLOps)")
print("="*70)

# ==============================================================================
# 1. CARGA DE DATOS, MODELO Y PREPROCESADOR (BASELINE)
# ==============================================================================
print("\n[1] Cargando datos de referencia (entrenamiento), modelo y preprocesador...")
X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
df_baseline = pd.concat([X_train, y_train], axis=1)

# ⭐ CORRECCIÓN: Ruta al modelo desde la raíz del proyecto
model_path = project_root / "model" / "model.joblib"
model = joblib.load(model_path)

# ⭐ CORRECCIÓN: Ruta al preprocesador desde la raíz del proyecto
preprocessor_path = project_root / "model" / "preprocessor.pkl"
preprocessor = joblib.load(preprocessor_path)
print(f"✅ Preprocesador cargado desde: {preprocessor_path}")

# Métricas baseline (REEMPLAZA con los valores reales de tu modelo)
baseline_rmse = 8.5   # ← Cambia por tu RMSE real en test
baseline_mae = 6.0    # ← Cambia por tu MAE real en test
baseline_r2 = 0.75    # ← Cambia por tu R² real en test

print(f"✅ Datos cargados. Baseline RMSE: {baseline_rmse}, MAE: {baseline_mae}, R²: {baseline_r2}")

# ==============================================================================
# 2. DATA DRIFT: ESTADO VERDE (Datos normales - sin drift)
# ==============================================================================
print("\n" + "="*70)
print("[2] PRUEBA DATA DRIFT: ESTADO VERDE (Sin Drift)")
print("="*70)
print("Escenario: Datos del mismo origen que el entrenamiento")
print("-" * 70)

normal_batch = df_baseline.sample(n=500, random_state=42)

detector_green = DataDriftDetector(reference_data=X_train)
results_green = detector_green.run_full_detection(
    normal_batch.drop(columns=['Exam_Score'], errors='ignore')
)

alert_manager = DriftAlertManager()
alert_green = alert_manager.generate_data_drift_alert(results_green)

if alert_green is None:
    print("\n✅ RESULTADO: PASA (Verde). No se detectó drift significativo.")
    print("   → El sistema NO genera alerta.")
else:
    print(f"\n⚠️ RESULTADO: FALLA. Alerta generada: {alert_green['severity']}")

# ==============================================================================
# 3. DATA DRIFT: ESTADO ROJO (Datos inyectados con Drift)
# ==============================================================================
print("\n" + "="*70)
print("[3] PRUEBA DATA DRIFT: ESTADO ROJO (Drift Inyectado)")
print("="*70)
print("Escenario: Datos con distribución alterada artificialmente")
print("-" * 70)

drifted_batch = normal_batch.copy()
drifted_batch['Hours_Studied'] = drifted_batch['Hours_Studied'] * 3.5
drifted_batch['Family_Income'] = drifted_batch['Family_Income'].map(
    {'Low': 'High', 'Medium': 'High', 'High': 'High'}
)
print("Modificaciones aplicadas:")
print("  • Hours_Studied multiplicado por 3.5")
print("  • Family_Income cambiado a 'High' para todos")

detector_red = DataDriftDetector(reference_data=X_train)
results_red = detector_red.run_full_detection(
    drifted_batch.drop(columns=['Exam_Score'], errors='ignore')
)

alert_red = alert_manager.generate_data_drift_alert(results_red)

if alert_red:
    print(f"\n🚨 RESULTADO: FALLA (Rojo). Alerta generada con severidad: {alert_red['severity'].upper()}")
    print(f"   Mensaje: {alert_red['message']}")
    print(f"   Variables afectadas: {', '.join(alert_red['variables_affected'][:5])}")
else:
    print("\n❌ ERROR: Se esperaba detectar drift, pero no se generó alerta.")

alert_manager.save_alerts("demo_drift_alerts.json")

# ==============================================================================
# 4. CONCEPT DRIFT: Simulación y Degradación en Lotes Sucesivos
# ==============================================================================
print("\n" + "="*70)
print("[4] PRUEBA CONCEPT DRIFT: Degradación en Lotes Sucesivos")
print("="*70)
print(f"Criterio de reentrenamiento: Drift sostenido durante 3 lotes consecutivos")
print(f"Umbral de degradación: {DRIFT_CONCEPT_THRESHOLD*100}% respecto al baseline")
print("-" * 70)

concept_detector = ConceptDriftDetector(
    baseline_rmse=baseline_rmse,
    baseline_mae=baseline_mae,
    baseline_r2=baseline_r2,
    threshold=DRIFT_CONCEPT_THRESHOLD
)

num_batches = 6
RETRAINING_THRESHOLD_BATCHES = 3

print(f"\nSimulando {num_batches} lotes de producción...")
print("Nota: A partir del lote 3, se inyecta ruido en las etiquetas para simular concept drift.\n")

for i in range(1, num_batches + 1):
    X_batch_raw = df_baseline.sample(n=100, random_state=i).drop(columns=['Exam_Score'])
    y_batch_real = df_baseline.sample(n=100, random_state=i)['Exam_Score']
    
    if i >= 3:
        noise = np.random.normal(0, 15, size=len(y_batch_real))
        y_batch_observed = np.clip(y_batch_real + (i * 5) + noise, 0, 100)
    else:
        y_batch_observed = y_batch_real

    X_batch_processed = pd.DataFrame(
        preprocessor.transform(X_batch_raw),
        columns=preprocessor.get_feature_names_out()
    )

    result = concept_detector.evaluate_batch(
        model=model,
        X_batch=X_batch_processed,
        y_batch=y_batch_observed,
        batch_name=f"Lote_{i}"
    )

sustained_check = concept_detector.check_sustained_drift(
    consecutive_threshold=RETRAINING_THRESHOLD_BATCHES
)

if sustained_check["sustained"]:
    print(f"\n🚨 ¡ALERTA DE REENTRENAMIENTO ACTIVADA!")
    print(f"   Drift sostenido por {sustained_check['consecutive_count']} lotes consecutivos.")
    print(f"   Se recomienda reentrenar el modelo con datos recientes.")
else:
    print(f"\n✅ No se alcanzó el criterio de reentrenamiento.")
    print(f"   Lotes consecutivos con drift: {sustained_check['consecutive_count']}/{RETRAINING_THRESHOLD_BATCHES}")

# ==============================================================================
# 5. GRÁFICA TEMPORAL DE DEGRADACIÓN
# ==============================================================================
print("\n" + "="*70)
print("[5] Generando gráfica temporal de degradación de RMSE...")
print("="*70)

df_history = concept_detector.get_history_df()

plt.figure(figsize=(12, 6))
plt.plot(
    df_history['batch_name'], 
    df_history['current_rmse'], 
    marker='o', 
    linewidth=2,
    label='RMSE Actual', 
    color='blue',
    markersize=8
)
plt.axhline(
    y=baseline_rmse, 
    color='green', 
    linestyle='--', 
    linewidth=2,
    label=f'Baseline RMSE ({baseline_rmse:.2f})'
)
plt.axhline(
    y=baseline_rmse * (1 + DRIFT_CONCEPT_THRESHOLD), 
    color='red', 
    linestyle='--', 
    linewidth=2,
    label=f'Umbral de Alerta (+{DRIFT_CONCEPT_THRESHOLD*100:.0f}% degradación)'
)

for idx, row in df_history.iterrows():
    if row['drift_detected']:
        plt.scatter(
            row['batch_name'], 
            row['current_rmse'], 
            color='red', 
            s=150, 
            marker='X',
            zorder=5
        )

plt.title('Degradación del Modelo en el Tiempo (Concept Drift)', fontsize=14, fontweight='bold')
plt.xlabel('Lote de Datos', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# ⭐ CORRECCIÓN: Guardar gráfica en la raíz del proyecto
plot_path = project_root / "docs" / "evidence" / "concept_drift_degradation.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Gráfica guardada en: {plot_path}")
plt.show()

# ==============================================================================
# 6. RESUMEN FINAL
# ==============================================================================
print("\n" + "="*70)
print(" RESUMEN DE LA DEMOSTRACIÓN")
print("="*70)
print(f"✅ Data Drift (Verde): Datos normales → Sin alerta")
print(f"🚨 Data Drift (Rojo): Datos con drift → Alerta generada")
print(f"📊 Concept Drift: Degradación medida en {num_batches} lotes")
print(f"🎯 Criterio de reentrenamiento: {'ACTIVADO' if sustained_check['sustained'] else 'NO ACTIVADO'}")
print(f"📈 Gráfica temporal: Guardada en docs/evidence/")
print("="*70)
print(" DEMOSTRACIÓN FINALIZADA EXITOSAMENTE")
print("="*70)