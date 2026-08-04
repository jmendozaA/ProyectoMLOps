"""
Configuración centralizada del proyecto
====================================================
Define constantes, rutas, semillas y parámetros del modelo.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_SCENARIOS_DIR = DATA_DIR / "drift_scenarios"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Crear directorios si no existen
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_SCENARIOS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
RAW_DATA_PATH = RAW_DATA_DIR / "StudentPerformanceFactors.csv"
TARGET_COLUMN = "Exam_Score"

# Columnas numéricas
NUMERICAL_COLUMNS = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity"
]

# Columnas categóricas
CATEGORICAL_COLUMNS = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]

# ============================================================
# REPRODUCIBILIDAD
# ============================================================
SEED = int(os.getenv("SEED", 42))
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # Proporción del train para validación

# ============================================================
# MLFLOW
# ============================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "student-performance-regression")
MODEL_NAME = os.getenv("MODEL_NAME", "student-performance-model")

# ============================================================
# HIPERPARÁMETROS POR MODELO
# ============================================================

# XGBoost - 5 configuraciones diferentes
XGB_PARAMS_GRID = [
    {
        "model_type": "XGBoost",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": SEED,
    },
    {
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "random_state": SEED,
    },
    {
        "model_type": "XGBoost",
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": SEED,
    },
    {
        "model_type": "XGBoost",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 1.0,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "random_state": SEED,
    },
    {
        "model_type": "XGBoost",
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 4,
        "random_state": SEED,
    },
]

# Random Forest - 5 configuraciones diferentes
RF_PARAMS_GRID = [
    {
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": SEED,
    },
    {
        "model_type": "RandomForest",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": SEED,
    },
    {
        "model_type": "RandomForest",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "log2",
        "random_state": SEED,
    },
    {
        "model_type": "RandomForest",
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "random_state": SEED,
    },
    {
        "model_type": "RandomForest",
        "n_estimators": 250,
        "max_depth": 25,
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": SEED,
    },
]

# Gradient Boosting - 5 configuraciones diferentes
GB_PARAMS_GRID = [
    {
        "model_type": "GradientBoosting",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "min_samples_split": 2,
        "random_state": SEED,
    },
    {
        "model_type": "GradientBoosting",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "min_samples_split": 5,
        "random_state": SEED,
    },
    {
        "model_type": "GradientBoosting",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 1.0,
        "min_samples_split": 3,
        "random_state": SEED,
    },
    {
        "model_type": "GradientBoosting",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "min_samples_split": 10,
        "random_state": SEED,
    },
    {
        "model_type": "GradientBoosting",
        "n_estimators": 250,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "min_samples_split": 4,
        "random_state": SEED,
    },
]

# Grid combinado de todos los modelos (15 ejecuciones totales)
ALL_PARAMS_GRID = XGB_PARAMS_GRID + RF_PARAMS_GRID + GB_PARAMS_GRID

# ============================================================
# UMBRALES DE DRIFT
# ============================================================
DRIFT_KS_THRESHOLD = 0.05       # p-value para Kolmogorov-Smirnov
DRIFT_PSI_THRESHOLD = 0.2       # PSI > 0.2 indica drift significativo
DRIFT_CONCEPT_THRESHOLD = 0.1   # 10% degradación en RMSE