"""
Preprocesamiento de datos
==========================================
Limpieza, transformación y división del dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import logging
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TARGET_COLUMN,
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS,
    SEED, TEST_SIZE, VAL_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla para reproducibilidad."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_data(path: str = None) -> pd.DataFrame:
    """Carga el dataset desde CSV."""
    path = path or str(RAW_DATA_PATH)
    logger.info(f"Cargando datos desde: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica del dataset:
    - Elimina duplicados
    - Maneja valores nulos
    - Elimina columnas con >50% de valores nulos
    """
    initial_shape = df.shape
    logger.info(f"Forma inicial: {initial_shape}")

    # Eliminar duplicados
    df = df.drop_duplicates()
    logger.info(f"Después de eliminar duplicados: {df.shape}")

    # Manejar valores nulos en columnas numéricas (imputar con mediana)
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputada columna numérica '{col}' con mediana={median_val:.2f}")

    # Manejar valores nulos en columnas categóricas (imputar con moda)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputada columna categórica '{col}' con moda='{mode_val}'")

    logger.info(f"Forma final después de limpieza: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """
    Divide el dataset en train, validation y test.
    Usa stratificación basada en cuantiles de la variable objetivo.
    """
    logger.info("Dividiendo datos en train/val/test...")

    X = df.drop(columns=[target])
    y = df[target]

    # Crear bins para estratificación
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')

    # Primera división: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_bins
    )

    # Segunda división: train vs val
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    y_train_val_bins = pd.qcut(y_train_val, q=4, labels=False, duplicates='drop')

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        random_state=SEED,
        stratify=y_train_val_bins
    )

    logger.info(f"  Train: {X_train.shape[0]} muestras")
    logger.info(f"  Val:   {X_val.shape[0]} muestras")
    logger.info(f"  Test:  {X_test.shape[0]} muestras")

    return X_train, X_val, X_test, y_train, y_val, y_test


from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Cambiar aquí

def build_preprocessor():
    """
    Construye el pipeline de preprocesamiento:
    - Escalado para numéricas
    - Codificación ordinal para categóricas
    """
    # Pipeline para variables numéricas
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Pipeline para variables categóricas
    categorical_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Combinar en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS)
        ],
        remainder="drop"
    )

    return preprocessor


def save_processed_data(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    preprocessor=None
) -> None:
    """Guarda los datos procesados en disco."""
    logger.info("Guardando datos procesados...")

    # Guardar DataFrames
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Guardar preprocesador
    if preprocessor is not None:
        joblib.dump(preprocessor, PROCESSED_DATA_DIR / "preprocessor.pkl")
        logger.info("Preprocesador guardado en: preprocessor.pkl")

    logger.info("Datos procesados guardados exitosamente.")


def run_preprocessing_pipeline():
    """
    Ejecuta el pipeline completo de preprocesamiento.
    Retorna los datos listos para entrenamiento.
    """
    set_seed()

    # 1. Cargar datos
    df = load_data()

    # 2. Limpiar datos
    df = clean_data(df)

    # 3. Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 4. Construir y ajustar preprocesador
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # 5. Guardar datos procesados
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)

    logger.info("=" * 60)
    logger.info("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()