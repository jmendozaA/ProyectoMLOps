# Student Performance Prediction - MLOps Project

Sistema de MLOps para predecir el rendimiento académico de estudiantes a partir de 19 variables, con entrenamiento reproducible, despliegue en FastAPI, contenedorización con Docker, orquestación en Kubernetes y monitoreo de drift.

## Tabla de contenidos
- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [API](#api)
- [Monitoreo](#monitoreo)
- [Tecnologías](#tecnologías)
- [Licencia](#licencia)

## Descripción

Este proyecto implementa un pipeline completo de MLOps para predecir `Exam_Score` usando un modelo de regresión basado en XGBoost. El flujo cubre preparación de datos, entrenamiento con MLflow, servicio de inferencia, despliegue en Kubernetes y monitoreo de data drift y concept drift.

### Objetivos
- Entrenamiento reproducible con tracking de experimentos en MLflow.
- Servicio de inferencia con FastAPI y validación con Pydantic.
- Despliegue escalable con Docker y Kubernetes.
- Monitoreo continuo de drift con alertas automáticas.

## Arquitectura

1. **Entrenamiento**: preprocesamiento, entrenamiento y registro del modelo.
2. **Inferencia**: API FastAPI para predicciones individuales y por lotes.
3. **Despliegue**: Docker y Kubernetes con réplicas y balanceo de carga.
4. **Observabilidad**: métricas del servicio, detección de drift y alertas.

## Características

- Modelo: XGBoost Regressor.
- Métricas: RMSE, MAE y R².
- Variables: 19 features numéricas y categóricas.
- API: `/predict`, `/predict/batch`, `/health`, `/metrics`, `/model/info`, `/model/reload`.
- Monitoreo: KS, Chi-cuadrado, PSI y detección de degradación de desempeño.

## Requisitos

### Software
- Python 3.10 o superior.
- Docker Desktop con Kubernetes habilitado.
- Kind o Minikube.
- Git.

### Hardware recomendado
- CPU de 4 núcleos o más.
- 8 GB de RAM o más.
- 20 GB de espacio libre o más.

## Instalación

```bash
git clone https://github.com/tu-usuario/student-performance-mlops.git
cd student-performance-mlops
```

```bash
python -m venv env
```

Activación del entorno virtual:

- Windows:

```bash
env\Scriptsctivate
```

- Linux/Mac:

```bash
source env/bin/activate
```

```bash
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz del proyecto:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=student-performance-model
DRIFT_KS_THRESHOLD=0.05
DRIFT_PSI_THRESHOLD=0.2
DRIFT_CONCEPT_THRESHOLD=0.15
```

## Uso

### 1. Entrenamiento del modelo

Preprocesa los datos:

```bash
python src/training/preprocess.py
```

Inicia MLflow:

```bash
mlflow ui --port 5000
```

Entrena el modelo:

```bash
python src/training/train.py
```

Registra el mejor modelo:

```bash
python src/training/register_model.py
```

### 2. Contenerización con Docker

Guarda el modelo localmente:

```bash
python scripts/savemodellocal.py
```

Construye la imagen:

```bash
docker build -t student-performance-api:v1 -f docker/Dockerfile .
```

Ejecuta el contenedor:

```bash
docker run -d --name test-api -p 8000:8000 student-performance-api:v1
```

Verifica el estado:

```bash
curl http://localhost:8000/health
```

Ejemplo de predicción:

```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{
    "Hours_Studied": 20,
    "Attendance": 85,
    "Parental_Involvement": "Medium",
    "AccesstoResources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Previous_Scores": 75,
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 2,
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Physical_Activity": 3,
    "Learning_Disabilities": "No",
    "ParentalEducationLevel": "College",
    "DistancefromHome": "Near",
    "Gender": "Male"
  }'
```

### 3. Despliegue en Kubernetes

Cargar imagen en Kind:

```bash
kind load docker-image student-performance-api:v1 --name my-cluster
```

Aplicar manifiestos:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Verificar pods:

```bash
kubectl get pods -n mlops
```

Exponer el servicio localmente:

```bash
kubectl port-forward svc/student-performance-service 30080:8000 -n mlops
```

Accede a la documentación en:

```text
http://localhost:30080/docs
```

### 4. Monitoreo de drift

Ejecuta la demo completa:

```bash
python src/monitoring/demodriftcompleto.py
```

Ejemplo manual de data drift:

```python
from src.monitoring.data_drift import DataDriftDetector
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
current_data = pd.read_csv("data/new_batch.csv")

detector = DataDriftDetector(reference_data=X_train)
results = detector.run_full_detection(current_data)
print(results["summary"])
```

Ejemplo manual de concept drift:

```python
from src.monitoring.concept_drift import ConceptDriftDetector
import joblib

model = joblib.load("model/model.joblib")

detector = ConceptDriftDetector(
    baseline_rmse=8.5,
    baseline_mae=6.0,
    baseline_r2=0.75,
    threshold=0.15,
)

result = detector.evaluate_batch(
    model=model,
    X_batch=X_batch,
    y_batch=y_batch,
    batch_name="Lote1",
)
print(result)
```

## Estructura del proyecto

```text
ProyectoFinal/
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── src/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── docker/
├── k8s/
├── scripts/
├── docs/
├── notebooks/
├── requirements.txt
└── README.md
```

## API

### GET /health
Devuelve el estado del servicio y del modelo.

### POST /predict
Recibe un registro y devuelve una predicción de `Exam_Score`.

### POST /predict/batch
Recibe una lista de registros y devuelve predicciones en lote.

### GET /metrics
Expone métricas básicas del servicio.

### GET /model/info
Devuelve información del modelo cargado.

### POST /model/reload?alias=champion
Recarga el modelo desde el registry.

## Monitoreo

### Data drift
- Kolmogorov-Smirnov para variables numéricas.
- Chi-cuadrado para variables categóricas.
- Population Stability Index para cambios de distribución.

### Concept drift
- Seguimiento de RMSE, MAE y R² por lotes.
- Alerta si la degradación supera el 15% respecto al baseline durante 3 lotes consecutivos.

### Alertas
- LOW: 1-15% de variables afectadas.
- MEDIUM: 15-30%.
- HIGH: 30-50%.
- CRITICAL: más del 50%.

## Tecnologías

- **Machine Learning**: scikit-learn, XGBoost, LightGBM, MLflow.
- **API y backend**: FastAPI, Pydantic, Uvicorn.
- **Contenedores**: Docker, Kubernetes, Kind.
- **Monitoreo**: SciPy, NumPy, Matplotlib, Seaborn, Plotly.
- **Desarrollo**: Python 3.10, Jupyter, Git, pytest.
