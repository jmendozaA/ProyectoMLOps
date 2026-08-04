import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Ver todas las versiones del modelo
versions = client.search_model_versions("name='student-performance-model'")

for v in versions:
    print(f"Versión: {v.version}")
    print(f"Stage: {v.current_stage}")
    print(f"Aliases: {v.aliases}")
    print(f"Run ID: {v.run_id}")
    print("-" * 50)