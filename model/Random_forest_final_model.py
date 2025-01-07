# Databricks notebook source
# MAGIC %md
# MAGIC # Library

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
import pandas as pd

import mlflow
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC # Read data

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
CATALOG_NAME = "brz_dev"
SCHEMA_NAME = "dbdemos"

# COMMAND ----------

table_name = "brz_dev.dbdemos.splited_data"  
train_df = spark.read.format("delta").table(f"{table_name}_train_df")
test_df = spark.read.format("delta").table(f"{table_name}_test_df")


# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-processing data

# COMMAND ----------

# Aquí defines el pipeline de preprocesamiento
indexer = StringIndexer(inputCol="payment_on_time", outputCol="label")

feature_columns = ["monthly_income", "age", "employment_years", "loan_amount", "credit_score", 
                   "fecha_corte_year", "fecha_corte_month", "fecha_corte_day", 
                   "fecha_pago_year", "fecha_pago_month", "fecha_pago_day"]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="unscaled_features")

scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features", withStd=True, withMean=True)

# Modelo de Random Forest
#model = RandomForestClassifier(
#    labelCol="label",
#    featuresCol="scaled_features",
#    numTrees=300,
#    bootstrap=True,
#    cacheNodeIds=False,
#    checkpointInterval=10,
#    featureSubsetStrategy='auto',
#    impurity='gini',
#    maxBins=32,
#    maxDepth=15,
#    maxMemoryInMB=256,
#    minInfoGain=0.0,
#    minInstancesPerNode=1,
#    minWeightFractionPerNode=0.0,
#    predictionCol="prediction",
#    probabilityCol="probability",
#    rawPredictionCol="rawPrediction",
#    seed=-7387420455837441889,
#    subsamplingRate=1.0
#)
model = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaled_features",
    numTrees=50
)

# Pipeline: preprocesamiento + modelo
pipeline = Pipeline(stages=[indexer, assembler, scaler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC # Train the model

# COMMAND ----------

with mlflow.start_run(run_name='tuned_random_forest'):

    pipeline_model = pipeline.fit(train_df)
    train_df_transformed = pipeline_model.transform(train_df)
    #train_df_transformed.display()

    predictions_test = pipeline_model.transform(test_df)
    #predictions_test.display()

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_score = evaluator.evaluate(predictions_test)
    print(f"Área bajo la curva ROC (AUC): {auc_score}")

    params = pipeline_model.stages[-1].extractParamMap()
    for param, value in params.items():
        mlflow.log_param(param.name, value)

    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(predictions_test.select(*feature_columns), predictions_test.select("prediction").toPandas())

    conda_env = _mlflow_conda_env(
        additional_conda_deps=[
            "python=3.12.3",   # Asegúrate de que se instale la versión de Python adecuada
        ],  
        additional_pip_deps=[
            "mlflow==2.15.1",  # Dependencia de MLflow
            "cloudpickle=={}".format(cloudpickle.__version__),  # Dependencia de Cloudpickle
            "pyspark",  # Agregar pyspark aquí
        ],
        additional_conda_channels=["conda-forge"]  # Asegúrate de que se use el canal condaforge
    )
    mlflow.spark.log_model(pipeline_model, "random_forest_model", conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model to Unity Catalog. 

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "tuned_random_forest"').iloc[0].run_id


# COMMAND ----------

model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.rf_model_v2"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Registering the model takes a few seconds, so add a small delay
time.sleep(15)


# COMMAND ----------

from mlflow.tracking import MlflowClient

# Crear una instancia del cliente MLflow
client = MlflowClient()

# Asignar el alias 'Champion' al modelo registrado
client.set_registered_model_alias(model_name, "Champion", model_version.version)


# COMMAND ----------

model = mlflow.spark.load_model(f"models:/{model_name}@Champion")
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

# Imprimir el AUC
print(f'AUC: {auc}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Serve the model

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name="random_forest_model_endpoint_dummy",
    config={
        "served_entities": [
            {
                "name": "random_forest_entity",
                "entity_name": model_name,
                "entity_version": model_version.version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
      }
)

#endpoint = client.create_endpoint(
#    name="random_forest_model_endpoint",
#    config={
#        "served_entities": [
#            {
#                "name": "random_forest_entity",
#                "entity_name": model_name,
#                "entity_version": model_version.version,
#                "workload_size": "Large",  # Usar el tamaño grande
#                "scale_to_zero_enabled": False,  # Asegurarse de que no se escale a cero
#                "autoscaling_enabled": True,  # Habilitar escalado automático
#                "min_instances": 1,  # Número mínimo de instancias
#                "max_instances": 4,  # Número máximo de instancias
#                "gpu": {
#                    "type": "A100",  # Especificar el tipo de GPU A100
#                    "count": 4  # Utilizar 4 GPUs A100
#                },
#                "concurrency": {
#                    "level": "Small",  # Concurrencia pequeña
#                    "min": 0,  # Concurrencia mínima
#                    "max": 4  # Concurrencia máxima (puedes ajustar este rango)
#                }
#            }
#        ],
#    }
#)

# COMMAND ----------

print(f"Endpoint creado: {endpoint}")
