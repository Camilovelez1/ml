# Databricks notebook source
# MAGIC %md
# MAGIC # Clasification model to finantial risk

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import mlflow

# COMMAND ----------

table_name = "brz_dev.dbdemos.splited_data"  
train_df = spark.read.format("delta").table(f"{table_name}_train_df")
test_df = spark.read.format("delta").table(f"{table_name}_test_df")

# COMMAND ----------

# MAGIC %md
# MAGIC # pre processing

# COMMAND ----------

# 1. Indexamos la columna 'payment_on_time' a 'label'
indexer = StringIndexer(inputCol="payment_on_time", outputCol="label")
train_df = indexer.fit(train_df).transform(train_df)
test_df = indexer.fit(test_df).transform(test_df)

# 2. Seleccionar las columnas de características (sin incluir 'payment_on_time' y 'label')
feature_columns = ["monthly_income", "age", "employment_years", "loan_amount", "credit_score", 
                   "fecha_corte_year", "fecha_corte_month", "fecha_corte_day", 
                   "fecha_pago_year", "fecha_pago_month", "fecha_pago_day"]

# 3. Ensamblaje de características
assembler = VectorAssembler(inputCols=feature_columns, outputCol="unscaled_features")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# 4. Escalado de características utilizando StandardScaler
scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(train_df)
train_df = scaler_model.transform(train_df)
test_df = scaler_model.transform(test_df)

# COMMAND ----------

train_class_counts = train_df.groupBy("payment_on_time").count().collect()
test_class_counts = test_df.groupBy("payment_on_time").count().collect()

print("Conjunto de entrenamiento:", train_class_counts)
print("Conjunto de prueba:", test_class_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC # logistic regression

# COMMAND ----------

# 5. Definimos el modelo de Regresión Logística
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")

# 6. Entrenamos el modelo
model_lr = lr.fit(train_df)

# 7. Predicciones en el conjunto de prueba
predictions = model_lr.transform(test_df)

# 8. Evaluación del modelo
# Evaluador de clasificación binaria: AUC (Área bajo la curva ROC)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"ROC AUC en el conjunto de prueba: {roc_auc}")

# 9. Evaluación adicional con precisión y recall
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Precisión: {accuracy}")

# COMMAND ----------

labels_and_predictions = predictions.select("label", "prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

# 4. Definimos el modelo Random Forest
#rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
#rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", numTrees=300)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaled_features",
    numTrees=300,
    bootstrap=True,
    cacheNodeIds=False,
    checkpointInterval=10,
    featureSubsetStrategy='auto',
    impurity='gini',
    maxBins=32,
    maxDepth=15,
    maxMemoryInMB=256,
    minInfoGain=0.0,
    minInstancesPerNode=1,
    minWeightFractionPerNode=0.0,
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    seed=-7387420455837441889,
    subsamplingRate=1.0
)


# 5. Entrenamos el modelo
model_rf = rf.fit(train_df)

# 6. Predicciones en el conjunto de prueba
predictions = model_rf.transform(test_df)

# 7. Evaluación del modelo
# Evaluador de clasificación binaria: AUC (Área bajo la curva ROC)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Área bajo la curva ROC (AUC): {roc_auc}")

# 8. Evaluación adicional con precisión y recall
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Precisión: {accuracy}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Save the model

# COMMAND ----------

from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Definir las columnas de entrada y salida del modelo
input_schema = Schema([ColSpec("double", col) for col in [
    "monthly_income", "age", "employment_years", "loan_amount", "credit_score", 
    "fecha_corte_year", "fecha_corte_month", "fecha_corte_day", 
    "fecha_pago_year", "fecha_pago_month", "fecha_pago_day"
]])

output_schema = Schema([ColSpec("double", "payment_on_time")])  # 'payment_on_time' es la salida

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Loguear el modelo con los mejores parámetros en MLflow
with mlflow.start_run():
    # Guarda el mejor modelo ajustado en MLflow+
    input_example = test_df.head(1)
    mlflow.spark.log_model(model_rf, "model",signature=signature)

    # Loguea los parámetros del mejor modelo
    best_params = model_rf.extractParamMap()
    for param, value in best_params.items():
        # Asegúrate de que el valor sea un tipo adecuado para loguear
        mlflow.log_param(param.name, value)

    # Asegúrate de calcular las métricas previamente
    mlflow.log_metric("best_accuracy", accuracy)
    mlflow.log_metric("best_auc", roc_auc)

    # Imprime todos los parámetros del mejor modelo
    print("Todos los parámetros del mejor modelo:")
    print(model_rf.explainParams())

    # Registrar el modelo en Unity Catalog después de guardarlo en el almacenamiento de artefactos
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "brz_dev.dbdemos.rf_model")  # Esta línea se queda solo una vez


# COMMAND ----------

# MAGIC %md
# MAGIC # Ajuste de hiperparametros

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Crear la cuadrícula de parámetros para la búsqueda
paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [5, 10, 15])
             .addGrid(rf.numTrees, [200, 300, 400])
             .addGrid(rf.maxBins, [32, 64])
             .build())

# Realizar validación cruzada
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator_accuracy, numFolds=3)

# Ajustar el modelo
cvModel = cv.fit(train_df)

# Mejor modelo después del ajuste
bestModel = cvModel.bestModel
bestPredictions = bestModel.transform(test_df)

# Evaluar el mejor modelo
bestAccuracy = evaluator_accuracy.evaluate(bestPredictions)
bestAUC = evaluator.evaluate(bestPredictions)
print(f"Mejor Precisión: {bestAccuracy}")
print(f"Mejor AUC: {bestAUC}")

print("Todos los parámetros del mejor modelo:")
print(bestModel.explainParams())

# COMMAND ----------

# Imprimir los mejores parámetros
print("Mejores parámetros encontrados:")
print(f"maxDepth: {bestModel.getOrDefault('maxDepth')}")
print(f"numTrees: {bestModel.getOrDefault('numTrees')}")
print(f"maxBins: {bestModel.getOrDefault('maxBins')}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save model

# COMMAND ----------

# Establece la URI del registro de modelos en Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Loguear el modelo con los mejores parámetros en MLflow
with mlflow.start_run():
    # Guarda el mejor modelo ajustado en MLflow
    mlflow.spark.log_model(bestModel, "model")

    # Loguea los parámetros del mejor modelo
    best_params = bestModel.extractParamMap()
    for param, value in best_params.items():
        mlflow.log_param(param.name, value)
    
    # Loguea la métrica de evaluación (como RMSE o AUC)
    mlflow.log_metric("best_rmse", bestAccuracy)
    mlflow.log_metric("best_auc", bestAUC)
    
    # Toma una fila de ejemplo para el modelo
    input_example = test_df.head(1)
    mlflow.spark.log_model(bestModel, "model", input_example=input_example)
    
    # Imprime todos los parámetros del mejor modelo
    print("Todos los parámetros del mejor modelo:")
    print(bestModel.explainParams())

    # Registrar el modelo en Unity Catalog después de guardarlo en el almacenamiento de artefactos
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "brz_dev.dbdemos.rf_model")
