# Databricks notebook source
# MAGIC %md
# MAGIC # Clasification model to finantial risk

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

table_name = "dbdemos.`ml-project-cav`.splited_data"  
train_df = spark.read.format("delta").table(f"{table_name}_train_df")
test_df = spark.read.format("delta").table(f"{table_name}_test_df")

# COMMAND ----------

# MAGIC %md
# MAGIC # pre processing

# COMMAND ----------

# Seleccionar las columnas de características y la etiqueta
feature_columns = ["monthly_income", "age", "employment_years", "loan_amount", "credit_score"]
label_column = "payment_on_time"

# Crear el vector de características
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_df = assembler.transform(train_df).select("features", label_column)
test_df = assembler.transform(test_df).select("features", label_column)

# COMMAND ----------

# MAGIC %md
# MAGIC # logistic regression

# COMMAND ----------



# Entrenar un modelo de clasificación (Logistic Regression como ejemplo)
lr = LogisticRegression(featuresCol="features", labelCol=label_column)
model = lr.fit(train_df)

# Evaluar el modelo en el conjunto de prueba
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol=label_column)
roc_auc = evaluator.evaluate(predictions)

print(f"ROC AUC en el conjunto de prueba: {roc_auc}")

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression

# Crear un modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="payment_on_time")

# Crear una cuadrícula de hiperparámetros
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Configurar validación cruzada
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(labelCol="payment_on_time"),
                    numFolds=3)

# Entrenar el modelo
cvModel = cv.fit(train_df)

# Evaluar en el conjunto de prueba
predictions = cvModel.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="payment_on_time")
roc_auc = evaluator.evaluate(predictions)

print(f"ROC AUC después de ajuste: {roc_auc}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

# Crear un modelo de Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol=label_column, numTrees=100, maxDepth=10, seed=42)

# Entrenar el modelo
rf_model = rf.fit(train_df)

# Hacer predicciones en el conjunto de prueba
predictions = rf_model.transform(test_df)

# Mostrar las predicciones
predictions.select("features", label_column, "prediction", "probability").show(10, truncate=False)

# Evaluar el modelo con ROC AUC
evaluator = BinaryClassificationEvaluator(labelCol=label_column, metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

print(f"ROC AUC en el conjunto de prueba: {roc_auc}")


# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Crear una cuadrícula de hiperparámetros
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.maxBins, [32, 64]) \
    .build()

# Configurar validación cruzada
cv = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(labelCol=label_column, metricName="areaUnderROC"),
                    numFolds=3)

# Entrenar con validación cruzada
cv_model = cv.fit(train_df)

# Hacer predicciones con el mejor modelo
best_predictions = cv_model.bestModel.transform(test_df)
roc_auc_cv = evaluator.evaluate(best_predictions)

print(f"ROC AUC después del ajuste de hiperparámetros: {roc_auc_cv}")

