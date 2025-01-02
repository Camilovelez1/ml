from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


table_name = "brz_dev.dbdemos.splited_data"  
train_df = spark.read.format("delta").table(f"{table_name}_train_df")
test_df = spark.read.format("delta").table(f"{table_name}_test_df")

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

# 4. Definimos el modelo Random Forest
#rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", numTrees=300)

# 5. Entrenamos el modelo
model = rf.fit(train_df)

# 6. Predicciones en el conjunto de prueba
predictions = model.transform(test_df)

# 7. Evaluación del modelo
# Evaluador de clasificación binaria: AUC (Área bajo la curva ROC)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Área bajo la curva ROC (AUC): {roc_auc}")

# 8. Evaluación adicional con precisión y recall
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Precisión: {accuracy}")
