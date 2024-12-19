# Databricks notebook source
# MAGIC %md
# MAGIC # pre-process data to model

# COMMAND ----------

from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, DateType

# COMMAND ----------

# Nombre de la tabla
delta_table = "dbdemos.`ml-project-cav`.raw"

df = spark.read.format("delta").table(delta_table)

# Ajustar el esquema de las columnas manualmente
df = df.withColumn("user_id", df["user_id"].cast("int")) \
       .withColumn("monthly_income", df["monthly_income"].cast("double")) \
       .withColumn("age", df["age"].cast("int")) \
       .withColumn("employment_years", df["employment_years"].cast("int")) \
       .withColumn("loan_amount", df["loan_amount"].cast("double")) \
       .withColumn("credit_score", df["credit_score"].cast("int")) \
       .withColumn("payment_on_time", df["payment_on_time"].cast("int")) \
       .withColumn("fecha_corte", df["fecha_corte"].cast("date")) \
       .withColumn("fecha_pago", df["fecha_pago"].cast("date"))



# COMMAND ----------

df.display()

# COMMAND ----------

output_table_name = "dbdemos.`ml-project-cav`.refined"  
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table_name)

# COMMAND ----------

df.groupBy("payment_on_time").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # split data

# COMMAND ----------

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
print(f"Número de filas en el conjunto de entrenamiento: {train_df.count()}")
print(f"Número de filas en el conjunto de prueba: {test_df.count()}")

# COMMAND ----------

output_table_name = "dbdemos.`ml-project-cav`.splited_data"  
train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{output_table_name}_train_df")
test_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{output_table_name}_test_df")
