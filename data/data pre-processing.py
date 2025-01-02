# Databricks notebook source
# MAGIC %md
# MAGIC # pre-process data to model

# COMMAND ----------

from pyspark.sql.functions import input_file_name
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, DateType

# COMMAND ----------

# Nombre de la tabla
delta_table = "brz_dev.dbdemos.raw"

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

output_table_name = "brz_dev.dbdemos.refined"  
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table_name)

# COMMAND ----------

df.groupBy("payment_on_time").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # split data
# MAGIC
# MAGIC balanceo de clases

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth

# COMMAND ----------

df = df.withColumn("fecha_corte_year", year("fecha_corte"))
df = df.withColumn("fecha_corte_month", month("fecha_corte"))
df = df.withColumn("fecha_corte_day", dayofmonth("fecha_corte"))

df = df.withColumn("fecha_pago_year", year("fecha_pago"))
df = df.withColumn("fecha_pago_month", month("fecha_pago"))
df = df.withColumn("fecha_pago_day", dayofmonth("fecha_pago"))


# COMMAND ----------

# Filtramos los datos por clase
class_1_df = df.filter(df["payment_on_time"] == 1)
class_0_df = df.filter(df["payment_on_time"] == 0)

# Contamos cuántos registros tiene la clase mayoritaria (payment_on_time = 1)
class_1_count = class_1_df.count()
class_0_count = class_0_df.count()

# Calcular cuántas veces debemos duplicar la clase minoritaria
duplicates_needed = int(class_1_count / class_0_count)  # Proporción de duplicados para la clase 0

# Duplicamos los registros de la clase minoritaria (payment_on_time = 0)
duplicated_class_0_df = class_0_df
for _ in range(duplicates_needed - 1):  # No duplicamos el primero, ya está incluido
    duplicated_class_0_df = duplicated_class_0_df.union(class_0_df)

# Combinamos las clases balanceadas
balanced_df = class_1_df.union(duplicated_class_0_df)

balanced_df = balanced_df.drop(*["fecha_corte", "fecha_pago"])


# COMMAND ----------

balanced_df.display()

# COMMAND ----------

train_df, test_df = balanced_df.randomSplit([0.7, 0.3], seed=42)
print(f"Número de filas en el conjunto de entrenamiento: {train_df.count()}")
print(f"Número de filas en el conjunto de prueba: {test_df.count()}")

# Verificamos el balance después del split
train_class_counts = train_df.groupBy("payment_on_time").count().collect()
test_class_counts = test_df.groupBy("payment_on_time").count().collect()

print("Conjunto de entrenamiento:", train_class_counts)
print("Conjunto de prueba:", test_class_counts)

# COMMAND ----------

output_table_name = "brz_dev.dbdemos.splited_data"  
train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{output_table_name}_train_df")
test_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{output_table_name}_test_df")
