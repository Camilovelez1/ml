# Databricks notebook source
# MAGIC %md
# MAGIC # This notebook will unify 47 tables of data

# COMMAND ----------

input_path = '/Volumes/dbdemos/ml-project-cav/database/*'
df = spark.read.format("csv").option("header", "true").load(input_path)

# COMMAND ----------

df.display()

# COMMAND ----------

output_table_name = "dbdemos.`ml-project-cav`.raw"  
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table_name)
