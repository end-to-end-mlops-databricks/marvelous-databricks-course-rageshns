# Databricks notebook source
# MAGIC %pip install mlops_with_databricks-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession

from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
# Load configuration
# with open("project_config.yml", "r") as file:
#     config = yaml.safe_load(file)

# print("Configuration loaded:")
# print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Load the house prices dataset
df = spark.read.csv("/Volumes/mlops_students/rageshns/data/data.csv", header=True, inferSchema=True).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
