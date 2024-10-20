from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("mlops_training").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
