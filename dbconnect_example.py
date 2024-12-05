from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("dev").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
