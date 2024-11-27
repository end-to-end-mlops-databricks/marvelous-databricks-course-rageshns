import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from house_price.config import ProjectConfig
from databricks.connect import DatabricksSession

# Load configuration
config = ProjectConfig.from_yaml(config_path="project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.profile("mlops_training").getOrCreate()


# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(int(id) for id in combined_set["Id"])


import pandas as pd
from pandas.api.types import CategoricalDtype



# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100):
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        print(column)
        a = df[column].unique()
        print(df[column])
        p=df[column].value_counts(normalize=True)
        print(a)
        print(p)
        if pd.api.types.is_numeric_dtype(df[column]) and column != "Id":
            # if column in ["LotArea"]:
            #     synthetic_data[column] = df.astype(str)
            if column in ["YearBuilt", "YearRemodAdd"]:
                synthetic_data[column] = np.random.randint(
                    df[column].min(), df[column].max() + 1, num_rows
                )  # Years between existing values
            else:
                mean, std = df[column].mean(), df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows
            )

        elif isinstance(df[column].dtype, pd.CategoricalDtype) or isinstance(df[column].dtype, pd.StringDtype):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows
            )
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, dtype=np.int64))
            else:
                synthetic_data[column] = [min_date] * num_rows

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Ensure no negative values for counts and other logical constraints
#     for col in [

# "MSZoning",
# "Street",
# "LotShape",
# "LandContour",
# "Neighborhood",
# "YearBuilt",
# "Condition1",
# "YearBuilt",
# "TotalBsmtSF",
# "HouseStyle",
# "RoofStyle",

#     ]:
#         if col in synthetic_data.columns:
#             synthetic_data[col] = synthetic_data[col].abs()
#             synthetic_data[col] = synthetic_data[col].round().astype(int)

#     # Handle 'avg_price_per_room' to ensure it's positive
    if "LotArea" in synthetic_data.columns:
        synthetic_data["LotArea"] = synthetic_data["LotArea"].astype(str)


    # Generate unique Booking_IDs
    # existing_ids = set(df["Id"])
    # new_ids = []
    # while len(new_ids) < num_rows:
    #     new_id = "Synthetic_" + str(np.random.randint(1e6, 1e7))
    #     if new_id not in existing_ids:
    #         new_ids.append(new_id)
    #         existing_ids.add(new_id)
    # synthetic_data["Id"] = new_ids


    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data["Id"] = new_ids

    return synthetic_data


# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

existing_schema = spark.table(f"{catalog_name}.{schema_name}.train_set").schema
print(existing_schema)
synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.source_data")
