import logging
import pandas as pd
import yaml
from databricks.connect import DatabricksSession
import mlflow
from mlflow.models import infer_signature
from house_price.data_processor_test import DataProcessor_test
from house_price.price_model_test import HousePriceModelWrapper
from house_price.utils import plot_feature_importance, visualize_results
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from house_price.data_processor import ProjectConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

volume_path = "/Volumes/mlops_students/rageshns/data/"
file_name = "data.csv"
data_path = volume_path + file_name

config = ProjectConfig.from_yaml(config_path="project_config.yml")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# Load configuration
with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)


logger.info("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

spark = DatabricksSession.builder.profile("mlops_training").getOrCreate()
df = spark.read.csv(data_path, header=True, inferSchema=True).toPandas()


run_id = mlflow.search_runs(
    experiment_names=["/Shared/house-prices"],
    filter_string="tags.branch='develop2'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# Initialize DataProcessor
data_processor = DataProcessor_test('data/data.csv', config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data_x_y()
logger.info("Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")


train = pd.concat([X_train, y_train], axis=1)
train.columns = list(X_train.columns) + ["target"]

test = pd.concat([X_test, y_test], axis=1)
test.columns = list(X_test.columns) + ["target"]

data_processor.pandas_df_to_delta(df=train, name="train", spark=spark)

data_processor.pandas_df_to_delta(df=test, name="test", spark=spark)

logger.info("Train and test sets saved as delta tables.")


train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()


wrapped_model = HousePriceModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)
mlflow.set_experiment(experiment_name="/Shared/house-prices-pyfunc")
git_sha = "ffa63b430205ff7"
with mlflow.start_run(tags={"branch": "develop2", "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": example_prediction})
    dataset = mlflow.data.from_spark(train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "code/mlops_with_databricks-0.0.1-py3-none-any.whl",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-house-price-model",
        code_paths=["./dist/mlops_with_databricks-0.0.1-py3-none-any.whl"],
        signature=signature,
    )
