from venv import logger
import pandas as pd
from pandas import read_csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor_test:
    def __init__(self, file_path: str, config):
        self.df = self.load_data(file_path)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, path):
        return read_csv(path)

    def preprocess_data(self):
        # Checking for missing values
        # missing_values = self.df.isnull().sum()
        # print("Missing values in each column:\n", missing_values)

        # Remove rows with missing values in the target column
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        num_features = self.config["num_features"]
        cat_features = self.config["cat_features"]

        logger.info(f"Numeric features: {num_features}")
        logger.info(f"cat_features: {cat_features}")
        logger.info(f"target: {target}")
        #  to update
        # self.y = self.df[self.config["target"]].astype(int)
        # self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        # self.X[self.config["cat_features"]] = self.X[self.config["cat_features"]].astype(str)
        # self.X[self.config["num_features"]] = self.X[self.config["num_features"]].astype(int)

        # Separate features and target
        self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        self.y = self.df[target]

        # Separate features and target variable based on configuration
        # self.X = self.df[self.config.num_features + self.config.cat_features]
        # self.y = self.df[target]

        # Remove outliers in numerical features
        # lower_bound = self.config.parameters["lower_bound"]
        # upper_bound = self.config.parameters["upper_bound"]
        # self.df = remove_outliers(self.df, num_features, lower_bound, upper_bound)

        # Impute missing values for numerical features with the median
        for col in num_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        # Convert numerical features to numeric type
        num_features = self.config["num_features"]
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical features to the appropriate type
        cat_features = self.config["cat_features"]
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Create preprocessing steps for numeric data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        # Create preprocessing steps for categorical data
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine numeric and categorical preprocessing steps into a single transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("cat", categorical_transformer, self.config["cat_features"]),
            ]
        )

        # Remove duplicate entries
        # self.df.drop_duplicates(inplace=True)

        # Imputation
        # for col in self.df.columns:
        #     if self.df[col].isnull().sum() > 0:
        #         if self.df[col].dtype == 'object':
        #             # Mode imputation for categorical features
        #             self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        #         else:
        #             # Mean imputation for numerical features
        #             self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Impute missing values for categorical features with the most frequent value
        # for col in cat_features:
        #     if self.df[col].isnull().any():
        #         self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Handle missing values and convert data types as needed
        # self.df["LotFrontage"] = pd.to_numeric(self.df["LotFrontage"], errors="coerce")

        # self.df["GarageYrBlt"] = pd.to_numeric(self.df["GarageYrBlt"], errors="coerce")
        # median_year = self.df["GarageYrBlt"].median()

        # self.df["GarageYrBlt"].fillna(median_year, inplace=True)
        # current_year = datetime.now().year

        # self.df["GarageAge"] = current_year - self.df["GarageYrBlt"]
        # self.df.drop(columns=["GarageYrBlt"], inplace=True)

        # Fill missing values with mean or default values
        # self.df.fillna(0, inplace=True)

        # self.df.dropna(inplace=True)

        # Fill missing values with mean or default values
        # self.df.fillna(
        #     {
        #         # "LotFrontage": self.df["LotFrontage"].mean(),
        #         # "MasVnrType": "None",
        #         "MasVnrArea": 0
        #     },
        #     inplace=True,
        # )

        # Handle missing values in the target column by replacing them with a specific value (e.g., 220000)
        # target = self.config.target
        # new_value = 22000  # New value to replace NaNs in the target column
        # self.df[target].fillna(new_value, inplace=True)

        return self.df


    def split_data(self, test_size=None, random_state=None):
        if test_size is None:
            test_size = self.config["test_size"]

        if random_state is None:
            random_state = self.config["seed"]

        return train_test_split(self.X_df_transformed, self.y, test_size=test_size, random_state=random_state)

    def pandas_df_to_delta(self, df, name, spark):
        self._pandas_to_spark_to_delta_cdf(df, name, spark)

    def save_raw_data_to_catalog(self, spark: SparkSession):
        self._pandas_to_spark_to_delta_cdf(self.df, "raw_data", spark)

    def _pandas_to_spark_to_delta_cdf(self, pandas_df: pd.DataFrame, tbl_name: str, spark: SparkSession):
        spark_df = spark.createDataFrame(pandas_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        delta_table_path = f"{self.config['catalog_name']}.{self.config['schema_name']}.{tbl_name}"

        spark_df.write.mode("overwrite").saveAsTable(delta_table_path)

        spark.sql(f"ALTER TABLE {delta_table_path} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

    def split_data_x_y(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            