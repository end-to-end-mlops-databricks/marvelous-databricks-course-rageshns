# import datetime
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price.config import ProjectConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""

        # Checking for missing values
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:\n", missing_values)

        # Remove rows with missing values in the target column
        target = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        num_features = self.config["num_features"]
        cat_features = self.config["cat_features"]

        logger.info(f"Numeric features: {num_features}")
        logger.info(f"cat_features: {cat_features}")
        logger.info(f"target: {target}")

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

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def split_data_x_y(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
