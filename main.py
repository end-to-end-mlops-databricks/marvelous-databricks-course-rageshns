import logging

from house_price.config import ProjectConfig
from house_price.data_processor import DataProcessor
from house_price.price_model import PriceModel
from house_price.utils import plot_feature_importance, visualize_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load configuration
config = ProjectConfig.from_yaml("project_config.yml")

# Initialise data processor and preprocess data
data_processor = DataProcessor("data/data.csv", config=ProjectConfig)
data_processor.preprocess()


# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")
logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

logger.info("Training set shape:", X_train.shape)
logger.info("Test set shape:", X_test.shape)

# Initialize and train the model
model = PriceModel(data_processor.preprocessor, config)
model.train(X_train, y_train)
logger.info("Model training completed.")


# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)
logger.info(f"Model evaluation completed: MSE={mse}, R2={r2}")

## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)
logger.info("Results visualization completed.")

## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
logger.info("Feature importance plot generated.")
