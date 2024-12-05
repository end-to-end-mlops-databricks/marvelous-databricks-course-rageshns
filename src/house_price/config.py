import logging
from typing import Any, Dict, List

import yaml

# from loguru import logger
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any] = Field(
        description="Parameters for model training"
    )  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any] = Field(description="Parameters for A/B testing")  # Dictionary to hold A/B test parameters
    pipeline_id: str  # pipeline id for data live tables

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)


# class Config(BaseModel):
#     catalog_name: str
#     schema_name: str
#     pipeline_id: str
#     parameters: Dict[str, Any] = Field(description="Parameters for model training")
#     ab_test: Dict[str, Any] = Field(description="Parameters for A/B testing")
#     num_features: List[NumFeature]
#     target: List[Target]
#     features: Features


# def setup_logging(log_file: Optional[str] = "", log_level: str = "DEBUG") -> None:
#     """
#     Sets up logging configuration with optional file logging.

#     Args:
#         log_file (str, optional): Path to the log file. Defaults to None.
#         log_level (str, optional): Logging level to use. Defaults to "DEBUG".
#     """

#     # Remove the default logger
#     logger.remove()

#     # Add file logger with rotation if log_file is provided
#     if log_file != "":
#         logger.add(log_file, level=log_level, rotation="500 MB")

#     # Add stdout logger
#     logger.add(
#         sys.stdout,
#         level=log_level,
#         colorize=True,
#         format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
#     )


# def load_config(config_path: str) -> ProjectConfig:
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             config_data = yaml.safe_load(f)
#         config = ProjectConfig(**config_data)  # Pydantic validation
#         logger.info(f"Loaded configuration from {config_path}")
#         return config
#     except FileNotFoundError:
#         logger.error(f"Configuration file not found: {config_path}")
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f"Error parsing YAML configuration file: {str(e)}")
#         raise
#     except ValidationError as e:
#         logger.error(f"Validation error in configuration: {e}")
#         raise
