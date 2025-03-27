"""
Configuration loading and management for the embedding pipeline.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager for the embedding pipeline.
    Handles loading configuration from YAML files and environment variables.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution.

        Returns:
            Dict containing configuration values
        """
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)

            # Process environment variable substitutions
            self._process_env_vars(self.config)

            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _process_env_vars(self, config_dict: Dict[str, Any]) -> None:
        """
        Process environment variable substitutions in configuration.
        Replaces ${VAR_NAME} with the value of the environment variable VAR_NAME.

        Args:
            config_dict: Configuration dictionary to process
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._process_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    config_dict[key] = env_value
                else:
                    logger.warning(f"Environment variable {env_var} not found")

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding-specific configuration.

        Returns:
            Dict containing embedding configuration
        """
        embedding_config = self.config.get("embedding", {})

        # Convert to the format expected by EmbeddingModelFactory
        factory_config = {
            "model_type": embedding_config.get("model_type", "e5"),
            "e5_model": embedding_config.get("primary_model"),
            "distiluse_model": embedding_config.get("fallback_model"),
            "device": embedding_config.get("device"),
            "title_weight": embedding_config.get("title_weight", 0.3),
            "enable_title_enhanced": embedding_config.get("enable_title_enhanced", True)
        }

        return factory_config

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database-specific configuration.

        Returns:
            Dict containing database configuration
        """
        return self.config.get("database", {})

    def get_processing_config(self) -> Dict[str, Any]:
        """
        Get processing-specific configuration.

        Returns:
            Dict containing processing configuration
        """
        return self.config.get("processing", {})

    def get_chunking_config(self) -> Dict[str, Any]:
        """
        Get chunking-specific configuration.

        Returns:
            Dict containing chunking configuration
        """
        return self.config.get("chunking", {})

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get pipeline-specific configuration.

        Returns:
            Dict containing pipeline configuration
        """
        return self.config.get("pipeline", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging-specific configuration.

        Returns:
            Dict containing logging configuration
        """
        return self.config.get("logging", {})