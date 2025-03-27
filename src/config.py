"""
Configuration utilities for loading and managing application configuration.
"""
import os
import re
import yaml
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Regex for finding environment variable references in the format ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r'\${([^}]+)}')

def substitute_env_vars(value: str) -> str:
    """
    Replace environment variable references in the string with their values.

    Args:
        value: String that may contain environment variable references.

    Returns:
        String with environment variables replaced with their values.
    """
    def replace_env_var(match):
        env_var_name = match.group(1)
        env_var_value = os.environ.get(env_var_name)

        if env_var_value is None:
            logger.warning(f"Environment variable '{env_var_name}' not found")
            return match.group(0)  # Return the original placeholder if variable not found

        logger.debug(f"Substituted environment variable: {env_var_name}")
        return env_var_value

    # Replace all environment variable references
    if isinstance(value, str):
        return ENV_VAR_PATTERN.sub(replace_env_var, value)
    return value

def process_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process configuration dictionary, substituting environment variables.

    Args:
        config: Configuration dictionary.

    Returns:
        Processed configuration dictionary.
    """
    result = {}

    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = process_config_dict(value)
        elif isinstance(value, list):
            # Process lists
            result[key] = [
                process_config_dict(item) if isinstance(item, dict)
                else substitute_env_vars(item) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            # Substitute environment variables in strings
            result[key] = substitute_env_vars(value)
        else:
            # Keep other types as is
            result[key] = value

    return result

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Configuration dictionary.
    """
    if not config_path:
        # Use default path relative to the project root
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Process configuration
        processed_config = process_config_dict(config)

        # Debug info
        logger.debug("Configuration loaded successfully")
        return processed_config

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

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
                    logger.debug(f"Applied environment variable {env_var} to config key {key}")
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