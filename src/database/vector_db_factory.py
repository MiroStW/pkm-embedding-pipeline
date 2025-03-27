"""
Factory for creating vector database uploader instances.
"""
import logging
from typing import Dict, Any, Optional

from src.database.vector_db import VectorDatabaseUploader
from src.database.mock_vector_db import MockVectorDatabaseUploader

logger = logging.getLogger(__name__)

def create_vector_db_uploader(config: Dict[str, Any]) -> Optional[Any]:
    """
    Create a vector database uploader based on configuration.

    Args:
        config: Database configuration dictionary

    Returns:
        Vector database uploader instance or None if configuration is invalid
    """
    if not config or "vector_db" not in config:
        logger.error("Missing vector_db configuration")
        return None

    vector_config = config.get("vector_db", {})
    provider = vector_config.get("provider", "").lower()

    if provider == "pinecone":
        # Extract Pinecone configuration
        api_key = vector_config.get("api_key")
        environment = vector_config.get("environment")
        index_name = vector_config.get("index_name")
        dimension = vector_config.get("dimension", 1024)

        if not all([api_key, environment, index_name]):
            logger.error("Missing required Pinecone configuration (api_key, environment, index_name)")
            return None

        try:
            return VectorDatabaseUploader(
                api_key=api_key,
                environment=environment,
                index_name=index_name,
                dimension=dimension,
                max_retries=vector_config.get("max_retries", 3),
                retry_delay=vector_config.get("retry_delay", 2.0),
                batch_size=vector_config.get("batch_size", 100)
            )
        except Exception as e:
            logger.error(f"Failed to create Pinecone uploader: {str(e)}")
            return None

    elif provider == "mock":
        # Create mock uploader for testing/benchmarking
        logger.info("Creating mock vector database uploader")
        return MockVectorDatabaseUploader(**vector_config)

    else:
        logger.error(f"Unsupported vector database provider: {provider}")
        return None