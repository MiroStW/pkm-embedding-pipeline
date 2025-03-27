"""
Vector database factory module for creating appropriate instances of VectorDatabaseUploader.
"""
import os
import logging
from typing import Dict, Any, Optional
import traceback

from src.database.vector_db import VectorDatabaseUploader
from src.database.mock_vector_db import MockVectorDatabaseUploader

logger = logging.getLogger(__name__)

def create_vector_db_uploader(config: Dict[str, Any]) -> Optional[VectorDatabaseUploader]:
    """
    Create an instance of VectorDatabaseUploader based on configuration.

    Args:
        config: Dictionary with application configuration.

    Returns:
        Instance of VectorDatabaseUploader or None if configuration is invalid.
    """
    # Super verbose logging to diagnose issues
    logger.debug(f"Config passed to vector_db_factory: {config.keys()}")

    # Get vector database configuration
    vector_db_config = config.get('database', {}).get('vector_db', {})

    # More verbose logging
    logger.info(f"Database config keys: {config.get('database', {}).keys()}")
    logger.info(f"Vector DB config: {vector_db_config}")

    # Get provider
    provider = vector_db_config.get('provider', 'mock')

    # For debugging purposes
    logger.info(f"Creating vector database uploader with provider: {provider}")

    # Create appropriate implementation based on provider
    if provider == 'pinecone':
        # Extract Pinecone configuration
        api_key = vector_db_config.get('api_key')
        environment = vector_db_config.get('environment')
        index_name = vector_db_config.get('index_name')
        dimension = vector_db_config.get('dimension', 1024)
        serverless = vector_db_config.get('serverless', False)

        # Validate required parameters
        if not all([api_key, environment, index_name]):
            missing = []
            if not api_key:
                missing.append("api_key")
                logger.error("Missing Pinecone API key")
            if not environment:
                missing.append("environment")
                logger.error("Missing Pinecone environment")
            if not index_name:
                missing.append("index_name")
                logger.error("Missing Pinecone index name")

            # Debug the actual values (careful with the API key)
            if api_key:
                # Only show first few characters for security
                logger.debug(f"API key (truncated): {api_key[:5]}...")
            else:
                # Debug environment variables
                env_api_key = os.environ.get('PINECONE_API_KEY')
                if env_api_key:
                    logger.debug(f"Environment variable PINECONE_API_KEY exists with value (truncated): {env_api_key[:5]}...")
                else:
                    logger.debug("Environment variable PINECONE_API_KEY is not set")

            logger.debug(f"Environment: {environment}")
            logger.debug(f"Index name: {index_name}")

            logger.error(f"Invalid Pinecone configuration, missing: {', '.join(missing)}")
            return None

        try:
            # Import Pinecone client
            from src.database.pinecone_client import PineconeClient

            logger.info(f"Creating PineconeClient with api_key={api_key[:5]}..., environment={environment}, index_name={index_name}")

            # Create Pinecone client
            client = PineconeClient(
                api_key=api_key,
                environment=environment,
                index_name=index_name,
                dimension=dimension,
                max_retries=vector_db_config.get('max_retries', 3),
                retry_delay=vector_db_config.get('retry_delay', 2.0),
                batch_size=vector_db_config.get('batch_size', 100),
                serverless=serverless,
                cloud_provider=vector_db_config.get('cloud_provider', 'aws'),
                region=vector_db_config.get('region', 'us-west-2')
            )

            logger.info("PineconeClient created successfully")
            return client

        except Exception as e:
            logger.error(f"Failed to create Pinecone client: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("Using mock implementation instead")
            return MockVectorDatabaseUploader()

    elif provider == 'mock':
        # Use mock implementation for testing
        logger.info("Creating MockVectorDatabaseUploader as specified in config")
        return MockVectorDatabaseUploader()

    else:
        logger.error(f"Unknown vector database provider: {provider}")
        logger.info("Using mock implementation as fallback")
        return MockVectorDatabaseUploader()