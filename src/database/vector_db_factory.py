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

async def create_vector_db_uploader(config: Dict[str, Any]) -> Optional[VectorDatabaseUploader]:
    """
    Create and return a vector database uploader based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        VectorDatabaseUploader instance or None if creation fails
    """
    try:
        # Look for vector_db config in both top-level and nested locations
        vector_db_config = config.get('vector_db', {})
        if not vector_db_config and 'database' in config:
            # Try to get it from database.vector_db (nested location)
            vector_db_config = config.get('database', {}).get('vector_db', {})

        provider = vector_db_config.get('provider', 'pinecone').lower()

        # DEBUG: Test Pinecone connection at this point
        print('DEBUG (vector_db_factory): PINECONE_API_KEY from env:', os.getenv('PINECONE_API_KEY'))
        try:
            from pinecone import Pinecone
            print('DEBUG (vector_db_factory): Testing Pinecone connection')
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            print('DEBUG (vector_db_factory): Indexes:', pc.list_indexes().names())
        except Exception as e:
            print('DEBUG (vector_db_factory): Pinecone connection failed:', e)

        if provider == 'pinecone':
            # Get required configuration
            api_key = vector_db_config.get('api_key')
            # Fallback to env if missing, empty, or a placeholder
            if not api_key or api_key.strip() == '' or api_key.strip().startswith('${'):
                api_key = os.getenv('PINECONE_API_KEY')
            print('DEBUG: api_key passed to PineconeClient:', api_key)
            environment = vector_db_config.get('environment')
            index_name = vector_db_config.get('index_name')
            dimension = vector_db_config.get('dimension', 1024)
            serverless = vector_db_config.get('serverless', True)

            if not all([api_key, environment, index_name]):
                logger.error("Missing required Pinecone configuration")
                raise ValueError("Missing required Pinecone configuration: 'api_key', 'environment', and 'index_name' must be provided")

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
                    region=vector_db_config.get('region', 'us-east-1')
                )

                logger.info("PineconeClient created successfully")
                return client

            except Exception as e:
                logger.error(f"Failed to create Pinecone client: {str(e)}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to create Pinecone client: {str(e)}")

        elif provider == 'mock':
            logger.error("MockVectorDatabaseUploader is not allowed; please configure vector_db.provider as 'pinecone'")
            raise ValueError("MockVectorDatabaseUploader is not allowed; configure vector_db.provider as 'pinecone'")

        else:
            logger.error(f"Unknown vector database provider: {provider}; allowed provider is 'pinecone'")
            raise ValueError(f"Unknown vector database provider: {provider}; allowed provider is 'pinecone'")

    except Exception as e:
        logger.error(f"Error creating vector database uploader: {str(e)}")
        logger.error(traceback.format_exc())
        return None