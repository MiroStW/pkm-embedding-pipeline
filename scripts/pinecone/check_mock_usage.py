#!/usr/bin/env python3
"""
Script to check if the system is using the real Pinecone client or falling back to the mock.
This script patches the vector_db_factory to log whenever it creates a client.
"""
import os
import sys
import logging
from dotenv import load_dotenv
import importlib
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monkey_patch_factory():
    """Monkey patch the vector_db_factory to log client creation."""
    try:
        # Add the project root to the path
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.insert(0, root_dir)

        # Import the module
        import src.database.vector_db_factory as factory

        # Get the original function
        original_func = factory.create_vector_db_uploader

        # Define the patched function
        async def patched_create_vector_db_uploader(*args, **kwargs):
            """Patched version that logs the creation."""
            logger.info(f"🔍 create_vector_db_uploader called with:")
            if args:
                logger.info(f"  Args: {args}")
            if kwargs:
                logger.info(f"  Kwargs: {kwargs}")

            # Call the original function
            result = await original_func(*args, **kwargs)

            # Log the result
            logger.info(f"🔍 Factory returned uploader of type: {type(result).__name__}")

            # Check if it's a mock
            from src.database.mock_vector_db import MockVectorDatabaseUploader
            if isinstance(result, MockVectorDatabaseUploader):
                logger.warning("⚠️ USING MOCK VECTOR DATABASE UPLOADER!")
                logger.info("Let's find out why...")

                # Try to trace code execution in the factory
                config = args[0] if args else kwargs.get('config')
                if config:
                    vector_db_config = config.get('vector_db', {})
                    provider = vector_db_config.get('provider', 'pinecone').lower()
                    logger.info(f"Provider configured: {provider}")

                    if provider == 'pinecone':
                        api_key = vector_db_config.get('api_key')
                        environment = vector_db_config.get('environment')
                        index_name = vector_db_config.get('index_name')

                        logger.info(f"Pinecone configuration:")
                        logger.info(f"  API Key set: {'Yes' if api_key else 'No'}")
                        logger.info(f"  Environment set: {'Yes' if environment else 'No'}")
                        logger.info(f"  Index name set: {'Yes' if index_name else 'No'}")

                        if not all([api_key, environment, index_name]):
                            logger.error("Missing required Pinecone configuration!")
                        else:
                            logger.error("All configuration present but still using mock!")
                            logger.info("Check for errors during client creation.")
            else:
                logger.info("✅ USING REAL VECTOR DATABASE UPLOADER")

            return result

        # Apply the patch
        factory.create_vector_db_uploader = patched_create_vector_db_uploader
        logger.info("✅ Successfully patched vector_db_factory")

    except Exception as e:
        logger.error(f"Failed to patch vector_db_factory: {str(e)}")

def main():
    """Main function to test Pinecone configuration."""
    # Load environment variables
    load_dotenv()

    # Patch the factory
    monkey_patch_factory()

    # Print environment variable status
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    logger.info(f"Environment Variables:")
    logger.info(f"  PINECONE_API_KEY set: {'Yes' if api_key else 'No'}")
    logger.info(f"  PINECONE_ENVIRONMENT set: {'Yes' if environment else 'No'}")
    logger.info(f"  PINECONE_INDEX_NAME set: {'Yes' if index_name else 'No'}")

    # Test creating a client using the config
    try:
        logger.info("Testing client creation with direct configuration...")

        # Add the project root to the path if not already added
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        # Import async utilities
        import asyncio

        # Create a test config
        config = {
            'vector_db': {
                'provider': 'pinecone',
                'api_key': api_key,
                'environment': environment,
                'index_name': index_name,
                'dimension': 1024
            }
        }

        # Import the factory again (now patched)
        from src.database.vector_db_factory import create_vector_db_uploader

        # Test creating a client
        async def test_client_creation():
            from src.database.vector_db_factory import create_vector_db_uploader
            uploader = await create_vector_db_uploader(config)
            return uploader

        # Run the async test
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        uploader = asyncio.run(test_client_creation())

        # Try a simple operation
        if hasattr(uploader, 'get_stats'):
            logger.info("Testing get_stats...")
            stats = uploader.get_stats()
            logger.info(f"Stats: {stats}")

    except Exception as e:
        logger.error(f"Error testing client creation: {str(e)}")

    logger.info("Check completed")

if __name__ == "__main__":
    main()