#!/usr/bin/env python3
"""
Debug script to verify that vector_db_factory properly creates a PineconeClient instance.
"""
import os
import sys
import logging
import asyncio
from dotenv import load_dotenv

# Add the project root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_test():
    """Run the debug test asynchronously."""
    # Load environment variables
    load_dotenv()

    # Check environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    logger.info(f"PINECONE_API_KEY: {'Set' if api_key else 'Not set'}")
    logger.info(f"PINECONE_ENVIRONMENT: {'Set' if environment else 'Not set'}")
    logger.info(f"PINECONE_INDEX_NAME: {'Set' if index_name else 'Not set'}")

    # Create config dictionary
    config = {
        'vector_db': {
            'provider': 'pinecone',
            'api_key': api_key,
            'environment': environment,
            'index_name': index_name,
            'dimension': 1024,
            'serverless': False,
            'max_retries': 3,
            'retry_delay': 2.0,
            'batch_size': 100
        }
    }

    try:
        # Try to import the factory
        from src.database.vector_db_factory import create_vector_db_uploader

        # Create uploader
        logger.info("Creating vector database uploader...")
        uploader = await create_vector_db_uploader(config)

        # Check what type of uploader was created
        if uploader is None:
            logger.error("Factory returned None")
        else:
            logger.info(f"Created uploader of type: {type(uploader).__name__}")

            # Check if it's a mock
            from src.database.mock_vector_db import MockVectorDatabaseUploader
            if isinstance(uploader, MockVectorDatabaseUploader):
                logger.error("Factory created a MOCK uploader instead of a real one!")
            else:
                logger.info("SUCCESS: Factory created a real uploader!")

                # Check connection
                logger.info("Checking connection...")
                if hasattr(uploader, 'check_connection'):
                    connection_result = uploader.check_connection()
                    logger.info(f"Connection result: {connection_result}")
                else:
                    logger.warning("Uploader doesn't have check_connection method")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)

def main():
    """Main entry point."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_test())

if __name__ == "__main__":
    main()