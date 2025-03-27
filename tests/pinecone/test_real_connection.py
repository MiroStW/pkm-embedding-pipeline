#!/usr/bin/env python3
"""
Test for the Pinecone client with real API key.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Test the Pinecone client with a real API key.
    """
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key:
        logger.error("Pinecone API key not found. Please set PINECONE_API_KEY in .env file.")
        sys.exit(1)

    if not index_name:
        logger.error("Pinecone index name not found. Please set PINECONE_INDEX_NAME in .env file.")
        sys.exit(1)

    try:
        # Import modules
        from src.database.pinecone_client import PineconeClient

        # Initialize client
        logger.info("Initializing PineconeClient...")
        client = PineconeClient(
            api_key=api_key,
            environment=environment or "gcp-starter",
            index_name=index_name,
            dimension=1536
        )

        # Test connection
        logger.info("Testing connection...")
        connected = client.check_connection()
        logger.info(f"Connection successful: {connected}")

        if not connected:
            logger.error("Failed to connect to Pinecone. Check your API key and index name.")
            sys.exit(1)

        # Get stats
        logger.info("Getting index statistics...")
        stats = client.get_stats()
        logger.info(f"Index stats: {stats}")

        logger.info("Test completed successfully!")

    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Make sure you have all required packages installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()