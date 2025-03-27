#!/usr/bin/env python3
"""Simple Pinecone test"""
import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a simple connection test to Pinecone."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    logger.info(f"API Key set: {'Yes' if api_key else 'No'}")

    if not api_key:
        logger.error("Pinecone API key not found in environment variables.")
        return False

    try:
        from pinecone import Pinecone

        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        logger.info("Connected to Pinecone")

        # List indexes
        indexes = pc.list_indexes().names()
        logger.info(f"Available indexes: {indexes}")

        # Get index name from environment
        index_name = os.getenv('PINECONE_INDEX_NAME')
        if index_name:
            if index_name in indexes:
                logger.info(f"✅ Index '{index_name}' found")
            else:
                logger.warning(f"⚠️ Index '{index_name}' not found in your Pinecone account")

        logger.info("Connection test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)