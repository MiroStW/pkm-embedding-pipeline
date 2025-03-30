#!/usr/bin/env python3
import os
import sys
import logging
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get Pinecone credentials from environment
    api_key = os.environ.get('PINECONE_API_KEY')

    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # List all indexes
    index_list = pc.list_indexes()
    indexes = index_list.names() if hasattr(index_list, 'names') else []
    logger.info(f"Found indexes: {indexes}")

    # Identify test indexes (indexes that start with "test-")
    test_indexes = [index for index in indexes if index.startswith("test-")]
    logger.info(f"Found {len(test_indexes)} test indexes: {test_indexes}")

    # Delete test indexes
    for idx_name in test_indexes:
        logger.info(f"Deleting test index: {idx_name}")
        try:
            pc.delete_index(idx_name)
            logger.info(f"Successfully deleted index {idx_name}")
        except Exception as e:
            logger.error(f"Failed to delete index {idx_name}: {e}")

    # Verify the test indexes are gone
    index_list = pc.list_indexes()
    indexes = index_list.names() if hasattr(index_list, 'names') else []
    test_indexes = [index for index in indexes if index.startswith("test-")]
    if test_indexes:
        logger.warning(f"Some test indexes still exist: {test_indexes}")
    else:
        logger.info("All test indexes have been deleted")

    logger.info("Test data cleanup complete")

if __name__ == "__main__":
    main()