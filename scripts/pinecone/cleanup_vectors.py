#!/usr/bin/env python3
"""
Script to clean up test vectors from Pinecone index.
"""
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
    """Clean up test vectors from Pinecone index."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables.")
        sys.exit(1)

    try:
        from pinecone import Pinecone

        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        logger.info("Successfully connected to Pinecone")

        # Check if index exists
        indexes = pc.list_indexes().names()
        if index_name not in indexes:
            logger.error(f"Index '{index_name}' does not exist")
            sys.exit(1)

        logger.info(f"Connecting to index '{index_name}'...")
        index = pc.Index(index_name)
        logger.info(f"Connected to index '{index_name}'")

        # Get stats before cleanup
        stats = index.describe_index_stats()
        logger.info(f"Index stats before cleanup: {stats}")

        # Query for test vectors
        # Since free tier doesn't support filter-based deletion, we need to get all vectors
        # and filter manually
        dimension = stats.get('dimension', 1536)
        test_vector = [0.1] * dimension

        # Create a list to store test vector IDs
        test_vector_ids = []

        # Query for vectors
        results = index.query(
            vector=test_vector,
            top_k=100,  # Increase if you have more test vectors
            include_metadata=True
        )

        # Filter for test vectors by ID pattern
        for match in results.get('matches', []):
            vector_id = match.get('id', '')
            if vector_id.startswith('test_') or vector_id.startswith('test-'):
                test_vector_ids.append(vector_id)
                logger.info(f"Found test vector: {vector_id}")

        if test_vector_ids:
            # Delete test vectors
            logger.info(f"Deleting {len(test_vector_ids)} test vectors: {test_vector_ids}")
            index.delete(ids=test_vector_ids)

            # Verify deletion
            stats_after = index.describe_index_stats()
            logger.info(f"Index stats after cleanup: {stats_after}")

            logger.info("Cleanup completed successfully!")
        else:
            logger.info("No test vectors found to clean up.")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()