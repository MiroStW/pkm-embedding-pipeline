#!/usr/bin/env python3
"""
Direct test for the Pinecone API
"""
import os
import sys
import logging
from dotenv import load_dotenv
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def generate_random_embedding(dimension=1536):
    """Generate a random normalized embedding vector"""
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    # Normalize the vector
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x/magnitude for x in vector]

def main():
    # Get environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key:
        logger.error("Missing Pinecone API key")
        sys.exit(1)

    try:
        from pinecone import Pinecone

        # Connect to Pinecone
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        logger.info("Connected to Pinecone")

        # List indexes
        logger.info("Listing indexes...")
        indexes = pc.list_indexes().names()
        logger.info(f"Available indexes: {indexes}")

        # Connect to index if it exists
        if index_name in indexes:
            logger.info(f"Connecting to index '{index_name}'...")
            index = pc.Index(index_name)

            # Get stats
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")

            # Test vector operations
            test_vector_id = f"test_{random.randint(10000000, 99999999)}"
            test_vector = generate_random_embedding()

            # Upsert a test vector
            logger.info(f"Upserting test vector with ID: {test_vector_id}")
            index.upsert(
                vectors=[{
                    "id": test_vector_id,
                    "values": test_vector,
                    "metadata": {"test": True, "timestamp": str(random.randint(1000000, 9999999))}
                }]
            )

            # Query for the vector
            logger.info("Querying for the test vector...")
            results = index.query(
                vector=test_vector,
                top_k=1,
                include_metadata=True
            )

            if results.get('matches'):
                match = results['matches'][0]
                logger.info(f"Found match: {match.get('id')} with score {match.get('score')}")
                logger.info(f"Metadata: {match.get('metadata')}")
            else:
                logger.warning("No matches found")

            # Delete the test vector
            logger.info(f"Deleting test vector: {test_vector_id}")
            index.delete(ids=[test_vector_id])

            logger.info("Direct Pinecone test completed successfully")
        else:
            logger.error(f"Index '{index_name}' not found. Available indexes: {indexes}")

    except Exception as e:
        logger.error(f"Error during direct Pinecone test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()