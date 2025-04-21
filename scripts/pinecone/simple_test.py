#!/usr/bin/env python3
"""
Simple test script to directly upload vectors to Pinecone.
"""
import os
import sys
import time
import random
import uuid
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_embedding(dimension=1024):
    """Generate a random normalized embedding vector"""
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x/magnitude for x in vector]

def main():
    """Main function to test Pinecone."""
    # Load environment variables
    load_dotenv()

    # Get environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables")
        return 1

    logger.info(f"Testing Pinecone with index: {index_name}")

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        logger.info("Connected to Pinecone API")

        # List indexes
        indexes = pc.list_indexes().names()
        logger.info(f"Available indexes: {indexes}")

        if index_name not in indexes:
            logger.error(f"Index '{index_name}' not found")
            return 1

        # Connect to index
        index = pc.Index(index_name)

        # Get initial stats
        initial_stats = index.describe_index_stats()
        logger.info(f"Initial stats: {initial_stats}")

        # Generate a unique test ID
        test_id = f"test_{uuid.uuid4().hex[:8]}"
        logger.info(f"Using test ID: {test_id}")

        # Create test vector
        test_vector = generate_random_embedding(1024)

        # Create metadata
        metadata = {
            "document_id": test_id,
            "source": "test_script",
            "timestamp": time.time()
        }

        # Upload vector
        logger.info("Uploading test vector...")
        index.upsert(
            vectors=[{
                "id": test_id,
                "values": test_vector,
                "metadata": metadata
            }]
        )

        # Wait for indexing
        time.sleep(2)

        # Get updated stats
        updated_stats = index.describe_index_stats()
        logger.info(f"Updated stats: {updated_stats}")

        # Query for the vector
        logger.info("Querying for test vector...")
        results = index.query(
            vector=test_vector,
            top_k=1,
            include_metadata=True
        )

        # Check if we got a result
        if results.get('matches') and len(results['matches']) > 0:
            match = results['matches'][0]
            logger.info(f"Found match with ID: {match['id']} and score: {match['score']}")
            logger.info(f"Match metadata: {match.get('metadata')}")

            if match['id'] == test_id:
                logger.info("✅ Test vector retrieved successfully!")
            else:
                logger.warning("⚠️ Retrieved a different vector than expected!")
        else:
            logger.error("❌ No matches found for test vector")

        # Clean up
        logger.info("Cleaning up test vector...")
        index.delete(ids=[test_id])

        # Final check
        final_stats = index.describe_index_stats()
        logger.info(f"Final stats: {final_stats}")

        logger.info("Test completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())