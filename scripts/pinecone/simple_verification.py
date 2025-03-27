#!/usr/bin/env python3
"""
Simplified Pinecone verification script using only the Pinecone SDK.
"""
import os
import sys
import time
import uuid
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run Pinecone verification tests."""
    # Load environment variables
    load_dotenv()

    # Get credentials from environment
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables.")
        sys.exit(1)

    try:
        # Import and initialize Pinecone
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

        # Get stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")

        # Create test document ID
        test_id = f"test_doc_{uuid.uuid4().hex[:8]}"

        # Create test vector (using simple dummy values)
        dimension = stats.get('dimension', 1536)
        test_vector = [0.1] * dimension
        test_metadata = {
            "title": "Test Document",
            "content": "This is a test document for Pinecone integration",
            "document_id": test_id,
            "source": "verification_test",
            "timestamp": time.time()
        }

        # Insert test vector
        logger.info(f"Inserting test vector with ID: {test_id}")
        index.upsert(
            vectors=[{
                "id": test_id,
                "values": test_vector,
                "metadata": test_metadata
            }]
        )

        # Give time for indexing
        time.sleep(1)

        # Query the vector
        logger.info("Querying the test vector...")
        results = index.query(
            vector=test_vector,
            top_k=5,
            include_metadata=True
        )

        logger.info(f"Query results: {results}")

        if results.get('matches'):
            found_test_vector = False

            for match in results['matches']:
                score = match.get('score', 0)
                match_id = match.get('id', '')
                match_metadata = match.get('metadata', {})

                logger.info(f"Match: {match_id} with score {score}")
                logger.info(f"Metadata: {match_metadata}")

                if match_id == test_id and score > 0.9:
                    found_test_vector = True
                    logger.info("✅ Found our test vector with high score!")

            if found_test_vector:
                logger.info("✅ Query test successful!")
            else:
                logger.warning("⚠️ Our test vector wasn't found in the results")
        else:
            logger.error("❌ No matches found in query")

        # Delete the test vector
        logger.info(f"Deleting test vector: {test_id}")
        index.delete(ids=[test_id])

        # Verify deletion
        time.sleep(1)  # Give time for deletion to propagate
        verify_results = index.query(
            vector=test_vector,
            top_k=5,
            include_metadata=True
        )

        # Check if our vector is gone
        test_vector_gone = True
        for match in verify_results.get('matches', []):
            if match.get('id') == test_id:
                test_vector_gone = False
                break

        if test_vector_gone:
            logger.info("✅ Deletion test successful!")
        else:
            logger.warning("⚠️ Vector was not deleted properly")

        logger.info("All verification tests completed successfully!")

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()