#!/usr/bin/env python3
"""
Final verification script for Pinecone integration.
Runs a sequence of tests to verify the Pinecone integration is working properly.
"""
import os
import sys
import logging
import random
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_embedding(dimension=1536):
    """Generate a random normalized embedding vector"""
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    # Normalize the vector
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x/magnitude for x in vector]

def test_direct_pinecone_connection():
    """Test 1: Direct connection to Pinecone using the SDK"""
    logger.info("=== TEST 1: DIRECT PINECONE CONNECTION ===")

    try:
        from pinecone import Pinecone

        api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME')

        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Connected to Pinecone API")

        # List indexes
        indexes = pc.list_indexes().names()
        logger.info(f"Available indexes: {indexes}")

        if index_name in indexes:
            logger.info(f"✅ Found index '{index_name}'")

            # Connect to index
            index = pc.Index(index_name)

            # Get stats
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")

            logger.info("✅ Direct Pinecone connection test passed")
            return True
        else:
            logger.error(f"❌ Index '{index_name}' not found")
            return False

    except Exception as e:
        logger.error(f"❌ Direct Pinecone connection test failed: {str(e)}")
        return False

def test_pinecone_client():
    """Test 2: PineconeClient integration"""
    logger.info("=== TEST 2: PINECONE CLIENT INTEGRATION ===")

    try:
        from src.database.pinecone_client import PineconeClient

        # Get environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT')
        index_name = os.getenv('PINECONE_INDEX_NAME')

        # Create client
        client = PineconeClient(
            api_key=api_key,
            environment=environment,
            index_name=index_name,
            dimension=1536,
            serverless=True,
            cloud_provider="aws",
            region="us-east-1"
        )

        # Check connection
        if client.check_connection():
            logger.info("✅ PineconeClient connected successfully")
        else:
            logger.error("❌ PineconeClient connection failed")
            return False

        # Get stats before test
        before_stats = client.get_stats()
        logger.info(f"Index stats before test: {before_stats}")

        # Create test document with unique ID
        document_id = f"verify_{int(time.time())}"
        chunks = [
            {
                "content": "This is a test document for verifying the Pinecone integration.",
                "metadata": {
                    "section_title": "Verification Test",
                    "section_level": 1
                }
            }
        ]

        # Generate embeddings
        embeddings = [generate_random_embedding(1536)]

        # Upload document
        logger.info(f"Uploading document with ID: {document_id}")
        success_count, error_count = client.upload_document_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        if success_count > 0 and error_count == 0:
            logger.info("✅ Document uploaded successfully")
        else:
            logger.error(f"❌ Document upload failed: {success_count} success, {error_count} errors")
            return False

        # Query for the document
        results = client.query_vectors(
            query_vector=embeddings[0],
            top_k=1
        )

        if results and len(results) > 0:
            logger.info("✅ Query returned results")

            # Verify result contains our document
            doc_id = results[0].get('metadata', {}).get('document_id')
            if doc_id == document_id:
                logger.info("✅ Query returned the correct document")
            else:
                logger.warning(f"⚠️ Query returned a different document: {doc_id}")
        else:
            logger.error("❌ Query returned no results")
            return False

        # Delete the document
        if client.delete_document(document_id):
            logger.info("✅ Document deleted successfully")
        else:
            logger.error("❌ Document deletion failed")
            return False

        # Verify deletion
        after_stats = client.get_stats()
        logger.info(f"Index stats after test: {after_stats}")

        logger.info("✅ PineconeClient integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ PineconeClient integration test failed: {str(e)}", exc_info=True)
        return False

def main():
    """Run all verification tests"""
    # Load environment variables
    load_dotenv()

    # Check environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables: PINECONE_API_KEY, PINECONE_INDEX_NAME")
        sys.exit(1)

    logger.info("Starting Pinecone integration verification...")

    # Run test 1
    test1_result = test_direct_pinecone_connection()

    # Run test 2
    test2_result = test_pinecone_client()

    # Final summary
    logger.info("=== VERIFICATION SUMMARY ===")
    logger.info(f"Test 1 (Direct Connection): {'✅ PASS' if test1_result else '❌ FAIL'}")
    logger.info(f"Test 2 (PineconeClient): {'✅ PASS' if test2_result else '❌ FAIL'}")

    if test1_result and test2_result:
        logger.info("🎉 All tests passed! Pinecone integration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)