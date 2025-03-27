#!/usr/bin/env python3
"""
Test script for document deletion functionality.
"""
import os
import sys
import time
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
    """Test document deletion functionality."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables.")
        sys.exit(1)

    try:
        from src.database.vector_db import VectorDatabaseUploader

        # Initialize uploader
        logger.info("Initializing VectorDatabaseUploader...")
        uploader = VectorDatabaseUploader(
            api_key=api_key,
            environment="gcp-starter",
            index_name=index_name,
            dimension=1536
        )

        # Generate test document ID
        test_doc_id = f"test_{int(time.time())}"
        logger.info(f"Generated test document ID: {test_doc_id}")

        # Create test vectors
        logger.info("Creating test vectors...")
        vectors = []
        for i in range(3):
            vector_id = f"{test_doc_id}_{i}"
            vector = {
                "id": vector_id,
                "values": [0.1] * 1536,
                "metadata": {
                    "document_id": test_doc_id,
                    "chunk_index": i,
                    "text": f"This is test chunk {i}",
                    "title": "Test Document"
                }
            }
            vectors.append(vector)

        # Upload vectors
        logger.info(f"Uploading {len(vectors)} test vectors...")
        success_count, error_count = uploader.upload_vectors(vectors)
        logger.info(f"Uploaded {success_count} vectors, {error_count} errors")

        if success_count == 0:
            logger.error("Failed to upload any vectors, aborting test")
            sys.exit(1)

        # Verify vectors were uploaded
        stats = uploader.get_stats()
        logger.info(f"Index stats after upload: {stats}")

        # Wait a moment to ensure indexing is complete
        time.sleep(1)

        # Now delete the document
        logger.info(f"Deleting document: {test_doc_id}")
        delete_success = uploader.delete_document(test_doc_id)
        logger.info(f"Document deletion {'succeeded' if delete_success else 'failed'}")

        # Verify vectors were deleted
        stats_after = uploader.get_stats()
        logger.info(f"Index stats after deletion: {stats_after}")

        # Test complete
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()