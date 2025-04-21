#!/usr/bin/env python3
"""
Script to process a single document and verify it's properly stored in Pinecone.
"""
import os
import sys
import logging
import asyncio
import time
from dotenv import load_dotenv

# Add the project root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_document(file_path):
    """Process a document and verify it's stored in Pinecone."""
    try:
        # Import required modules
        from src.config import load_config
        from src.processors import DocumentProcessor
        from src.database.vector_db_factory import create_vector_db_uploader

        # Load configuration
        config = load_config()

        # Override with environment variables
        config['vector_db'] = {
            'provider': 'pinecone',
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT'),
            'index_name': os.getenv('PINECONE_INDEX_NAME'),
            'dimension': 1024,
            'serverless': False,
            'max_retries': 3,
            'retry_delay': 2.0,
            'batch_size': 100
        }

        # Create document processor
        logger.info("Creating document processor...")
        processor = DocumentProcessor(config)

        # Create vector database uploader
        logger.info("Creating vector database uploader...")
        vector_db = await create_vector_db_uploader(config)

        # Check if using mock
        from src.database.mock_vector_db import MockVectorDatabaseUploader
        if isinstance(vector_db, MockVectorDatabaseUploader):
            logger.error("Using MOCK vector database - this is not what we want!")
            return False
        else:
            logger.info(f"Using real vector database: {type(vector_db).__name__}")

        # Get stats before processing
        before_stats = vector_db.get_stats()
        logger.info(f"Vector database stats before processing: {before_stats}")

        # Process the document
        logger.info(f"Processing document: {file_path}")
        result = processor.process_file(file_path)

        # Check if processing was successful
        if result['status'] != 'success':
            logger.error(f"Document processing failed: {result.get('error', 'Unknown error')}")
            return False

        logger.info(f"Document processed successfully with {len(result['chunks'])} chunks")

        # Index the document
        logger.info("Indexing document in vector database...")
        success = await vector_db.index_document(result)

        if not success:
            logger.error("Failed to index document in vector database")
            return False

        logger.info("Document indexed successfully")

        # Get stats after processing
        time.sleep(2)  # Wait for indexing to complete
        after_stats = vector_db.get_stats()
        logger.info(f"Vector database stats after processing: {after_stats}")

        # Verify document is in vector database
        doc_id = result['metadata']['id']
        logger.info(f"Verifying document {doc_id} is in vector database...")

        if hasattr(vector_db, 'verify_document_indexed'):
            is_indexed = vector_db.verify_document_indexed(doc_id)
            logger.info(f"Document indexed verification: {is_indexed}")
        else:
            logger.warning("Vector database doesn't have verify_document_indexed method")

        return True

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return False

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Get document path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "dummy-md-files/tech.ai.prompting.techniques.chain-of-thought.md"

    # Ensure file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1

    # Process document
    success = await process_document(file_path)

    return 0 if success else 1

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sys.exit(asyncio.run(main()))