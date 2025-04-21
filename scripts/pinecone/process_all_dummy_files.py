#!/usr/bin/env python3
"""
Script to process all dummy markdown files and store them in Pinecone.
"""
import os
import sys
import logging
import asyncio
from dotenv import load_dotenv

# Add the project root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_document(file_path):
    """Process a document and store it in Pinecone."""
    try:
        # Import required modules
        from src.config import load_config
        from src.processors import DocumentProcessor
        from src.database.vector_db_factory import create_vector_db_uploader

        # Load configuration
        config = load_config()

        # Create document processor
        logger.info(f"Creating document processor for {file_path}...")
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

        logger.info(f"Document {file_path} indexed successfully")
        return True

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
        return False

async def process_all_documents():
    """Process all dummy markdown files."""
    # Get list of dummy files
    dummy_dir = "dummy-md-files"
    files = [
        os.path.join(dummy_dir, f) for f in os.listdir(dummy_dir)
        if f.endswith('.md')
    ]

    logger.info(f"Found {len(files)} files to process")

    # Process each file
    results = []
    for file_path in files:
        logger.info(f"=== Processing {file_path} ===")
        success = await process_document(file_path)
        results.append((file_path, success))

    # Print summary
    logger.info("=== Processing Summary ===")
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Successfully processed {success_count} out of {len(results)} files")

    # List failures if any
    failures = [(path, success) for path, success in results if not success]
    if failures:
        logger.error("Failed to process the following files:")
        for path, _ in failures:
            logger.error(f" - {path}")

    return success_count == len(results)

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Process all documents
    success = await process_all_documents()

    # Run the check_files script to verify files are in Pinecone
    if success:
        logger.info("Running check_files.py to verify documents in Pinecone...")
        check_script_path = os.path.join(os.path.dirname(__file__), 'check_files.py')
        os.system(f"python {check_script_path}")

    return 0 if success else 1

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sys.exit(asyncio.run(main()))