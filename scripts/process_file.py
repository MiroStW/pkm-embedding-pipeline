#!/usr/bin/env python3
"""
Script to process a single markdown file and index it in the vector database.
"""
import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.processors import DocumentProcessor
from src.database.vector_db_factory import create_vector_db_uploader

# Configure logging
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_file(file_path: str) -> bool:
    """
    Process a single markdown file and index it in the vector database.

    Args:
        file_path: Path to the markdown file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Handle file path with special characters
        file_path = os.path.normpath(os.fsdecode(file_path))
        logger.debug(f"Processing file: {file_path}")

        # Load configuration
        config = ConfigManager().load_config()
        logger.debug(f"Loaded config: {config}")

        # Create processor and vector database uploader
        processor = DocumentProcessor(config)
        logger.debug("Created document processor")

        try:
            vector_db = await create_vector_db_uploader(config)
            logger.debug(f"Created vector database uploader: {type(vector_db).__name__}")
        except Exception as e:
            logger.error(f"Error creating vector database uploader: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

        if not vector_db:
            logger.error("Failed to create vector database uploader")
            return False

        # Process the file
        logger.info(f"Processing file: {file_path}")
        try:
            result = processor.process_file(file_path)
            logger.debug(f"Processing result: {result}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

        if result.get('status') != 'success':
            logger.error(f"Failed to process file: {result.get('error', 'Unknown error')}")
            return False

        # Index the document
        logger.info("Indexing document in vector database")
        try:
            success = await vector_db.index_document(result)
            logger.debug(f"Indexing result: {success}")
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

        if success:
            logger.info("Successfully processed and indexed file")
        else:
            logger.error("Failed to index document")

        return success

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: process_file.py <markdown_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Run the async process_file function
    success = asyncio.run(process_file(file_path))
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()