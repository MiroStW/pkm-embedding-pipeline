#!/usr/bin/env python3
"""
Script to process a document and verify it gets stored in Pinecone.
Uses the patching mechanism to monitor vector database creation.
"""
import os
import sys
import logging
import asyncio
import time
import uuid
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monkey_patch_factory():
    """Monkey patch the vector_db_factory to log client creation."""
    try:
        # Add the project root to the path
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.insert(0, root_dir)

        # Import the module
        import src.database.vector_db_factory as factory

        # Get the original function
        original_func = factory.create_vector_db_uploader

        # Define the patched function
        async def patched_create_vector_db_uploader(*args, **kwargs):
            """Patched version that logs the creation."""
            logger.info(f"🔍 create_vector_db_uploader called with:")
            if args:
                logger.info(f"  Args: {args}")

            # Call the original function
            result = await original_func(*args, **kwargs)

            # Log the result
            logger.info(f"🔍 Factory returned uploader of type: {type(result).__name__}")

            # Check if it's a mock
            from src.database.mock_vector_db import MockVectorDatabaseUploader
            if isinstance(result, MockVectorDatabaseUploader):
                logger.warning("⚠️ USING MOCK VECTOR DATABASE UPLOADER!")
            else:
                logger.info("✅ USING REAL VECTOR DATABASE UPLOADER")

            return result

        # Apply the patch
        factory.create_vector_db_uploader = patched_create_vector_db_uploader
        logger.info("✅ Successfully patched vector_db_factory")

    except Exception as e:
        logger.error(f"Failed to patch vector_db_factory: {str(e)}")

async def process_file_manually(file_path):
    """Manually process a file to verify Pinecone storage."""
    try:
        # Import required modules
        from src.config import load_config
        from src.processors import DocumentProcessor
        from src.database.vector_db_factory import create_vector_db_uploader

        # Load environment variables for Pinecone
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT')
        index_name = os.getenv('PINECONE_INDEX_NAME')

        logger.info(f"Pinecone Configuration:")
        logger.info(f"  API Key set: {'Yes' if api_key else 'No'}")
        logger.info(f"  Environment set: {'Yes' if environment else 'No'}")
        logger.info(f"  Index name set: {'Yes' if index_name else 'No'}")

        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Explicitly override vector_db configuration
        config['vector_db'] = {
            'provider': 'pinecone',
            'api_key': api_key,
            'environment': environment,
            'index_name': index_name,
            'dimension': 1024,
            'serverless': False,
            'max_retries': 3,
            'retry_delay': 2.0,
            'batch_size': 100
        }

        # Create document processor
        logger.info("Creating document processor...")
        processor = DocumentProcessor(config)

        # Create vector database uploader with explicit config
        logger.info("Creating vector database uploader...")
        vector_db = await create_vector_db_uploader({"vector_db": config['vector_db']})

        # Check if using mock
        from src.database.mock_vector_db import MockVectorDatabaseUploader
        if isinstance(vector_db, MockVectorDatabaseUploader):
            logger.error("Using MOCK vector database - this is not what we want!")
            return False
        else:
            logger.info(f"Using real vector database: {type(vector_db).__name__}")

        # Get stats before processing
        before_stats = vector_db.get_stats()
        initial_count = before_stats.get('total_vector_count', 0)
        logger.info(f"Initial vector count: {initial_count}")

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
        final_count = after_stats.get('total_vector_count', 0)
        logger.info(f"Final vector count: {final_count}")

        # Check if vectors were added
        if final_count > initial_count:
            logger.info(f"✅ Added {final_count - initial_count} new vectors to Pinecone")
            return True
        else:
            logger.warning("⚠️ No new vectors were added to Pinecone")
            return False

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return False

async def main():
    """Main function."""
    # Load environment variables
    load_dotenv()

    # Patch the factory
    monkey_patch_factory()

    # Get document path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "dummy-md-files/tech.ai.prompting.techniques.chain-of-thought.md"

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1

    # Process the document
    success = await process_file_manually(file_path)

    if success:
        logger.info("✅ Document was successfully processed and stored in Pinecone")
        return 0
    else:
        logger.error("❌ Document processing or storage failed")
        return 1

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    exit_code = asyncio.run(main())
    sys.exit(exit_code)