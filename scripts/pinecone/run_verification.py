#!/usr/bin/env python3
"""
Main script to run all Pinecone verification steps.
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

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def run_connection_test():
    """Run the simple connection test."""
    logger.info("=== RUNNING SIMPLE CONNECTION TEST ===")
    from scripts.pinecone.simple_connection_test import main as connection_test
    return connection_test()

def run_verification():
    """Run the full verification suite."""
    logger.info("=== RUNNING FULL VERIFICATION SUITE ===")
    from scripts.pinecone.verify_integration import main as verification
    return verification()

def run_cleanup():
    """Run cleanup to remove test vectors."""
    logger.info("=== RUNNING CLEANUP ===")
    from scripts.pinecone.cleanup_vectors import main as cleanup
    return cleanup()

def main():
    """Run all verification steps."""
    # Load environment variables
    load_dotenv()

    # Check environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    if not api_key or not index_name:
        logger.error("Missing required environment variables: PINECONE_API_KEY, PINECONE_INDEX_NAME")
        sys.exit(1)

    logger.info("Starting Pinecone verification...")

    # Step 1: Test connection
    try:
        run_connection_test()
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")

    # Step 2: Run verification suite
    try:
        run_verification()
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")

    # Step 3: Run cleanup
    try:
        run_cleanup()
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

    logger.info("Verification process completed.")

if __name__ == "__main__":
    main()