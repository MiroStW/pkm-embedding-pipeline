#!/usr/bin/env python3
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Define file moves based on their function
    moves = [
        # Test files to tests directory
        ("test_title_enhanced.py", "tests/"),

        # Pinecone utility scripts to scripts/pinecone
        ("check_test_vectors.py", "scripts/pinecone/"),
        ("cleanup_test_data.py", "scripts/pinecone/"),
        ("count_doc_vectors.py", "scripts/pinecone/"),
        ("verify_pinecone_integration.py", "scripts/pinecone/"),
    ]

    # Execute moves
    for source, dest_dir in moves:
        if not os.path.exists(source):
            logger.warning(f"Source file not found: {source}")
            continue

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Check if file already exists in destination
        dest_path = os.path.join(dest_dir, source)
        if os.path.exists(dest_path):
            logger.warning(f"File already exists at destination: {dest_path}")
            continue

        # Move the file
        try:
            shutil.move(source, dest_path)
            logger.info(f"Moved {source} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to move {source}: {e}")

    logger.info("File organization complete")

if __name__ == "__main__":
    main()