import os
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
import hashlib # Import hashlib
import json # Import json for metadata parsing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Print environment info (without exposing API key)
api_key = os.getenv('PINECONE_API_KEY')
env = os.getenv('PINECONE_ENVIRONMENT')
index_name = os.getenv('PINECONE_INDEX_NAME')

logger.info(f"Environment loaded:")
logger.info(f"API Key present: {bool(api_key)}")
logger.info(f"Environment: {env}")
logger.info(f"Index name: {index_name}")

try:
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    pc = Pinecone(api_key=api_key)

    # List available indexes
    logger.info("Listing available indexes...")
    indexes = pc.list_indexes()
    logger.info(f"Available indexes: {indexes.names()}")

    # Connect to index
    logger.info(f"Connecting to index {index_name}...")
    index = pc.Index(index_name)

    # Base directory for dummy files
    base_dir = "dummy-md-files"

    # List of files to check (relative paths)
    files_to_check = [
        'logs.journal.2023-11-17-journal.md',
        '👨‍👩‍👧‍👧-family.kids.baby-names.md',
        'logs.meetings.2024-09-12-phillip-wenig.md',
        'logs.journal.2025.04.04.md',
        'tech.ai.prompting.techniques.chain-of-thought.md'
    ]

    print("\nChecking for files in Pinecone using document_id...")
    print("-" * 50)

    # Get total vector count first
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")

    # Query for each file using its expected document_id (SHA256 hash of absolute path)
    found_count = 0
    for file_name in files_to_check:
        try:
            # Construct the full absolute path
            full_file_path = os.path.abspath(os.path.join(base_dir, file_name))
            # Calculate the expected document_id (must match generation in DocumentProcessor)
            expected_doc_id = hashlib.sha256(full_file_path.encode('utf-8')).hexdigest()
            logger.info(f"Checking file: {file_name} (Expected ID: {expected_doc_id})")

            # Query with document_id filter
            result = index.query(
                vector=[0] * 1024,  # Dummy vector
                top_k=1,           # Need only 1 match to confirm existence
                include_metadata=True,
                filter={
                    "document_id": {"$eq": expected_doc_id}
                }
            )

            if result.matches:
                print(f"✅ Found: {file_name} (ID: {expected_doc_id})")
                found_count += 1
                # Optionally, get and print chunk count for this document_id
                # Note: This requires another query or adjusting top_k in the first query
                # For simplicity, we just confirm existence here.
                # Example: Print metadata from the first match
                if result.matches[0].metadata:
                     # Safely parse metadata if needed
                     try:
                         meta_str = json.dumps(result.matches[0].metadata, indent=2, ensure_ascii=False)
                         logger.info(f"Sample metadata: {meta_str}")
                     except Exception as json_err:
                          logger.warning(f"Could not format metadata: {json_err}")
                          logger.info(f"Raw metadata: {result.matches[0].metadata}")

            else:
                print(f"❌ Not found: {file_name} (ID: {expected_doc_id})")

        except Exception as e:
            logger.error(f"Error checking file {file_name}: {str(e)}")

    print("-" * 50)
    print(f"Found {found_count} out of {len(files_to_check)} files by document_id.")
    print(f"Total vectors in index: {stats.total_vector_count}")
    print(f"Namespaces: {stats.namespaces}")

except Exception as e:
    logger.error(f"Error: {str(e)}")
    raise