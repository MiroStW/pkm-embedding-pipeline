#!/usr/bin/env python3
"""
Simplified verification script to test if Pinecone is being used correctly.
"""
import os
import sys
import logging
from dotenv import load_dotenv
from pinecone import Pinecone

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Verify Pinecone configuration and connection."""
    # Load environment variables
    load_dotenv()

    # Get environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    print(f"\n=== PINECONE VERIFICATION ===")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print(f"Environment: {'Set' if environment else 'Not set'}")
    print(f"Index Name: {'Set' if index_name else 'Not set'}")

    # Check for missing configuration
    if not all([api_key, environment, index_name]):
        print(f"\n❌ ERROR: Missing required Pinecone configuration")
        return 1

    try:
        # Connect to Pinecone
        print(f"\n=== CONNECTING TO PINECONE ===")
        pc = Pinecone(api_key=api_key)

        # List indexes
        indexes = pc.list_indexes().names()
        print(f"Available indexes: {indexes}")

        # Check if our index exists
        if index_name not in indexes:
            print(f"\n❌ ERROR: Index '{index_name}' not found")
            return 1

        print(f"✅ Index '{index_name}' found")

        # Connect to index
        index = pc.Index(index_name)

        # Get stats
        stats = index.describe_index_stats()
        print(f"\n=== INDEX STATISTICS ===")
        print(f"Dimension: {stats.get('dimension')}")
        print(f"Total vectors: {stats.get('total_vector_count')}")

        # Done
        print(f"\n✅ Pinecone connection verified successfully")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())