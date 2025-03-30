#!/usr/bin/env python3
"""
Script to check Pinecone for test indexes and vectors.
"""
import os
import logging
import sys
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to check Pinecone for test indexes and vectors."""
    # Get Pinecone API key from environment
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        return

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # List available indexes
    logger.info("Available indexes:")
    indexes = pc.list_indexes().names()
    logger.info(indexes)

    # Check for test indexes
    test_indexes = [idx for idx in indexes if idx.startswith('test-embeddings')]
    if test_indexes:
        logger.info(f"Found {len(test_indexes)} test indexes: {test_indexes}")

        # Connect to the first test index
        index_name = test_indexes[0]
        index = pc.Index(index_name)

        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats for {index_name}:")
        logger.info(stats)

        # Determine test document ID
        test_doc_id = None
        if len(sys.argv) > 1:
            test_doc_id = sys.argv[1]
            logger.info(f"Using document ID from command line: {test_doc_id}")
        else:
            # Try with a likely test ID
            test_doc_id = f"test_1743363795"
            logger.info(f"No document ID specified, using default: {test_doc_id}")

        # Try to fetch vectors for different chunk indices
        logger.info(f"Attempting to fetch vectors with IDs starting with {test_doc_id}")

        # Try with different possible IDs
        for i in range(3):
            vector_id = f"{test_doc_id}_{i}"
            enhanced_id = f"{test_doc_id}_{i}_enhanced"

            try:
                result = index.fetch(ids=[vector_id, enhanced_id])
                logger.info(f"Fetched vectors with IDs {vector_id}, {enhanced_id}:")

                # Access vectors directly as a property
                if hasattr(result, 'vectors') and result.vectors:
                    logger.info(f"Found {len(result.vectors)} vectors")

                    for id, vector in result.vectors.items():
                        is_enhanced = "_enhanced" in id
                        vector_type = "title_enhanced" if is_enhanced else "regular"

                        logger.info(f"Vector ID: {id}")
                        logger.info(f"  Type: {vector_type}")

                        # Access metadata as a property
                        if hasattr(vector, 'metadata'):
                            metadata = vector.metadata
                            logger.info(f"  Document ID: {metadata.get('document_id', 'unknown')}")
                            logger.info(f"  Section Title: {metadata.get('section_title', 'none')}")

                        # Access values as a property
                        if hasattr(vector, 'values') and vector.values:
                            values = vector.values
                            if len(values) > 2:
                                logger.info(f"  Values: [{values[0]:.6f}, {values[1]:.6f}, ..., {values[-1]:.6f}] (len={len(values)})")
                            else:
                                logger.info(f"  Values: {values} (len={len(values)})")
                else:
                    logger.info(f"No vectors found with these IDs")
            except Exception as e:
                logger.warning(f"Error fetching vectors: {str(e)}")
    else:
        logger.info("No test indexes found")

if __name__ == "__main__":
    main()