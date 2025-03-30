#!/usr/bin/env python3
import sys
import os
import logging
from src.database.pinecone_client import PineconeClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_doc_vectors.py <document_id>")
        sys.exit(1)

    document_id = sys.argv[1]

    # Get Pinecone credentials from environment
    api_key = os.environ.get('PINECONE_API_KEY')
    environment = os.environ.get('PINECONE_ENVIRONMENT', 'gcp-starter')
    index_name = os.environ.get('PINECONE_INDEX_NAME', 'pkm-embeddings')

    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    # Create Pinecone client
    client = PineconeClient(
        api_key=api_key,
        environment=environment,
        index_name=index_name,
        dimension=1536
    )

    # Query for vectors with the document ID
    vectors = client.query_vectors(
        query_vector=[0.1] * client.dimension,  # Dummy vector for metadata query
        top_k=100,  # Set high to get all vectors
        filter={"document_id": {"$eq": document_id}},
        include_metadata=True
    )

    # Count regular and enhanced vectors
    regular_vectors = [v for v in vectors if v.get('metadata', {}).get('embedding_type') != 'title_enhanced']
    enhanced_vectors = [v for v in vectors if v.get('metadata', {}).get('embedding_type') == 'title_enhanced']

    logger.info(f"Document ID: {document_id}")
    logger.info(f"Total vectors found: {len(vectors)}")
    logger.info(f"Regular vectors: {len(regular_vectors)}")
    logger.info(f"Title-enhanced vectors: {len(enhanced_vectors)}")

    # Print chunk info
    if vectors:
        logger.info("Sample chunks:")
        for i, vector in enumerate(vectors[:5]):  # Show first 5 chunks
            metadata = vector.get('metadata', {})
            logger.info(f"  {i+1}. ID: {vector.get('id')}")
            logger.info(f"     Type: {metadata.get('embedding_type', 'regular')}")
            logger.info(f"     Chunk Index: {metadata.get('chunk_index')}")
            text = metadata.get('text', '')[:50]
            logger.info(f"     Text: '{text}...'")

if __name__ == "__main__":
    main()