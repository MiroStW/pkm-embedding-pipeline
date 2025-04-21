#!/usr/bin/env python3
"""
Direct test for the PineconeClient class
"""
import os
import sys
import logging
from dotenv import load_dotenv
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def generate_random_embedding(dimension=1536):
    """Generate a random normalized embedding vector"""
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    # Normalize the vector
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x/magnitude for x in vector]

def main():
    # Import pinecone client
    from src.database.pinecone_client import PineconeClient

    # Get environment variables
    api_key = os.getenv('PINECONE_API_KEY')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    logger.info(f"API Key: {'*' * 8}{api_key[-4:] if api_key else 'not set'}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Index Name: {index_name}")

    try:
        # Create client
        logger.info("Creating PineconeClient...")
        client = PineconeClient(
            api_key=api_key,
            environment=environment,
            index_name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            serverless=True,
            cloud_provider="aws",
            region="us-east-1"  # Free tier region
        )

        # Check connection
        logger.info("Checking connection...")
        if client.check_connection():
            logger.info("✅ Connection successful")
        else:
            logger.error("❌ Connection failed")
            return

        # Get stats
        logger.info("Getting index statistics...")
        stats = client.get_stats()
        logger.info(f"Index stats: {stats}")
        logger.info(f"Total vectors: {stats.get('totalVectorCount', 0)}")

        # Create document chunks and upload
        logger.info("Creating test document...")
        document_id = f"test_{random.randint(10000000, 99999999)}"
        chunks = [
            {
                "content": "This is a test document for the Pinecone integration.",
                "metadata": {
                    "section_title": "Test Document",
                    "section_level": 1
                }
            }
        ]

        # Generate embeddings
        embeddings = [generate_random_embedding(1536)]

        # Upload document
        logger.info(f"Uploading document with ID: {document_id}")
        success_count, error_count = client.upload_document_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        logger.info(f"Upload results: {success_count} vectors uploaded, {error_count} errors")

        # Get stats after upload
        stats = client.get_stats()
        logger.info(f"Total vectors after upload: {stats.get('totalVectorCount', 0)}")

        # Query for the document
        logger.info("Querying for the document...")
        results = client.query_vectors(
            query_vector=embeddings[0],
            top_k=1
        )

        if results:
            logger.info(f"Found {len(results)} results")
            logger.info(f"Top result ID: {results[0].get('id', 'unknown')}")
            logger.info(f"Top result score: {results[0].get('score', 0)}")
            logger.info(f"Top result metadata: {results[0].get('metadata', {})}")
        else:
            logger.info("No results found")

        # Delete the document
        logger.info(f"Deleting document: {document_id}")
        if client.delete_document(document_id):
            logger.info(f"✅ Document {document_id} deleted successfully")
        else:
            logger.error(f"❌ Failed to delete document {document_id}")

        # Get stats after deletion
        stats = client.get_stats()
        logger.info(f"Total vectors after deletion: {stats.get('totalVectorCount', 0)}")

        logger.info("PineconeClient test completed successfully!")

    except Exception as e:
        logger.error(f"Error during PineconeClient test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()