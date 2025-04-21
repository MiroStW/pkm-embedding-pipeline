#!/usr/bin/env python3
"""
Script to delete and recreate the Pinecone index specified in environment variables.
"""
import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()

    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')
    environment = os.getenv('PINECONE_ENVIRONMENT')
    dimension = int(os.getenv('PINECONE_DIMENSION', '1024'))
    metric = os.getenv('PINECONE_METRIC', 'cosine')
    serverless = os.getenv('PINECONE_SERVERLESS', 'true').lower() == 'true'
    cloud_provider = os.getenv('PINECONE_CLOUD_PROVIDER', 'aws')
    region = os.getenv('PINECONE_REGION', 'us-east-1')

    if not api_key or not index_name or not environment:
        logger.error("Missing required environment variables.")
        sys.exit(1)

    try:
        from pinecone import Pinecone, ServerlessSpec, PodSpec

        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        logger.info("Successfully connected to Pinecone")
        logger.info(f"Using cloud_provider={cloud_provider}, region={region}")

        # Check if index exists
        indexes = pc.list_indexes().names()
        if index_name in indexes:
            logger.info(f"Deleting existing index '{index_name}'...")
            pc.delete_index(index_name)
            # Wait for deletion
            while index_name in pc.list_indexes().names():
                logger.info("Waiting for index to be deleted...")
                import time; time.sleep(1)
            logger.info(f"Index '{index_name}' deleted.")
        else:
            logger.info(f"Index '{index_name}' does not exist, nothing to delete.")

        # Create the index
        logger.info(f"Creating index '{index_name}' with dimension {dimension}, metric {metric}, serverless={serverless}, cloud_provider={cloud_provider}, region={region}")
        if serverless:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud_provider,
                    region=region
                )
            )
        else:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=PodSpec(
                    environment=environment,
                    pod_type="p1.x1"
                )
            )
        # Wait for index to be ready
        while index_name not in pc.list_indexes().names():
            logger.info("Waiting for index to be created...")
            import time; time.sleep(1)
        logger.info(f"Index '{index_name}' created successfully.")

    except Exception as e:
        logger.error(f"Error during index reset: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()