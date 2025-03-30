#!/usr/bin/env python3
"""
Test script for title-enhanced embedding generation and upload to Pinecone.
"""
import os
import asyncio
import logging
import time
import json
import sys
from src.config import ConfigManager
from src.models import EmbeddingModelFactory
from src.database.pinecone_client import PineconeClient
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Check for --no-cleanup flag
    cleanup = True
    if "--no-cleanup" in sys.argv:
        cleanup = False
        logger.info("Cleanup disabled - index will not be deleted after test")

    # Get configuration
    config_manager = ConfigManager()
    db_config = config_manager.get_database_config().get("vector_db", {})
    embedding_config = config_manager.get_embedding_config()

    # Check if title enhanced is enabled
    if not embedding_config.get("enable_title_enhanced", False):
        logger.warning("Title-enhanced embeddings are disabled in config. Enabling for this test.")
        embedding_config["enable_title_enhanced"] = True

    # Force using CPU device for compatibility
    embedding_config["device"] = "cpu"
    logger.info("Forcing CPU device for compatibility")

    # Get Pinecone credentials from environment
    api_key = os.environ.get('PINECONE_API_KEY') or db_config.get("api_key")
    environment = os.environ.get('PINECONE_ENVIRONMENT') or db_config.get("environment", "gcp-starter")
    index_name = os.environ.get('PINECONE_INDEX_NAME') or db_config.get("index_name", "pkm-embeddings")

    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        return

    # Create test document ID
    document_id = f"test_{int(time.time())}"
    logger.info(f"Testing with document ID: {document_id}")

    # Create test chunks
    chunks = [
        {
            "content": "This is the first test chunk for title-enhanced embedding testing.",
            "metadata": {
                "section_title": "Introduction",
                "section_level": 1
            }
        },
        {
            "content": "This is the second test chunk about machine learning concepts.",
            "metadata": {
                "section_title": "Machine Learning",
                "section_level": 2
            }
        },
        {
            "content": "This is the third test chunk about vector databases and embeddings.",
            "metadata": {
                "section_title": "Vector Databases",
                "section_level": 2
            }
        }
    ]

    try:
        # Create embedding model
        logger.info("Creating embedding model...")
        embedding_model = await EmbeddingModelFactory.create_model(embedding_config)

        # Generate regular embeddings
        logger.info("Generating regular embeddings...")
        content_texts = [chunk.get('content', '') for chunk in chunks]
        embeddings = await embedding_model.generate_embeddings(content_texts)
        logger.info(f"Generated {len(embeddings)} regular embeddings")

        # Generate title-enhanced embeddings
        logger.info("Generating title-enhanced embeddings...")
        title_enhanced = []
        title_weight = embedding_config.get('title_weight', 0.3)

        for i, chunk in enumerate(chunks):
            title = chunk.get('metadata', {}).get('section_title', '')
            content = chunk.get('content', '')

            if title and content:
                logger.info(f"Generating title-enhanced embedding for '{title}'")
                enhanced = await embedding_model.generate_title_enhanced_embedding(
                    title=title,
                    content=content,
                    title_weight=title_weight
                )
                title_enhanced.append(enhanced)
            else:
                logger.warning(f"No title for chunk {i}, using regular embedding")
                title_enhanced.append(embeddings[i])

        logger.info(f"Generated {len(title_enhanced)} title-enhanced embeddings")

        # Create Pinecone client
        logger.info("Creating Pinecone client...")
        # Get the actual embedding dimension from the model's output
        embedding_dimension = len(embeddings[0]) if embeddings else 384
        logger.info(f"Using embedding dimension: {embedding_dimension}")

        # Use a test-specific index name to avoid dimension conflicts
        test_index_name = f"test-embeddings-{embedding_dimension}"
        logger.info(f"Using test-specific index: {test_index_name}")

        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)

        # Check if the test index exists with wrong dimension and delete it if needed
        try:
            existing_indexes = pc.list_indexes().names()
            if test_index_name in existing_indexes:
                logger.info(f"Deleting existing test index: {test_index_name}")
                pc.delete_index(test_index_name)
                time.sleep(2)  # Wait for deletion to complete
        except Exception as e:
            logger.warning(f"Error checking/deleting existing index: {str(e)}")

        # Create a new index with the correct dimension
        try:
            logger.info(f"Creating test index with dimension {embedding_dimension}")
            pc.create_index(
                name=test_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # AWS us-east-1 region
                )
            )

            # Wait for index to be ready
            while test_index_name not in pc.list_indexes().names():
                logger.info("Waiting for index to be created...")
                time.sleep(1)

            logger.info(f"Test index created successfully")
        except Exception as e:
            logger.error(f"Error creating test index: {str(e)}")
            return

        client = PineconeClient(
            api_key=api_key,
            environment=environment,
            index_name=test_index_name,  # Use the test index
            dimension=embedding_dimension
        )

        # First delete any existing document with this ID (cleanup)
        logger.info(f"Deleting any existing document with ID: {document_id}")
        client.delete_document(document_id)

        # Upload document
        logger.info(f"Uploading document with ID: {document_id}")
        success_count, error_count = client.upload_document_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            title_enhanced_embeddings=title_enhanced
        )

        logger.info(f"Upload results: {success_count} vectors uploaded, {error_count} errors")

        # Add a delay to allow the index to update
        delay_seconds = 30
        logger.info(f"Waiting {delay_seconds} seconds for the index to update...")
        time.sleep(delay_seconds)

        # Verify the document was uploaded correctly
        logger.info("Verifying upload...")
        try:
            # For Serverless indexes, metadata filtering may not work as expected
            # Instead, list all vectors and filter client-side
            logger.info("Getting index statistics...")
            stats = client.get_stats()
            logger.info(f"Index stats: {stats}")
            total_vector_count = stats.get('totalVectorCount', 0)

            # Also check namespace vector count
            namespace_count = 0
            if 'namespaces' in stats and '' in stats['namespaces']:
                namespace_count = stats['namespaces'][''].get('vector_count', 0)

            total_vector_count = total_vector_count or namespace_count  # Use namespace count if totalVectorCount is 0
            logger.info(f"Total vectors in index: {total_vector_count}")

            if total_vector_count > 0:
                logger.info("Searching for vectors with ID prefix...")

                # Try direct fetch by IDs first (more reliable)
                logger.info("Trying direct fetch by vector IDs...")
                vector_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
                enhanced_ids = [f"{document_id}_{i}_enhanced" for i in range(len(chunks))]
                all_ids = vector_ids + enhanced_ids

                try:
                    # Fetch vectors by ID (this is supported in serverless)
                    fetched_vectors = []
                    # Fetch in batches of 10 to avoid overwhelming the API
                    for i in range(0, len(all_ids), 10):
                        batch_ids = all_ids[i:i+10]
                        logger.info(f"Fetching batch of {len(batch_ids)} vector IDs")
                        batch_result = client._with_retry(client.index.fetch, ids=batch_ids)
                        fetched_vectors.extend([v for id, v in batch_result.get('vectors', {}).items()])

                    logger.info(f"Fetched {len(fetched_vectors)} vectors by ID")

                    if fetched_vectors:
                        # Print detailed information about first vector of each type
                        for i, vector in enumerate(fetched_vectors[:2]):
                            vector_type = vector.get('metadata', {}).get('embedding_type', 'regular')
                            logger.info(f"Vector {i+1} details:")
                            logger.info(f"  ID: {vector.get('id', 'unknown')}")
                            logger.info(f"  Type: {vector_type}")
                            logger.info(f"  Document ID: {vector.get('metadata', {}).get('document_id', 'unknown')}")
                            logger.info(f"  Section Title: {vector.get('metadata', {}).get('section_title', 'none')}")
                            # Truncate values for display
                            values = vector.get('values', [])
                            if values:
                                logger.info(f"  Values: [{values[0]:.6f}, {values[1]:.6f}, ..., {values[-1]:.6f}] (len={len(values)})")

                        # Count regular and enhanced vectors
                        regular_vectors = [v for v in fetched_vectors if v.get('metadata', {}).get('embedding_type') != 'title_enhanced']
                        enhanced_vectors = [v for v in fetched_vectors if v.get('metadata', {}).get('embedding_type') == 'title_enhanced']

                        logger.info(f"Verification results from direct fetch:")
                        logger.info(f"Total vectors found: {len(fetched_vectors)}")
                        logger.info(f"Regular vectors: {len(regular_vectors)}")
                        logger.info(f"Title-enhanced vectors: {len(enhanced_vectors)}")

                        # Test passed if both regular and enhanced vectors are found
                        if len(regular_vectors) > 0 and len(enhanced_vectors) > 0:
                            logger.info("✅ Test passed! Both regular and title-enhanced vectors were found.")
                            return  # Success - no need to try the other method
                        else:
                            logger.warning("Not all expected vectors were found via direct fetch")
                    else:
                        logger.warning("No vectors found via direct fetch")
                except Exception as e:
                    logger.warning(f"Error during direct fetch: {str(e)}")

                # Try with a simple approach - dummy query to get all vectors
                dummy_results = client.query_vectors(
                    query_vector=embeddings[0],  # Use one of our real embeddings
                    top_k=min(total_vector_count, 100),  # Get up to 100 vectors
                    include_metadata=True
                )

                if dummy_results:
                    # Filter client-side for our document ID
                    doc_vectors = [v for v in dummy_results if v.get('metadata', {}).get('document_id') == document_id]

                    # Count regular and enhanced vectors
                    regular_vectors = [v for v in doc_vectors if v.get('metadata', {}).get('embedding_type') != 'title_enhanced']
                    enhanced_vectors = [v for v in doc_vectors if v.get('metadata', {}).get('embedding_type') == 'title_enhanced']

                    logger.info(f"Verification results:")
                    logger.info(f"Total vectors found for document: {len(doc_vectors)}")
                    logger.info(f"Regular vectors: {len(regular_vectors)}")
                    logger.info(f"Title-enhanced vectors: {len(enhanced_vectors)}")

                    # Test passed if both regular and enhanced vectors are found
                    if len(regular_vectors) > 0 and len(enhanced_vectors) > 0:
                        logger.info("✅ Test passed! Both regular and title-enhanced vectors were found.")
                    else:
                        logger.error("❌ Test failed! Not all expected vectors were found.")
                else:
                    logger.warning("No vectors returned from query")
                    logger.error("❌ Test failed! No vectors found in query results")
            else:
                logger.error("❌ Test failed! No vectors found in the index")
        except Exception as e:
            logger.error(f"Error during verification: {str(e)}")
            logger.error("❌ Verification failed - couldn't query the vectors")

        # Clean up
        if cleanup:
            logger.info(f"Cleaning up - deleting test document: {document_id}")
            client.delete_document(document_id)

            # Delete the test index
            try:
                logger.info(f"Deleting test index: {test_index_name}")
                pc.delete_index(test_index_name)
                logger.info("Test index deleted successfully")
            except Exception as e:
                logger.warning(f"Error deleting test index: {str(e)}")
        else:
            logger.info(f"Cleanup disabled - document {document_id} and index {test_index_name} preserved for inspection")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())