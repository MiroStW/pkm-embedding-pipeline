#!/usr/bin/env python
"""
Runner script to execute the RAG test with the scones.md file.

Usage:
    python tests/pinecone/run_rag_test.py
"""
import os
import sys
import logging
import time
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.processors.document_processor import DocumentProcessor
from src.models.embedding_factory import EmbeddingModelFactory
from src.database.vector_db_factory import create_vector_db_uploader
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_test_runner")

def safe_json_serialize(obj):
    """
    Safely convert an object to a JSON-serializable representation.
    Handles special types like ScoredVector from Pinecone.
    """
    if hasattr(obj, '__dict__'):
        # Convert to dictionary for objects with __dict__
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    elif isinstance(obj, dict):
        # Handle dictionaries
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples
        return [safe_json_serialize(x) for x in obj]
    elif hasattr(obj, 'to_dict'):
        # Use to_dict() method if available
        return safe_json_serialize(obj.to_dict())
    else:
        # Try to use the object directly, or convert to string if not serializable
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

async def async_run_rag_test():
    """
    Run the end-to-end RAG test with the scones.md file asynchronously.
    """
    start_time = time.time()
    logger.info("Starting RAG test in production-like mode")

    # Step 1: Load configuration
    logger.info("Loading test configuration")
    config_path = os.path.join("config", "test_config.yaml")
    config = load_config(config_path)

    # Step 2: Initialize components
    logger.info("Initializing components")
    document_processor = DocumentProcessor(config)

    try:
        # Create embedding model asynchronously
        logger.info(f"Creating embedding model using '{config.get('embedding', {}).get('model_type', 'unknown')}' model type")
        embedding_model = await EmbeddingModelFactory.create_model(config)
        logger.info(f"Using embedding model: {embedding_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to create embedding model: {str(e)}")
        logger.error("Check your environment variables and network connection")
        return False

    # Step 3: Process the test document
    try:
        logger.info(f"Processing document: scones.md")
        processed_doc = document_processor.process_file("scones.md")

        if processed_doc["status"] != "success":
            logger.error(f"Failed to process document: {processed_doc.get('reason', 'Unknown error')}")
            return False

        if len(processed_doc["chunks"]) == 0:
            logger.error("No chunks were generated from the document")
            return False

        document_id = processed_doc["metadata"]["id"]
        logger.info(f"Document processed with ID: {document_id}")
        logger.info(f"Generated {len(processed_doc['chunks'])} chunks")

        # Debug: Print structure of first chunk
        first_chunk = processed_doc["chunks"][0]
        logger.debug(f"First chunk structure: {json.dumps(first_chunk, indent=2)}")

        # Identify the content key in chunks
        content_key = "content"
        if "content" in first_chunk:
            content_key = "content"
        elif "text" in first_chunk:
            content_key = "text"
        else:
            # List all keys in the first chunk to help debugging
            keys = list(first_chunk.keys())
            logger.info(f"Available keys in chunk: {keys}")
            # Try a common key that might contain text
            for possible_key in ["content", "text", "value", "markdown", "chunk"]:
                if possible_key in keys:
                    content_key = possible_key
                    break

        logger.info(f"Using '{content_key}' as the content key for chunks")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False

    # Step 4: Generate embeddings for chunks
    try:
        logger.info("Generating embeddings for document chunks")
        chunks = processed_doc["chunks"]
        embeddings = []

        # Generate embeddings using the async generate_embedding method
        for i, chunk in enumerate(chunks):
            chunk_text = chunk[content_key]
            # Show progress for large documents
            if i % 10 == 0 and i > 0:
                logger.info(f"Generated embeddings for {i}/{len(chunks)} chunks")

            embedding = await embedding_model.generate_embedding(chunk_text)
            embeddings.append(embedding)

            # If this is the first chunk, output its dimension
            if i == 0:
                embedding_dimension = len(embedding)
                logger.info(f"Detected embedding dimension: {embedding_dimension}")

        # Generate title-enhanced embeddings if supported
        title = processed_doc["metadata"].get("title", "")
        title_enhanced_embeddings = None

        if hasattr(embedding_model, 'generate_title_enhanced_embedding'):
            logger.info("Generating title-enhanced embeddings")
            title_enhanced_embeddings = []
            for chunk in chunks:
                chunk_text = chunk[content_key]
                enhanced_embedding = await embedding_model.generate_title_enhanced_embedding(
                    title, chunk_text
                )
                title_enhanced_embeddings.append(enhanced_embedding)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return False

    # Step 5: Create or update a test index with the correct dimension
    try:
        embedding_dimension = len(embeddings[0])
        logger.info(f"Using embedding dimension: {embedding_dimension}")

        # Use a test-specific index name with dimension
        test_index_name = f"test-rag-{embedding_dimension}"
        test_namespace = "test_namespace"

        # Get Pinecone credentials from config
        api_key = config.get("database", {}).get("vector_db", {}).get("api_key")

        if not api_key:
            logger.error("No Pinecone API key found in config")
            return False

        # Initialize Pinecone client directly
        logger.info(f"Initializing Pinecone with test index: {test_index_name}")
        pc = Pinecone(api_key=api_key)

        # Check if the index exists with correct dimension
        existing_indexes = pc.list_indexes().names()

        if test_index_name not in existing_indexes:
            # Create new test index with correct dimension
            logger.info(f"Creating test index with dimension {embedding_dimension}")
            pc.create_index(
                name=test_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            # Wait for index to be ready
            while test_index_name not in pc.list_indexes().names():
                logger.info("Waiting for index to be created...")
                time.sleep(1)

            logger.info(f"Test index created successfully")
        else:
            logger.info(f"Using existing test index: {test_index_name}")

        # Create a new vector DB client with the right dimension and index
        vector_db_config = config.get("database", {}).get("vector_db", {}).copy()
        vector_db_config["dimension"] = embedding_dimension
        vector_db_config["index_name"] = test_index_name

        # Update the config
        config["database"]["vector_db"] = vector_db_config

        # Create the vector DB client
        vector_db = create_vector_db_uploader(config)
        if vector_db is None:
            logger.error("Failed to create vector database client - got None")
            return False

        logger.info(f"Using vector database: {vector_db.__class__.__name__}")

        # Set the test namespace
        if hasattr(vector_db, "namespace"):
            vector_db.namespace = test_namespace
            logger.info(f"Using namespace: {vector_db.namespace}")
    except Exception as e:
        logger.error(f"Error setting up Pinecone index: {str(e)}")
        return False

    # Step 6: Upload embeddings to vector database
    try:
        logger.info(f"Uploading embeddings to vector database (namespace: {getattr(vector_db, 'namespace', 'default')})")

        # Delete any existing data for this document
        try:
            logger.info(f"Deleting any existing data for document ID: {document_id}")
            vector_db.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Error during cleanup (non-fatal): {str(e)}")

        # Upload the new embeddings
        upload_success = vector_db.sync_document(
            document_id=document_id,
            new_chunks=chunks,
            embeddings=embeddings,
            title_enhanced_embeddings=title_enhanced_embeddings
        )

        if not upload_success:
            logger.error("Failed to upload embeddings to vector database")
            return False
    except Exception as e:
        logger.error(f"Error uploading embeddings: {str(e)}")
        return False

    # Step 7: Verify document was added to vector database
    try:
        logger.info("Verifying document was indexed")
        is_indexed = vector_db.verify_document_indexed(document_id)

        if not is_indexed:
            logger.error("Document was not properly indexed")
            return False
    except Exception as e:
        logger.error(f"Error verifying document indexing: {str(e)}")
        return False

    # Step 8: Query vector database with test question
    try:
        test_question = "please show me my scones recipe"
        logger.info(f"Querying with: '{test_question}'")
        query_embedding = await embedding_model.generate_embedding(test_question)

        query_results = vector_db.query_vectors(
            query_vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        if len(query_results) == 0:
            logger.error("No results returned from query")
            return False
    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        return False

    # Step 9: Verify retrieved chunks contain expected content
    try:
        logger.info("Verifying retrieved chunks contain expected content")

        # Expected content fragments
        expected_content_fragments = [
            "all-purpose",
            "frozen grated butter",
            "cheddar",
            "sourdough discard",
            "250°C"
        ]

        # Debug the structure of first result - safely serialize
        if query_results:
            try:
                # Instead of trying to serialize the whole object, print the relevant parts
                first_result = query_results[0]
                result_info = {
                    "id": first_result.get("id", "unknown"),
                    "score": first_result.get("score", 0),
                    "metadata_keys": list(first_result.get("metadata", {}).keys())
                }
                logger.debug(f"First result structure: {json.dumps(result_info, indent=2)}")
            except Exception as json_err:
                # Fallback if serialization fails
                logger.debug(f"Could not serialize first result: {str(json_err)}")

        # Try to extract text from query results, handling different metadata structures
        all_content = ""
        for result in query_results:
            metadata = result.get("metadata", {})
            if isinstance(metadata, dict):
                # Try different possible keys where text might be stored
                text = metadata.get("text", metadata.get("content", metadata.get("chunk", "")))
                all_content += " " + text
            elif isinstance(metadata, str):
                all_content += " " + metadata

        # Log a sample of the retrieved content
        content_preview = all_content[:200] + "..." if len(all_content) > 200 else all_content
        logger.info(f"Retrieved content sample: {content_preview}")

        found_fragments = []
        missing_fragments = []

        for fragment in expected_content_fragments:
            if fragment.lower() in all_content.lower():
                found_fragments.append(fragment)
            else:
                missing_fragments.append(fragment)

        if missing_fragments:
            logger.warning(f"Missing expected fragments: {missing_fragments}")

        # Calculate success rate (at least 80% of expected fragments found)
        success_threshold = 0.8
        success_rate = len(found_fragments) / len(expected_content_fragments)

        if success_rate < success_threshold:
            logger.error(f"Only {success_rate*100:.1f}% of expected content fragments found")
            return False

        # Display results
        logger.info(f"RAG test successful! Found {len(found_fragments)}/{len(expected_content_fragments)} expected fragments")
        logger.info(f"Found fragments: {found_fragments}")

        # Show top result scores
        scores = [f"{result.get('score', 0):.4f}" for result in query_results[:3]]
        logger.info(f"Top result similarity scores: {scores}")

        # Display retrieved text snippets
        logger.info("Retrieved text snippets:")
        for i, result in enumerate(query_results[:3]):
            try:
                metadata = result.get("metadata", {})
                if isinstance(metadata, dict):
                    text = metadata.get("text", metadata.get("content", metadata.get("chunk", "No text found")))
                else:
                    text = str(metadata)
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"[{i+1}] Score: {result.get('score', 0):.4f} - {preview}")
            except Exception as snippet_err:
                logger.warning(f"Error displaying snippet {i+1}: {str(snippet_err)}")
    except Exception as e:
        logger.error(f"Error verifying results: {str(e)}")
        return False

    # Step 10: Clean up test data
    try:
        logger.info("Cleaning up test data")
        vector_db.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Error during cleanup (non-fatal): {str(e)}")

    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"RAG test completed in {execution_time:.2f} seconds")

    return True

def run_rag_test():
    """
    Synchronous wrapper for the async RAG test.
    """
    return asyncio.run(async_run_rag_test())

if __name__ == "__main__":
    try:
        success = run_rag_test()
        if success:
            logger.info("✅ RAG test passed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ RAG test failed!")
            sys.exit(1)
    except Exception as e:
        logger.exception("Error during RAG test execution")
        sys.exit(1)