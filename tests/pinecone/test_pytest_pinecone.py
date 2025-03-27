#!/usr/bin/env python3
"""
Pytest-compatible tests for Pinecone integration
"""
import os
import sys
import pytest
import random
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Load environment variables
load_dotenv()

# Skip tests if no API key is available
api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')

pytestmark = [
    pytest.mark.skipif(
        not api_key or not index_name,
        reason="Pinecone API key or index name not set"
    ),
    pytest.mark.pinecone
]

def generate_random_embedding(dimension=1536):
    """Generate a random normalized embedding vector"""
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    # Normalize the vector
    magnitude = sum(x**2 for x in vector) ** 0.5
    return [x/magnitude for x in vector]

@pytest.mark.pinecone
def test_direct_connection():
    """Test direct connection to Pinecone API"""
    from pinecone import Pinecone

    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)

    # List indexes
    indexes = pc.list_indexes().names()
    assert len(indexes) > 0, "No indexes found in Pinecone account"

    # Check if our index exists
    assert index_name in indexes, f"Index '{index_name}' not found"

    # Connect to index
    index = pc.Index(index_name)

    # Get stats
    stats = index.describe_index_stats()
    assert "dimension" in stats, "Failed to get index stats"
    assert "total_vector_count" in stats, "Failed to get vector count"

@pytest.mark.pinecone
def test_pinecone_client():
    """Test PineconeClient functionality"""
    from src.database.pinecone_client import PineconeClient

    # Initialize client
    client = PineconeClient(
        api_key=api_key,
        environment=os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter'),
        index_name=index_name,
        dimension=1536
    )

    # Check connection
    assert client.check_connection(), "Failed to connect to Pinecone"

    # Get stats
    stats = client.get_stats()
    assert stats is not None, "Failed to get index stats"

    # Test document operations
    document_id = f"pytest_{random.randint(10000000, 99999999)}"

    try:
        # Create document chunks and embeddings
        chunks = [
            {
                "content": "This is a pytest test document for the Pinecone integration.",
                "metadata": {
                    "section_title": "Test Document",
                    "section_level": 1
                }
            }
        ]
        embeddings = [generate_random_embedding(1536)]

        # Upload document
        success_count, error_count = client.upload_document_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )
        assert success_count > 0, "Failed to upload document chunks"
        assert error_count == 0, "Errors occurred during upload"

        # Query for the document
        results = client.query_vectors(
            query_vector=embeddings[0],
            top_k=1
        )
        assert results is not None, "No results returned from query"
        assert len(results) > 0, "No matches found in query results"

        # Verify the document is found
        assert results[0]['id'].startswith(f"{document_id}_"), "Query did not return the uploaded document"

    finally:
        # Clean up - delete the document
        delete_success = client.delete_document(document_id)
        assert delete_success, "Failed to delete test document"

@pytest.mark.pinecone
def test_vector_db_uploader():
    """Test VectorDatabaseUploader functionality"""
    from src.database.vector_db import VectorDatabaseUploader

    # Initialize uploader
    uploader = VectorDatabaseUploader(
        api_key=api_key,
        environment=os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter'),
        index_name=index_name,
        dimension=1536
    )

    # Generate test document ID
    test_doc_id = f"pytest_{random.randint(10000000, 99999999)}"

    try:
        # Create test vectors
        vectors = []
        for i in range(3):
            vector_id = f"{test_doc_id}_{i}"
            vector = {
                "id": vector_id,
                "values": generate_random_embedding(1536),
                "metadata": {
                    "document_id": test_doc_id,
                    "chunk_index": i,
                    "text": f"This is pytest test chunk {i}",
                    "title": "Test Document"
                }
            }
            vectors.append(vector)

        # Upload vectors
        success_count, error_count = uploader.upload_vectors(vectors)
        assert success_count == 3, f"Expected to upload 3 vectors, but uploaded {success_count}"
        assert error_count == 0, f"Expected 0 errors, but got {error_count}"

        # Test document deletion
        delete_success = uploader.delete_document(test_doc_id)
        assert delete_success, "Failed to delete test document"

    finally:
        # Ensure cleanup
        uploader.delete_document(test_doc_id)