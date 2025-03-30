"""
Test the end-to-end RAG process with the scones.md file.

This test verifies that:
1. A markdown file can be processed through the embedding pipeline
2. The embeddings are correctly stored in Pinecone
3. A query can retrieve the relevant chunks from Pinecone
4. The retrieved chunks contain the expected information
"""
import os
import pytest
import logging
from typing import Dict, List, Any

from src.config import Config
from src.processors.document_processor import DocumentProcessor
from src.models.embedding_factory import EmbeddingModelFactory
from src.database.pinecone_client import PineconeClient
from src.database.vector_db_factory import VectorDBFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRagProcess:
    """Test the end-to-end RAG process with the scones.md file."""

    @pytest.fixture
    def config(self):
        """Load the configuration from the config file."""
        config_path = os.path.join("config", "test_config.yaml")
        return Config.load_config(config_path)

    @pytest.fixture
    def document_processor(self, config):
        """Create a document processor instance."""
        return DocumentProcessor(config)

    @pytest.fixture
    def embedding_model(self, config):
        """Create an embedding model instance."""
        factory = EmbeddingModelFactory(config)
        return factory.create_model()

    @pytest.fixture
    def vector_db(self, config):
        """Create a vector database client."""
        factory = VectorDBFactory(config)
        client = factory.create_client()

        # Use a test namespace to avoid affecting production data
        client.namespace = "test_namespace"

        return client

    @pytest.fixture
    def test_file_path(self):
        """Path to the test markdown file."""
        return "scones.md"

    @pytest.fixture
    def test_question(self):
        """Test question for RAG query."""
        return "please show me my scones recipe"

    @pytest.fixture
    def expected_content_fragments(self):
        """Expected content fragments that should be in the RAG response."""
        return [
            "all-purpose",
            "frozen grated butter",
            "cheddar",
            "sourdough discard",
            "250°C"
        ]

    def test_end_to_end_rag_process(self, config, document_processor, embedding_model,
                                  vector_db, test_file_path, test_question,
                                  expected_content_fragments):
        """Test the full end-to-end RAG process."""
        try:
            # STEP 1: Process the test document
            logger.info(f"Processing document: {test_file_path}")
            processed_doc = document_processor.process_file(test_file_path)

            assert processed_doc["status"] == "success", f"Failed to process document: {processed_doc.get('reason', 'Unknown error')}"
            assert len(processed_doc["chunks"]) > 0, "No chunks were generated from the document"

            document_id = processed_doc["metadata"]["id"]
            logger.info(f"Document processed with ID: {document_id}, generated {len(processed_doc['chunks'])} chunks")

            # STEP 2: Generate embeddings for the chunks
            logger.info("Generating embeddings for document chunks")
            chunks = processed_doc["chunks"]
            embeddings = []

            # Generate embeddings for each chunk
            for chunk in chunks:
                chunk_text = chunk["text"]
                embedding = embedding_model.encode(chunk_text)
                embeddings.append(embedding)

            # Generate title-enhanced embeddings if supported
            title = processed_doc["metadata"].get("title", "")
            title_enhanced_embeddings = None

            if hasattr(embedding_model, 'encode_with_title'):
                title_enhanced_embeddings = []
                for chunk in chunks:
                    chunk_text = chunk["text"]
                    enhanced_embedding = embedding_model.encode_with_title(title, chunk_text)
                    title_enhanced_embeddings.append(enhanced_embedding)

            # STEP 3: Upload embeddings to Pinecone
            logger.info(f"Uploading embeddings to Pinecone (namespace: {vector_db.namespace})")

            # Delete any existing data for this document in the test namespace
            vector_db.delete_document(document_id)

            # Upload the new embeddings
            upload_success = vector_db.sync_document(
                document_id=document_id,
                new_chunks=chunks,
                embeddings=embeddings,
                title_enhanced_embeddings=title_enhanced_embeddings
            )

            assert upload_success, "Failed to upload embeddings to Pinecone"

            # STEP 4: Verify document was added to Pinecone
            logger.info("Verifying document was added to Pinecone")
            is_indexed = vector_db.verify_document_indexed(document_id)
            assert is_indexed, "Document was not properly indexed in Pinecone"

            # STEP 5: Query Pinecone with test question
            logger.info(f"Querying Pinecone with: '{test_question}'")
            query_embedding = embedding_model.encode(test_question)

            query_results = vector_db.query_vectors(
                query_vector=query_embedding,
                top_k=3,
                include_metadata=True
            )

            assert len(query_results) > 0, "No results returned from Pinecone query"

            # STEP 6: Verify retrieved chunks contain expected content
            logger.info("Verifying retrieved chunks contain expected content")
            all_content = " ".join([result["metadata"]["text"] for result in query_results])

            found_fragments = []
            missing_fragments = []

            for fragment in expected_content_fragments:
                if fragment.lower() in all_content.lower():
                    found_fragments.append(fragment)
                else:
                    missing_fragments.append(fragment)

            if missing_fragments:
                logger.warning(f"Missing expected fragments: {missing_fragments}")

            # Check that at least 80% of expected fragments are found
            success_threshold = 0.8
            success_rate = len(found_fragments) / len(expected_content_fragments)

            assert success_rate >= success_threshold, f"Only {success_rate*100:.1f}% of expected content fragments found in results"

            logger.info(f"RAG test successful! Found {len(found_fragments)}/{len(expected_content_fragments)} expected fragments")

            # Show matching fragments for debugging
            logger.info(f"Found fragments: {found_fragments}")

            # Show top result scores
            scores = [f"{result.get('score', 0):.4f}" for result in query_results[:3]]
            logger.info(f"Top result similarity scores: {scores}")

        finally:
            # Clean up - remove test data
            logger.info("Cleaning up test data")
            vector_db.delete_document(document_id)