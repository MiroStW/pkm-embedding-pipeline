"""
Test script for the embedding models.
"""
import os
import sys
import asyncio
import logging
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import EmbeddingModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_e5_embedding():
    """Test the E5 embedding model."""
    logger.info("Testing E5 embedding model...")

    config = {
        "model_type": "e5",
        "e5_model": "intfloat/multilingual-e5-large-instruct",
        "device": "mps"  # For M2 Max optimization
    }

    try:
        model = await EmbeddingModelFactory.create_model(config)

        # Test single embedding
        text = "This is a test text for embedding generation."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with length: {len(embedding)}")
        assert len(embedding) > 0, "Embedding should not be empty"

        # Test title-enhanced embedding
        title = "Test Title"
        content = "This is the content of the test document."
        enhanced_embedding = await model.generate_title_enhanced_embedding(title, content)
        logger.info(f"Generated title-enhanced embedding with length: {len(enhanced_embedding)}")
        assert len(enhanced_embedding) > 0, "Enhanced embedding should not be empty"

        # Test batch embedding
        chunks = [
            {"title": "Document 1", "content": "Content of document 1"},
            {"title": "Document 2", "content": "Content of document 2"}
        ]
        processed_chunks = await model.batch_generate_embeddings(chunks)
        logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
        assert len(processed_chunks) == len(chunks), "All chunks should be processed"

    except Exception as e:
        logger.error(f"E5 embedding test failed: {str(e)}")
        pytest.fail(f"E5 embedding test failed: {str(e)}")

@pytest.mark.asyncio
async def test_distiluse_embedding():
    """Test the DistilUSE embedding model."""
    logger.info("Testing DistilUSE embedding model...")

    config = {
        "model_type": "distiluse",
        "device": "mps"  # For M2 Max optimization
    }

    try:
        model = await EmbeddingModelFactory.create_model(config)

        # Test single embedding
        text = "This is a test text for embedding generation."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with length: {len(embedding)}")
        assert len(embedding) > 0, "Embedding should not be empty"

        # Test title-enhanced embedding
        title = "Test Title"
        content = "This is the content of the test document."
        enhanced_embedding = await model.generate_title_enhanced_embedding(title, content)
        logger.info(f"Generated title-enhanced embedding with length: {len(enhanced_embedding)}")
        assert len(enhanced_embedding) > 0, "Enhanced embedding should not be empty"

        # Test batch embedding
        chunks = [
            {"title": "Document 1", "content": "Content of document 1"},
            {"title": "Document 2", "content": "Content of document 2"}
        ]
        processed_chunks = await model.batch_generate_embeddings(chunks)
        logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
        assert len(processed_chunks) == len(chunks), "All chunks should be processed"

    except Exception as e:
        logger.error(f"DistilUSE embedding test failed: {str(e)}")
        pytest.fail(f"DistilUSE embedding test failed: {str(e)}")

@pytest.mark.asyncio
async def test_fallback_mechanism():
    """Test the fallback mechanism from E5 to DistilUSE."""
    logger.info("Testing fallback mechanism from E5 to DistilUSE...")

    try:
        # Create a config with a non-existent E5 model to force fallback
        config = {
            "model_type": "e5",
            "e5_model": "intfloat/non-existent-model",  # Invalid model to force fallback
            "device": "mps"
        }

        model = await EmbeddingModelFactory.create_model(config)
        logger.info(f"Successfully created model with fallback: {model.__class__.__name__}")
        logger.info(f"Using model: {model.model_name}, is_fallback: {model.is_fallback}")
        assert model.is_fallback, "Model should be using fallback"

        # Test single embedding to confirm it works
        text = "Testing fallback mechanism."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with fallback model, length: {len(embedding)}")
        assert len(embedding) > 0, "Fallback embedding should not be empty"

    except Exception as e:
        logger.error(f"Fallback test failed: {str(e)}")
        pytest.fail(f"Fallback test failed: {str(e)}")

async def main():
    """Run all tests."""
    results = []

    # Test E5 embeddings
    e5_result = await test_e5_embedding()
    results.append(("E5 Embedding", e5_result))

    # Test DistilUSE embeddings
    distiluse_result = await test_distiluse_embedding()
    results.append(("DistilUSE Embedding", distiluse_result))

    # Test fallback mechanism
    fallback_result = await test_fallback_mechanism()
    results.append(("E5 to DistilUSE Fallback", fallback_result))

    # Print summary
    logger.info("=== Test Results ===")
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())