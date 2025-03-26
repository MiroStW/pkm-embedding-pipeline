"""
Test script for the embedding models.
"""
import os
import sys
import asyncio
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import EmbeddingModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_openai_embedding():
    """Test the OpenAI embedding model."""
    logger.info("Testing OpenAI embedding model...")

    config = {
        "model_type": "openai",
        "openai_model": "text-embedding-3-small"
    }

    try:
        model = await EmbeddingModelFactory.create_model(config)

        # Test single embedding
        text = "This is a test text for embedding generation."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with length: {len(embedding)}")

        # Test title-enhanced embedding
        title = "Test Title"
        content = "This is the content of the test document."
        enhanced_embedding = await model.generate_title_enhanced_embedding(title, content)
        logger.info(f"Generated title-enhanced embedding with length: {len(enhanced_embedding)}")

        # Test batch embedding
        chunks = [
            {"title": "Document 1", "content": "Content of document 1"},
            {"title": "Document 2", "content": "Content of document 2"}
        ]
        processed_chunks = await model.batch_generate_embeddings(chunks)
        logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")

        return True
    except Exception as e:
        logger.error(f"OpenAI embedding test failed: {str(e)}")
        return False

async def test_sentence_transformers_embedding():
    """Test the Sentence Transformers embedding model."""
    logger.info("Testing Sentence Transformers embedding model...")

    config = {
        "model_type": "sentence-transformers",
        "sentence_transformers_model": "all-MiniLM-L6-v2"
    }

    try:
        model = await EmbeddingModelFactory.create_model(config)

        # Test single embedding
        text = "This is a test text for embedding generation."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with length: {len(embedding)}")

        # Test title-enhanced embedding
        title = "Test Title"
        content = "This is the content of the test document."
        enhanced_embedding = await model.generate_title_enhanced_embedding(title, content)
        logger.info(f"Generated title-enhanced embedding with length: {len(enhanced_embedding)}")

        # Test batch embedding
        chunks = [
            {"title": "Document 1", "content": "Content of document 1"},
            {"title": "Document 2", "content": "Content of document 2"}
        ]
        processed_chunks = await model.batch_generate_embeddings(chunks)
        logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")

        return True
    except Exception as e:
        logger.error(f"Sentence Transformers embedding test failed: {str(e)}")
        return False

async def test_fallback_mechanism():
    """Test the fallback mechanism from OpenAI to Sentence Transformers."""
    logger.info("Testing fallback mechanism...")

    # Temporarily save the API key
    original_api_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Set invalid API key to force fallback
        os.environ["OPENAI_API_KEY"] = "invalid_key"

        config = {
            "model_type": "openai",  # Should fallback to sentence-transformers
        }

        model = await EmbeddingModelFactory.create_model(config)
        logger.info(f"Successfully created model with fallback: {model.__class__.__name__}")

        # Test single embedding to confirm it works
        text = "Testing fallback mechanism."
        embedding = await model.generate_embedding(text)
        logger.info(f"Generated embedding with fallback model, length: {len(embedding)}")

        return True
    except Exception as e:
        logger.error(f"Fallback test failed: {str(e)}")
        return False
    finally:
        # Restore original API key
        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

async def main():
    """Run all tests."""
    results = []

    # Test OpenAI embeddings if API key is set
    if os.environ.get("OPENAI_API_KEY"):
        results.append(("OpenAI Embedding", await test_openai_embedding()))
    else:
        logger.warning("Skipping OpenAI embedding test: No API key set")

    # Test Sentence Transformers embeddings
    results.append(("Sentence Transformers Embedding", await test_sentence_transformers_embedding()))

    # Test fallback mechanism if API key is set
    if os.environ.get("OPENAI_API_KEY"):
        results.append(("Fallback Mechanism", await test_fallback_mechanism()))
    else:
        logger.warning("Skipping fallback test: No API key set")

    # Print summary
    logger.info("=== Test Results ===")
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())