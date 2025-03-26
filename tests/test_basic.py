"""
Basic test script that doesn't require model downloads.
"""
import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import EmbeddingModel, OpenAIEmbedding, SentenceTransformersEmbedding, EmbeddingModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_module_structure():
    """Test that the module structure is correct."""
    # Check if EmbeddingModel is abstract
    try:
        model = EmbeddingModel()
        logger.error("EmbeddingModel should be abstract and not instantiable")
        return False
    except TypeError:
        logger.info("EmbeddingModel is correctly abstract")

    # Verify OpenAIEmbedding inherits from EmbeddingModel
    if not issubclass(OpenAIEmbedding, EmbeddingModel):
        logger.error("OpenAIEmbedding should inherit from EmbeddingModel")
        return False
    logger.info("OpenAIEmbedding inherits correctly from EmbeddingModel")

    # Verify SentenceTransformersEmbedding inherits from EmbeddingModel
    if not issubclass(SentenceTransformersEmbedding, EmbeddingModel):
        logger.error("SentenceTransformersEmbedding should inherit from EmbeddingModel")
        return False
    logger.info("SentenceTransformersEmbedding inherits correctly from EmbeddingModel")

    # Check factory class exists
    if not hasattr(EmbeddingModelFactory, 'create_model'):
        logger.error("EmbeddingModelFactory should have create_model method")
        return False
    logger.info("EmbeddingModelFactory has correct create_model method")

    return True

def main():
    """Run all tests."""
    logger.info("Running basic structure tests...")

    if test_module_structure():
        logger.info("All basic structure tests PASSED")
    else:
        logger.error("Basic structure tests FAILED")

if __name__ == "__main__":
    main()