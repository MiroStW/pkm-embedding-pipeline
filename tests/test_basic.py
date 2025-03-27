"""
Basic test script that doesn't require model downloads.
"""
import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import EmbeddingModel, SentenceTransformersEmbedding, EmbeddingModelFactory

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
        assert False, "EmbeddingModel should be abstract and not instantiable"
    except TypeError:
        logger.info("EmbeddingModel is correctly abstract")

    # Verify SentenceTransformersEmbedding inherits from EmbeddingModel
    assert issubclass(SentenceTransformersEmbedding, EmbeddingModel), "SentenceTransformersEmbedding should inherit from EmbeddingModel"
    logger.info("SentenceTransformersEmbedding inherits correctly from EmbeddingModel")

    # Check factory class exists
    assert hasattr(EmbeddingModelFactory, 'create_model'), "EmbeddingModelFactory should have create_model method"
    logger.info("EmbeddingModelFactory has correct create_model method")

def main():
    """Run all tests."""
    logger.info("Running basic structure tests...")

    if test_module_structure():
        logger.info("All basic structure tests PASSED")
    else:
        logger.error("Basic structure tests FAILED")

if __name__ == "__main__":
    main()