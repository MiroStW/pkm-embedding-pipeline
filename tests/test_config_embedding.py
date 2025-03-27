"""
Test script for configuration-based embedding model creation.
"""
import os
import sys
import asyncio
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager
from src.models import EmbeddingModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_config_based_embedding():
    """Test creating and using embedding models from configuration."""
    logger.info("Testing configuration-based embedding model creation...")

    try:
        # Load configuration
        config_manager = ConfigManager()
        embedding_config = config_manager.get_embedding_config()

        logger.info(f"Loaded embedding configuration: {embedding_config}")

        # Create model using configuration
        model = await EmbeddingModelFactory.create_model(embedding_config)

        logger.info(f"Created model: {model.__class__.__name__}")
        if hasattr(model, 'model_name'):
            logger.info(f"Model name: {model.model_name}")
        if hasattr(model, 'device'):
            logger.info(f"Device: {model.device}")
        if hasattr(model, 'is_fallback'):
            logger.info(f"Is fallback: {model.is_fallback}")

        # Test a simple embedding
        test_text = "This is a test of the configuration-based embedding system."
        embedding = await model.generate_embedding(test_text)

        logger.info(f"Generated embedding with length: {len(embedding)}")

        # Test title-enhanced embedding if enabled
        if embedding_config.get("enable_title_enhanced", True):
            title = "Configuration Test"
            content = "Testing the configuration-based embedding generation with title enhancement."
            title_weight = embedding_config.get("title_weight", 0.3)

            enhanced_embedding = await model.generate_title_enhanced_embedding(
                title=title,
                content=content,
                title_weight=title_weight
            )

            logger.info(f"Generated title-enhanced embedding with length: {len(enhanced_embedding)}")

        return True
    except Exception as e:
        logger.error(f"Configuration-based embedding test failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    results = []

    # Test configuration-based embedding
    results.append(("Configuration-based Embedding", await test_config_based_embedding()))

    # Print summary
    logger.info("=== Test Results ===")
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())