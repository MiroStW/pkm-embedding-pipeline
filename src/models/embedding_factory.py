"""
Factory for creating embedding model instances.
"""
import os
import logging
from typing import Dict, Any, Optional

from src.models.embedding_model import EmbeddingModel
from src.models.sentence_transformers_embedding import SentenceTransformersEmbedding
from src.models.mock_embedding_model import MockEmbeddingModel

logger = logging.getLogger(__name__)

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models.
    Supports different model types and implements fallback mechanisms.
    """

    @staticmethod
    async def create_model(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create and return an embedding model based on configuration.

        Args:
            config: Configuration dictionary with embedding model settings

        Returns:
            An instance of EmbeddingModel
        """
        model_type = config.get("model_type", "e5").lower()

        # Check for mock model first
        if model_type == "mock":
            logger.info("Creating mock embedding model for benchmarking")
            dimension = config.get("dimension", 1024)
            return MockEmbeddingModel(dimension=dimension)

        # Try to create the requested model
        try:
            if model_type == "e5":
                return await EmbeddingModelFactory._create_e5_model(config)
            elif model_type == "distiluse":
                return await EmbeddingModelFactory._create_fallback_model(config)
            else:
                logger.warning(f"Unknown model type: {model_type}, falling back to E5")
                return await EmbeddingModelFactory._create_e5_model(config)
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {str(e)}")

            # Try to create fallback model
            try:
                logger.info("Attempting to create fallback model")
                return await EmbeddingModelFactory._create_fallback_model(config)
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback model: {str(fallback_error)}")

                # As a last resort, create mock model
                logger.warning("All model creation attempts failed, using mock model")
                dimension = config.get("dimension", 1024)
                return MockEmbeddingModel(dimension=dimension)

    @staticmethod
    async def _create_e5_model(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create and return an E5 embedding model based on the provided configuration.

        Args:
            config: Configuration dictionary with embedding model settings

        Returns:
            An instance of SentenceTransformersEmbedding
        """
        model_name = config.get("e5_model", "intfloat/multilingual-e5-large-instruct")
        device = config.get("device")  # Optional device override

        try:
            return SentenceTransformersEmbedding(
                model_name=model_name,
                is_fallback=False,
                device=device
            )
        except Exception as e:
            logger.error(f"Failed to initialize E5 model: {str(e)}")
            raise ValueError("Could not initialize E5 embedding model. Make sure dependencies are installed.")

    @staticmethod
    async def _create_fallback_model(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create and return a fallback embedding model.

        Args:
            config: Configuration dictionary with embedding model settings

        Returns:
            An instance of SentenceTransformersEmbedding
        """
        model_name = config.get(
            "distiluse_model",
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        device = config.get("device")

        try:
            return SentenceTransformersEmbedding(
                model_name=model_name,
                is_fallback=True,
                device=device
            )
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {str(e)}")
            raise ValueError("Could not initialize any embedding model. Make sure dependencies are installed.")