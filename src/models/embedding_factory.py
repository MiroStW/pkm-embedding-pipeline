"""
Factory for creating embedding model instances.
"""
import logging
from typing import Dict, Any

from src.models.embedding_model import EmbeddingModel
from src.models.sentence_transformers_embedding import SentenceTransformersEmbedding

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

        # Only real embedding models are supported; error on unknown types
        if model_type not in ("e5", "distiluse"):
            logger.error(f"Unknown embedding model type: {model_type}; allowed types: 'e5', 'distiluse'")
            raise ValueError(f"Unknown embedding model type: {model_type}; allowed types: 'e5', 'distiluse'")

        # Create the requested model
        if model_type == "e5":
            return await EmbeddingModelFactory._create_e5_model(config)
        # distiluse fallback
        return await EmbeddingModelFactory._create_fallback_model(config)

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