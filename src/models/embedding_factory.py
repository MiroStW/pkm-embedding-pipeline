"""
Factory class for creating embedding models.
"""
import os
import logging
from typing import Dict, Any, Optional

from .embedding_model import EmbeddingModel
from .sentence_transformers_embedding import SentenceTransformersEmbedding

logger = logging.getLogger(__name__)

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models based on configuration.
    """

    @staticmethod
    async def create_model(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create and return an embedding model based on the provided configuration.

        Args:
            config: Configuration dictionary with embedding model settings

        Returns:
            An instance of EmbeddingModel
        """
        model_type = config.get("model_type", "e5").lower()
        device = config.get("device")  # Optional device override

        if model_type == "e5":
            # Try to create E5 model
            model_name = config.get("e5_model", "intfloat/multilingual-e5-large-instruct")

            try:
                return SentenceTransformersEmbedding(
                    model_name=model_name,
                    is_fallback=False,
                    device=device
                )
            except Exception as e:
                logger.warning(f"Failed to initialize E5 model: {str(e)}")
                logger.warning("Falling back to distiluse model")
                # Fall back to distiluse
                return await EmbeddingModelFactory._create_fallback_model(config)

        elif model_type == "distiluse":
            return await EmbeddingModelFactory._create_fallback_model(config)

        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")

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