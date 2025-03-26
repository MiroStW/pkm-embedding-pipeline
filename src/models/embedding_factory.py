"""
Factory class for creating embedding models.
"""
import os
import logging
from typing import Dict, Any, Optional

from .embedding_model import EmbeddingModel
from .openai_embedding import OpenAIEmbedding
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
        model_type = config.get("model_type", "openai").lower()

        if model_type == "openai":
            # Try to create OpenAI model
            api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            model_name = config.get("openai_model", "text-embedding-3-small")

            try:
                return OpenAIEmbedding(api_key=api_key, model=model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embedding model: {str(e)}")
                logger.warning("Falling back to sentence-transformers model")
                # Fall back to sentence-transformers
                return await EmbeddingModelFactory._create_sentence_transformers_model(config)

        elif model_type == "sentence-transformers":
            return await EmbeddingModelFactory._create_sentence_transformers_model(config)

        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")

    @staticmethod
    async def _create_sentence_transformers_model(config: Dict[str, Any]) -> EmbeddingModel:
        """
        Create and return a sentence-transformers embedding model.

        Args:
            config: Configuration dictionary with embedding model settings

        Returns:
            An instance of SentenceTransformersEmbedding
        """
        model_name = config.get("sentence_transformers_model", "all-MiniLM-L6-v2")

        try:
            return SentenceTransformersEmbedding(model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers model: {str(e)}")
            raise ValueError("Could not initialize any embedding model. Make sure dependencies are installed.")