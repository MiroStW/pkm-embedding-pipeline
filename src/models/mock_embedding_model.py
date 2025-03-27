"""
Mock embedding model for benchmarking.
"""
import logging
import random
from typing import List, Dict, Any, Optional

from src.models.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class MockEmbeddingModel(EmbeddingModel):
    """
    Mock implementation of an embedding model for benchmarking.
    Generates random embeddings of the specified dimension.
    """

    def __init__(self, dimension: int = 1024):
        """
        Initialize the mock embedding model.

        Args:
            dimension: The dimension of the mock embeddings to generate
        """
        self.dimension = dimension
        logger.info(f"Initialized MockEmbeddingModel with dimension {dimension}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        return [self._generate_random_embedding() for _ in texts]

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding for a single text chunk.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self._generate_random_embedding()

    async def generate_title_enhanced_embedding(self,
                                         title: str,
                                         content: str,
                                         title_weight: float = 0.3) -> List[float]:
        """
        Generate mock embedding that simulates a title-enhanced embedding.

        Args:
            title: Document title
            content: Document content/chunk
            title_weight: Weight to apply to title embedding (0-1)

        Returns:
            Combined embedding vector as list of floats
        """
        return self._generate_random_embedding()

    async def batch_generate_embeddings(self,
                                 chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of document chunks and add mock embeddings.

        Args:
            chunks: List of document chunks with at least 'content' and 'title' keys

        Returns:
            List of document chunks with added 'embedding' key
        """
        for chunk in chunks:
            chunk["embedding"] = self._generate_random_embedding()

        return chunks

    def _generate_random_embedding(self) -> List[float]:
        """
        Generate a random embedding vector.

        Returns:
            A random embedding vector with the specified dimension
        """
        return [random.random() for _ in range(self.dimension)]