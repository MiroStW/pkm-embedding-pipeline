"""
Embedding model interface and implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import numpy as np

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    All embedding model implementations should inherit from this class.
    """

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors (as lists of floats)
        """
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def generate_title_enhanced_embedding(self,
                                         title: str,
                                         content: str,
                                         title_weight: float = 0.3) -> List[float]:
        """
        Generate embedding that combines title and content with specified weighting.

        Args:
            title: Document title
            content: Document content/chunk
            title_weight: Weight to apply to title embedding (0-1)

        Returns:
            Combined embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def batch_generate_embeddings(self,
                                 chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of document chunks and add embeddings.

        Args:
            chunks: List of document chunks with at least 'content' and 'title' keys

        Returns:
            List of document chunks with added 'embedding' key
        """
        pass