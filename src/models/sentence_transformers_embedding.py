"""
Sentence Transformers embedding model implementation.
Used as a fallback when OpenAI API is unavailable.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from functools import lru_cache

from .embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class SentenceTransformersEmbedding(EmbeddingModel):
    """
    Sentence Transformers embedding model implementation.
    Uses locally running models for embedding generation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformers embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized Sentence Transformers embedding model: {model_name}")
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Install it with 'pip install sentence-transformers'")
        except Exception as e:
            raise ValueError(f"Failed to load Sentence Transformers model {model_name}: {str(e)}")

    async def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return (vec / norm).tolist()
        return embedding

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run CPU-intensive operations in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            # Run embedding generation in a separate thread to not block the async loop
            embeddings = await self._run_in_thread(self.model.encode, texts, convert_to_numpy=True)

            # Convert to list and normalize
            return [await self._normalize_embedding(embedding.tolist()) for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings with Sentence Transformers: {str(e)}")
            return [[] for _ in texts]

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text:
            return []

        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

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
            Combined embedding vector
        """
        title_embedding = await self.generate_embedding(title)
        content_embedding = await self.generate_embedding(content)

        if not title_embedding or not content_embedding:
            return content_embedding or title_embedding or []

        # Combine embeddings with weighting
        title_array = np.array(title_embedding)
        content_array = np.array(content_embedding)

        combined = (title_weight * title_array) + ((1 - title_weight) * content_array)

        # Normalize the combined embedding
        return await self._normalize_embedding(combined.tolist())

    async def batch_generate_embeddings(self,
                                 chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of document chunks and add embeddings.

        Args:
            chunks: List of document chunks with at least 'content' and 'title' keys

        Returns:
            List of document chunks with added 'embedding' key
        """
        processed_chunks = []

        # Generate title-enhanced embeddings for each chunk
        for chunk in chunks:
            title = chunk.get("title", "")
            content = chunk.get("content", "")

            if not content:
                logger.warning(f"Empty content for chunk with title: {title}")
                chunk["embedding"] = []
            else:
                # Generate title-enhanced embedding
                embedding = await self.generate_title_enhanced_embedding(
                    title=title,
                    content=content
                )
                chunk["embedding"] = embedding

            processed_chunks.append(chunk)

        return processed_chunks