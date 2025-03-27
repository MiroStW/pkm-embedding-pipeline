"""
Sentence Transformers embedding model implementation.
Primary and fallback models for the embedding pipeline.
"""
import os
import asyncio
import torch
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from functools import lru_cache

from .embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class SentenceTransformersEmbedding(EmbeddingModel):
    """
    Sentence Transformers embedding model implementation.
    Supports both primary (E5) and fallback (DistilUSE) models.
    """

    def __init__(self,
                 model_name: str = "intfloat/multilingual-e5-large-instruct",
                 is_fallback: bool = False,
                 device: Optional[str] = None):
        """
        Initialize the Sentence Transformers embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
            is_fallback: Whether this is the fallback model
            device: Device to use for computation (auto-detected if None)
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Set default fallback model if needed
            if is_fallback:
                model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

            self.model_name = model_name
            self.is_fallback = is_fallback

            # Configure device
            if device is None:
                if torch.backends.mps.is_available():
                    device = "mps"  # Use M2 Max Neural Engine
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self.device = device
            logger.info(f"Using device: {device}")

            # Initialize model
            self.model = SentenceTransformer(model_name)
            self.model.to(device)

            # Configure for E5 model specifics
            self.is_e5_model = "e5" in model_name.lower()

            logger.info(f"Initialized {'fallback' if is_fallback else 'primary'} model: {model_name}")

        except ImportError:
            raise ImportError("sentence-transformers is not installed. Install it with 'pip install sentence-transformers'")
        except Exception as e:
            raise ValueError(f"Failed to load Sentence Transformers model {model_name}: {str(e)}")

    def _prepare_text(self, text: str) -> str:
        """Prepare text based on model requirements."""
        if self.is_e5_model:
            # E5 models expect "query: " or "passage: " prefix
            return f"passage: {text}"
        return text

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
            # Prepare texts according to model requirements
            prepared_texts = [self._prepare_text(text) for text in texts]

            # Run embedding generation in a separate thread to not block the async loop
            embeddings = await self._run_in_thread(
                self.model.encode,
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # Built-in normalization
            )

            # Convert to list format
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"Error generating embeddings with model {self.model_name}: {str(e)}")
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