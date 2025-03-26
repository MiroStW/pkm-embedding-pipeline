"""
OpenAI embedding model implementation.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from openai import AsyncOpenAI

from .embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

class OpenAIEmbedding(EmbeddingModel):
    """
    OpenAI embedding model implementation.
    Uses the OpenAI API to generate embeddings.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedding model.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: OpenAI embedding model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")

        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI embedding model using {model}")

        # Rate limiting settings
        self.max_batch_size = 100  # Max number of texts to embed in a single API call
        self.rate_limit_requests = 20  # Max requests per minute
        self.request_interval = 60 / self.rate_limit_requests  # Time between requests in seconds
        self.last_request_time = 0

    async def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return (vec / norm).tolist()
        return embedding

    async def _wait_for_rate_limit(self):
        """Sleep if necessary to respect rate limits."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using OpenAI API.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches to respect OpenAI's rate limits
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i+self.max_batch_size]

            # Respect rate limits
            await self._wait_for_rate_limit()

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                # Normalize embeddings
                batch_embeddings = [await self._normalize_embedding(emb) for emb in batch_embeddings]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
                # Return empty embeddings on error
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
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