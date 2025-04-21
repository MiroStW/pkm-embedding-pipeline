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
            resolved_device = device # Keep original config value if provided
            if device is None or device == "auto":
                if torch.backends.mps.is_available():
                    resolved_device = "mps"  # Use M2 Max Neural Engine
                elif torch.cuda.is_available():
                    resolved_device = "cuda"
                else:
                    resolved_device = "cpu"
                logger.info(f"Auto-detected device: {resolved_device} (config was '{device}')")
            else:
                 logger.info(f"Using specified device: {resolved_device}")

            self.device = resolved_device # Store the resolved device name
            logger.info(f"Using device: {self.device}")

            # Initialize model
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device) # Use the resolved device string here

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

    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors, or None if an error occurred.
        """
        if not texts:
            return []

        try:
            # Prepare texts according to model requirements
            prepared_texts = [self._prepare_text(text) for text in texts]

            # Run embedding generation in a separate thread to not block the async loop
            logger.debug(f"Encoding {len(prepared_texts)} texts with {self.model_name} on {self.device}")
            embeddings = await self._run_in_thread(
                self.model.encode,
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # Built-in normalization
            )
            logger.debug(f"Successfully encoded {len(prepared_texts)} texts.")

            # Convert to list format and check validity
            result_embeddings = [emb.tolist() for emb in embeddings]
            if not result_embeddings or len(result_embeddings) != len(texts):
                 logger.error(f"Embedding result length mismatch or empty for model {self.model_name}")
                 return None # Indicate error

            return result_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with model {self.model_name}: {str(e)}")
            logger.exception("Full traceback for embedding generation error:") # Log full traceback
            return None # Indicate error by returning None

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to embed

        Returns:
            Embedding vector, or None if an error occurred.
        """
        if not text:
            return None # Return None for empty text

        embeddings = await self.generate_embeddings([text])
        # Return the first embedding if successful, otherwise None
        return embeddings[0] if embeddings and embeddings[0] else None

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