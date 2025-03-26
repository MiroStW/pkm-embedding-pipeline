# Embedding Models

This document describes the embedding model components of the PKM chatbot embedding pipeline.

## Overview

The embedding model components are responsible for generating vector embeddings for document chunks. The system supports multiple embedding models and includes a fallback mechanism for reliability.

## Components

### `EmbeddingModel` (Abstract Base Class)

The base interface that all embedding models implement, with these key methods:

- `generate_embeddings(texts)`: Generate embeddings for multiple texts
- `generate_embedding(text)`: Generate embedding for a single text
- `generate_title_enhanced_embedding(title, content)`: Generate embedding that combines title and content
- `batch_generate_embeddings(chunks)`: Process a batch of document chunks

### `OpenAIEmbedding`

Primary embedding model that uses OpenAI's API:

- Uses `text-embedding-3-small` by default (configurable)
- Includes rate limiting to respect OpenAI's API limits
- Normalizes embeddings to unit length
- Handles errors gracefully
- Implements title-enhanced embeddings with configurable weighting

### `SentenceTransformersEmbedding`

Fallback embedding model that uses local sentence-transformers models:

- Uses `all-MiniLM-L6-v2` by default (configurable)
- Runs CPU-intensive operations in thread pool to avoid blocking async loop
- Normalizes embeddings to unit length
- Handles errors gracefully
- Implements title-enhanced embeddings with configurable weighting

### `EmbeddingModelFactory`

Factory class for creating embedding models:

- Creates the appropriate model based on configuration
- Implements fallback mechanism from OpenAI to sentence-transformers
- Handles initialization errors gracefully

## Usage

```python
from src.models import EmbeddingModelFactory

# Configuration
config = {
    "model_type": "openai",  # or "sentence-transformers"
    "openai_api_key": "your-api-key",  # optional, can use env var
    "openai_model": "text-embedding-3-small",  # optional
    "sentence_transformers_model": "all-MiniLM-L6-v2"  # optional
}

# Create embedding model
model = await EmbeddingModelFactory.create_model(config)

# Generate embeddings
embedding = await model.generate_embedding("Your text here")

# Process document chunks
chunks = [
    {"title": "Document 1", "content": "Content of document 1"},
    {"title": "Document 2", "content": "Content of document 2"}
]
processed_chunks = await model.batch_generate_embeddings(chunks)
```

## Title-Enhanced Embeddings

The embedding models support title-enhanced embeddings, which combine the embeddings of the title and content:

1. Generate embedding for title
2. Generate embedding for content
3. Combine with weighted average (default: 30% title, 70% content)
4. Normalize the combined embedding

This approach improves retrieval quality by incorporating the semantic meaning of both the title and content.
