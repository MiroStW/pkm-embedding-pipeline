# Embedding Models

This document describes the embedding model components of the PKM chatbot embedding pipeline.

## Overview

The embedding model components are responsible for generating vector embeddings for document chunks. The system uses state-of-the-art multilingual models with a fallback mechanism for reliability.

## Components

### `EmbeddingModel` (Abstract Base Class)

The base interface that all embedding models implement, with these key methods:

- `generate_embeddings(texts)`: Generate embeddings for multiple texts
- `generate_embedding(text)`: Generate embedding for a single text
- `generate_title_enhanced_embedding(title, content)`: Generate embedding that combines title and content
- `batch_generate_embeddings(chunks)`: Process a batch of document chunks

### `SentenceTransformersEmbedding`

Flexible embedding model implementation that supports both primary and fallback models:

#### Primary Model (E5)

- Uses `intfloat/multilingual-e5-large-instruct` by default
- Optimized for high-quality multilingual embeddings
- Configured for M2 Max Neural Engine through PyTorch MPS
- Handles instruction-tuned model requirements
- Implements title-enhanced embeddings with configurable weighting

#### Fallback Model (DistilUSE)

- Uses `sentence-transformers/distiluse-base-multilingual-cased-v2`
- Provides efficient processing with lower resource requirements
- Maintains multilingual support
- Shares same interface and features as primary model

### `EmbeddingModelFactory`

Factory class for creating embedding models:

- Creates the appropriate model based on configuration
- Implements fallback mechanism from E5 to DistilUSE
- Handles hardware optimization and device selection
- Manages initialization errors gracefully

## Usage

```python
from src.models import EmbeddingModelFactory

# Configuration
config = {
    "model_type": "e5",  # or "distiluse"
    "e5_model": "intfloat/multilingual-e5-large-instruct",  # optional
    "distiluse_model": "sentence-transformers/distiluse-base-multilingual-cased-v2",  # optional
    "device": "mps"  # optional, auto-detected if not specified
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

## Hardware Optimization

The implementation automatically detects and utilizes available hardware:

1. M2 Max Neural Engine (MPS) - Primary choice for Apple Silicon
2. CUDA - Used when available on systems with NVIDIA GPUs
3. CPU - Fallback for other systems

The device can be manually specified in the configuration if needed.
