# Pinecone Integration

## Overview

The Pinecone integration provides vector database capabilities for the embedding pipeline. It allows for uploading, querying, and managing vector embeddings in Pinecone, a specialized vector database service.

## Configuration

The Pinecone integration is configured in `config/config.yaml` under the `database.vector_db` section:

```yaml
database:
  vector_db:
    provider: "pinecone"
    api_key: "${PINECONE_API_KEY}"
    environment: "${PINECONE_ENVIRONMENT}"
    index_name: "${PINECONE_INDEX_NAME}"
    dimension: 1024
    max_retries: 3
    retry_delay: 2.0
    batch_size: 100
    serverless: false
    cloud_provider: "aws"
    region: "us-west-2"
```

### Configuration Options

- `provider`: Set to `"pinecone"` to use Pinecone
- `api_key`: Pinecone API key (can use environment variable)
- `environment`: Pinecone environment
- `index_name`: Name of the Pinecone index
- `dimension`: Vector dimension (must match embedding model dimension)
- `max_retries`: Maximum number of retry attempts for operations
- `retry_delay`: Delay between retry attempts (seconds)
- `batch_size`: Number of vectors to upload in a single batch
- `serverless`: Whether to use serverless deployment (default: false)
- `cloud_provider`: Cloud provider for serverless ("aws", "gcp", "azure")
- `region`: Region for serverless deployment

## Environment Variables

The Pinecone integration requires the following environment variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment
- `PINECONE_INDEX_NAME`: Name of your Pinecone index

Add these to your `.env` file:

```
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=pkm-embeddings
```

## Usage

### Verification

You can verify the Pinecone integration using the provided verification tool:

```bash
python -m src.database.pinecone_verify
```

This will run a series of tests to check connection, indexing, querying, synchronization, and deletion.

### CLI Tool

A command-line interface is provided for interacting with Pinecone:

```bash
# Show index information
python src/tools/pinecone_cli.py info

# Query the index
python src/tools/pinecone_cli.py query "your search query" --top-k 5

# Upload a test document
python src/tools/pinecone_cli.py upload --text "Test content"

# Delete a document
python src/tools/pinecone_cli.py delete --document-id doc_123

# Run verification tests
python src/tools/pinecone_cli.py verify
```

## Features

### Vector Upload

The integration handles uploading vectors to Pinecone with proper retry logic:

```python
from src.config import ConfigManager
from src.database.vector_db_factory import create_vector_db_uploader

# Get database client
config_manager = ConfigManager()
db_config = config_manager.get_database_config()
client = create_vector_db_uploader(db_config)

# Upload vectors
vectors = [
    {
        "id": "vec1",
        "values": [0.1, 0.2, 0.3, ...],  # Vector values
        "metadata": {"text": "Example text"}
    }
]
success_count, error_count = client.upload_vectors(vectors)
```

### Document Indexing

Documents are indexed with their associated chunks and embeddings:

```python
# Index a document with chunks and embeddings
success_count, error_count = client.upload_document_chunks(
    document_id="doc_123",
    chunks=document_chunks,
    embeddings=embeddings,
    title_enhanced_embeddings=title_embeddings  # Optional
)
```

### Querying

You can query vectors by similarity:

```python
# Query by vector
results = client.query_vectors(
    query_vector=embedding,
    top_k=5,
    filter={"metadata_field": "value"}  # Optional filter
)

# Query by text (requires embedding model)
from src.models.embedding import EmbeddingModelFactory

embedding_factory = EmbeddingModelFactory(**embedding_config)
embedding_model = embedding_factory.create_model()

results = client.query_text(
    text="your search query",
    embedding_model=embedding_model,
    top_k=5
)
```

### Document Management

The integration provides document management capabilities:

```python
# Delete a document
client.delete_document("doc_123")

# Synchronize a document (delete old vectors and upload new ones)
client.sync_document(
    document_id="doc_123",
    new_chunks=updated_chunks,
    embeddings=updated_embeddings
)

# Verify a document is indexed
is_indexed = client.verify_document_indexed("doc_123")

# Bulk delete vectors with filter
client.bulk_delete({"metadata_field": "value"})
```

## Error Handling

The integration includes comprehensive error handling with retry logic for all operations. Failed operations are logged and retried with exponential backoff to handle transient network issues.

## Testing

Unit tests for the Pinecone integration are in the `tests/test_pinecone_integration.py` file. Run them with:

```bash
pytest tests/test_pinecone_integration.py
```
