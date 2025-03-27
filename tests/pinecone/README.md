# Pinecone Integration Tests

This directory contains test files for the Pinecone vector database integration.

## Available Tests

- **test_delete_document.py**: Tests the document deletion functionality
- **test_direct_connection.py**: Tests direct connection to Pinecone using the official SDK
- **test_pinecone_client.py**: Tests the custom PineconeClient implementation
- **test_real_connection.py**: Tests the connection to a real Pinecone index with API key
- **test_pytest_pinecone.py**: Pytest-compatible version of tests for use with pytest framework

## Running Tests

### Using Direct Python Execution

To run a specific test directly:

```bash
python tests/pinecone/test_pinecone_client.py
```

### Using Pytest

To run the pytest-compatible tests:

```bash
# Run all pytest-compatible tests
python -m pytest tests/pinecone/test_pytest_pinecone.py

# Run with verbose output
python -m pytest tests/pinecone/test_pytest_pinecone.py -v

# Run a specific test
python -m pytest tests/pinecone/test_pytest_pinecone.py::test_pinecone_client -v
```

## Environment Variables

These tests require the following environment variables to be set in a `.env` file:

```
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_ENVIRONMENT=gcp-starter  # For the free tier
```
