# Pinecone Integration Scripts

This directory contains scripts for testing and verifying the Pinecone vector database integration.

## Available Scripts

- **run_verification.py**: Main script that runs all verification steps in sequence
- **simple_connection_test.py**: Tests basic connection to Pinecone API
- **verify_integration.py**: Comprehensive verification suite for the Pinecone integration
- **simple_verification.py**: Simplified verification for quick testing
- **cleanup_vectors.py**: Utility script to clean up test vectors from Pinecone

## Usage

To run the full verification suite:

```bash
python scripts/pinecone/run_verification.py
```

To run individual scripts:

```bash
python scripts/pinecone/simple_connection_test.py
python scripts/pinecone/verify_integration.py
python scripts/pinecone/cleanup_vectors.py
```

## Environment Variables

These scripts require the following environment variables to be set in a `.env` file:

```
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_ENVIRONMENT=gcp-starter  # For the free tier
```

## Test Vectors

All test vectors created by these scripts will have IDs that start with `test_` or `verify_`.
The cleanup script will automatically remove any vectors with IDs matching these patterns.
