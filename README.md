# PKM Chatbot Embedding Pipeline

A pipeline for processing markdown files, generating embeddings, and synchronizing them with a vector database for use in a personal knowledge management (PKM) chatbot.

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up configuration:

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml to match your environment
```

## Usage

### Basic Usage

```bash
python -m src.main
```

### Pinecone Integration

The system supports Pinecone as a vector database. To test and verify your Pinecone integration:

1. Set up environment variables (create a `.env` file in the root directory):

```
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_ENVIRONMENT=gcp-starter  # For free tier
```

2. Run the verification script:

```bash
python scripts/pinecone/run_verification.py
```

For more detailed information about Pinecone integration:

- See [scripts/pinecone/README.md](scripts/pinecone/README.md) for available scripts
- See [tests/pinecone/README.md](tests/pinecone/README.md) for tests

### Installing Git Hooks

The embedding pipeline can be integrated with Git to automatically process changes when commits are made or files are merged:

```bash
python -m src.git_hooks.cli install
```

This will install the following Git hooks:

- `post-commit`: Triggered after each commit to identify changed files
- `post-merge`: Triggered after merges or pulls to identify changes from remote

### Git Hooks Status

Check the status of installed Git hooks:

```bash
python -m src.git_hooks.cli status
```

### List All Markdown Files

List all markdown files tracked by Git:

```bash
python -m src.git_hooks.cli list-files
```

## Components

1. **Document Processor**: Parses markdown files and extracts chunks and metadata
2. **Embedding Model**: Generates vector embeddings from text chunks
3. **Git Integration**: Tracks file changes using Git hooks
4. **Database**: Tracks document processing status and manages work queue
5. **Pinecone Integration**: Synchronizes vectors with the Pinecone vector database

## Development

### Running Tests

```bash
python -m unittest discover tests
```

### Testing Pinecone Integration

```bash
# Run the pytest-compatible tests
python -m pytest tests/pinecone/test_pytest_pinecone.py -v

# Run standalone test scripts
python tests/pinecone/test_pinecone_client.py
python tests/pinecone/test_direct_connection.py
```

### Testing Git Hooks Specifically

```bash
python -m unittest tests.test_git_hooks
```

For manual testing of Git hooks, use the shell script:

```bash
# Run from the project root directory
tests/scripts/test_git_hooks.sh
```

## Project Structure

```
embedding-pipeline/
├── config/              # Configuration files
├── data/                # Data storage (database, logs)
├── docs/                # Documentation
├── logs/                # Log files
├── scripts/             # Utility scripts
│   └── pinecone/        # Pinecone-related scripts
├── src/                 # Source code
│   ├── database/        # Database-related code
│   ├── models/          # Embedding models
│   ├── processors/      # Document processing logic
│   └── main.py          # Main entry point
├── tests/               # Test code
│   └── pinecone/        # Pinecone integration tests
├── venv/                # Virtual environment
└── README.md            # This file
```

## Features

- Processes markdown files with frontmatter metadata
- Implements semantic chunking
- Generates embeddings using multilingual transformer models
- Tracks document states in SQLite database
- Supports bulk and incremental processing modes
- Integrates with Pinecone vector database

## Implementation Status

This project is currently in development. See `instructions/` directory for implementation plan and details.
