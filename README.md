# PKM Chatbot Embedding Pipeline

A pipeline for processing markdown files, generating embeddings, and synchronizing them with a vector database for use in a personal knowledge management (PKM) chatbot.

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Verify you're in the virtual environment (should show path to venv)
which python
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

## Document Tracking System

The pipeline uses a SQLite database to track the status of documents being processed. This helps maintain state across runs and enables incremental processing.

### Document States

Documents can be in one of these states:

- `pending`: Marked for processing but not yet processed
- `processing`: Currently being processed
- `completed`: Successfully processed and stored in the vector database
- `error`: Failed to process due to an error

### Managing Document States

You can check and manage document states using Python:

```python
from src.database.document_db import DocumentTracker

# Initialize the tracker
tracker = DocumentTracker()

# Get document statistics
stats = tracker.get_statistics()
print(f"Total documents: {stats['total']}")
print(f"Pending: {stats['pending']} ({stats['pending_percentage']}%)")
print(f"Completed: {stats['completed']} ({stats['completed_percentage']}%)")
print(f"Errors: {stats['errors']} ({stats['error_percentage']}%)")

# Get lists of documents by state
pending_docs = tracker.get_pending_documents()
completed_docs = tracker.get_completed_documents()
error_docs = tracker.get_error_documents()

# Mark a document for processing
tracker.mark_for_processing("path/to/document.md")

# Reset the status of processed documents
# You can do this directly with SQLite:
import sqlite3
conn = sqlite3.connect("data/document_tracker.db")
cursor = conn.cursor()
cursor.execute('DELETE FROM documents WHERE status = "completed"')
conn.commit()
conn.close()
```

The document tracking database is automatically created at `data/document_tracker.db` when needed.

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

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'yaml'**

   - This means you're missing required dependencies
   - Make sure you're in the virtual environment (`which python` should point to your venv)
   - Run `pip install -r requirements.txt` again

2. **sqlite3.OperationalError: unable to open database file**

   - The `data` directory might not exist
   - Create it manually: `mkdir -p data`
   - The database will be automatically initialized on next operation

3. **Document status not updating**
   - Check if you have write permissions in the `data` directory
   - Verify the database exists: `ls -l data/document_tracker.db`
   - You can reset document states using the DocumentTracker examples above

### Dependencies Overview

Key dependencies and their purposes:

- `pyyaml`: Configuration file handling
- `sqlalchemy`: Database ORM and management
- `pinecone-client`: Vector database integration
- `transformers`: For embedding generation
- `torch`: Required by transformers
- `python-dotenv`: Environment variable management

Make sure to install all dependencies in a virtual environment to avoid conflicts.

### Database Location

The SQLite database files are stored in the following locations:

- Document tracking: `data/document_tracker.db`
- Processing queue: `data/processing_queue.db` (created when needed)

You can safely delete these files to reset all document states, they will be recreated automatically.

## Implementation Status

This project is currently in development. See `instructions/` directory for implementation plan and details.
