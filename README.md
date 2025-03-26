# PKM Chatbot Embedding Pipeline

This project implements an embedding pipeline for a Personal Knowledge Management (PKM) chatbot. It processes markdown files, generates vector embeddings, and synchronizes them with a cloud vector database (Pinecone).

## Project Structure

```
embedding-pipeline/
├── config/              # Configuration files
├── data/                # Data storage (database, logs)
├── src/                 # Source code
│   ├── database/        # Database-related code
│   ├── models/          # Embedding models
│   ├── processors/      # Document processing logic
│   └── main.py          # Main entry point
├── venv/                # Virtual environment
└── README.md            # This file
```

## Setup

1. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```
   pip install python-frontmatter markdown asyncio aiohttp pyyaml sentence-transformers openai pinecone sqlalchemy
   ```

3. Set up environment variables or edit the config file:
   ```
   export OPENAI_API_KEY="your-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   export PINECONE_ENVIRONMENT="your-pinecone-environment"
   export PINECONE_INDEX_NAME="your-index-name"
   ```

## Usage

Run the pipeline with:

```
python src/main.py
```

Additional options:

- `--mode [bulk|incremental|auto]`: Processing mode
- `--workers N`: Number of worker processes
- `--verbose`: Enable verbose logging

## Features

- Processes markdown files with frontmatter metadata
- Implements semantic chunking
- Generates embeddings using OpenAI or sentence-transformers
- Tracks document states in SQLite database
- Supports bulk and incremental processing modes
- Integrates with Pinecone vector database

## Implementation Status

This project is currently in development. See `instructions/` directory for implementation plan and details.
