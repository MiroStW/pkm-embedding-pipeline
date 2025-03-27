# PKM Chatbot Embedding Pipeline CLI

This document provides information about the Command-Line Interface (CLI) for the PKM Chatbot embedding pipeline, including automation features and scheduled processing.

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

The CLI can be accessed using:

```bash
python -m src.cli.main [command] [options]
```

## Available Commands

### Process Documents

Process documents through the embedding pipeline:

```bash
# Process all documents in auto mode
python -m src.cli.main process

# Process all documents in bulk mode
python -m src.cli.main process --mode bulk

# Process a single file
python -m src.cli.main process --file path/to/file.md

# Process all markdown files in a directory
python -m src.cli.main process --directory path/to/directory

# Resume from last checkpoint
python -m src.cli.main process --resume

# Force reprocessing of already processed files
python -m src.cli.main process --force
```

#### Options

- `--mode`: Processing mode (`bulk`, `incremental`, or `auto`)
- `--workers`: Number of worker processes
- `--verbose` or `-v`: Enable verbose output
- `--file`: Process a single file
- `--directory`: Process all markdown files in a directory
- `--resume`: Resume from last checkpoint
- `--batch-size`: Batch size for bulk processing
- `--adaptive-scaling`: Enable adaptive worker pool scaling
- `--force`: Force processing of already processed files

### Check Status

View the current processing status:

```bash
# Show summary status
python -m src.cli.main status

# Show detailed status
python -m src.cli.main status --detailed

# Export status as JSON
python -m src.cli.main status --format json --output status.json

# Filter status to show only failed documents
python -m src.cli.main status --filter failed
```

#### Options

- `--detailed`: Show detailed status with file information
- `--format`: Output format (`table`, `json`, or `csv`)
- `--output`: Output file path
- `--filter`: Filter by document state (`all`, `completed`, `failed`, or `pending`)

### Schedule Processing

Schedule automated pipeline runs:

```bash
# Schedule daily processing at 2 AM
python -m src.cli.main schedule --time "02:00"

# Schedule processing every 30 minutes
python -m src.cli.main schedule --interval 30

# List current scheduled tasks
python -m src.cli.main schedule --list

# Remove existing scheduled tasks
python -m src.cli.main schedule --remove
```

#### Options

- `--time`: Time to run (format: `HH:MM`)
- `--interval`: Interval in minutes
- `--list`: List scheduled tasks
- `--remove`: Remove existing schedule

### Verify Pipeline Integrity

Verify the integrity of different pipeline components:

```bash
# Verify Pinecone integration
python -m src.cli.main verify --pinecone

# Verify document database
python -m src.cli.main verify --db

# Verify and fix issues
python -m src.cli.main verify --pinecone --db --fix
```

#### Options

- `--pinecone`: Verify Pinecone integration
- `--db`: Verify document database
- `--fix`: Attempt to fix issues

### ID Management

Generate and manage document IDs:

```bash
# Generate a new unique ID
python -m src.cli.main id --generate

# Add ID to a specific markdown file
python -m src.cli.main id --file path/to/file.md

# Add IDs to all markdown files in a directory
python -m src.cli.main id --directory path/to/directory

# Force ID generation even if ID exists
python -m src.cli.main id --directory path/to/directory --force
```

#### Options

- `--generate`: Generate a new unique ID
- `--file`: Add ID to a specific markdown file
- `--directory`: Add IDs to all markdown files in a directory
- `--force`: Force ID generation even if ID exists
- `--format`: Output format (`table`, `json`, or `csv`)
- `--output`: Output file for results

## Automation Scripts

### Scheduled Processing

Run the pipeline on a schedule using the script:

```bash
./scripts/run_scheduled_processing.py --mode auto
```

#### Options

- `--mode`: Processing mode (`auto`, `bulk`, or `incremental`)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, or `ERROR`)

### Git Hooks Installation

Install git hooks for automatic tracking of document changes:

```bash
# Install hooks for current directory
./scripts/install_git_hooks.py

# Install hooks for specific repository
./scripts/install_git_hooks.py --git-dir /path/to/repo

# Force overwrite existing hooks
./scripts/install_git_hooks.py --force

# Uninstall hooks
./scripts/install_git_hooks.py --uninstall
```

#### Options

- `--git-dir`: Path to the git repository (defaults to current directory)
- `--force`: Force overwrite existing hooks
- `--uninstall`: Uninstall hooks instead of installing them

## Environment Configuration

The CLI uses the same configuration as the main pipeline, which is loaded from `config/config.yaml`. See the main documentation for details on configuration options.

## Logging

Logs are stored in:

- Main logs: `logs/pipeline.log`
- Scheduled run logs: `logs/scheduled/scheduled_run_[timestamp].log`

## Error Handling

The CLI provides detailed error messages and returns appropriate exit codes:

- `0`: Success
- `1`: Error

## Examples

### Process all documents and save detailed status to CSV

```bash
python -m src.cli.main process && python -m src.cli.main status --detailed --format csv --output status.csv
```

### Set up daily processing and verify integration

```bash
python -m src.cli.main verify --pinecone --db && python -m src.cli.main schedule --time "03:30"
```

### Process failed documents only

```bash
python -m src.cli.main status --filter failed --format json | jq -r '.documents[].file' | xargs -I{} python -m src.cli.main process --file "{}" --force
```
