---
id: i0g33enw6jpdopbb2c8ctpz
title: Embedding Pipeline
desc: Technical implementation details for the embedding pipeline of the PKM chatbot
updated: 1742985801100
created: "2024-06-24T00:00:00.000Z"
---

# PKM Chatbot Embedding Pipeline

## Overview

This document outlines the technical implementation decisions for the local embedding pipeline component of the PKM chatbot. The pipeline is responsible for processing markdown files, generating vector embeddings, and synchronizing them with the cloud vector database.

## Core Technology Decisions

### 1. Programming Language & Framework

- **Language**: Python
- **Core Libraries**: `asyncio`, `aiohttp` for asynchronous processing
- **Rationale**: Python offers the richest ecosystem for NLP tasks and embedding generation, with excellent library support for markdown processing and vector operations.

### 2. Markdown Processing

#### Document Parsing

- **Libraries**: `python-frontmatter` for metadata extraction, `markdown` for content parsing
- **Implementation**: Custom parser that preserves document structure and metadata

#### Chunking Strategy

- **Approach**: Semantic chunking based on document structure
- **Implementation Details**:
  - Primary chunks follow natural document sections (h1, h2 boundaries)
  - Maximum chunk size of 1024 tokens to balance context and specificity
  - Overlap of 100 tokens between chunks to maintain context across boundaries
  - Special handling for lists, tables, and code blocks to keep them intact
- **Rationale**: Semantic chunking preserves the logical structure of documents, making retrieved chunks more coherent and useful for LLM context than arbitrary splits.

### 3. Embedding Model

#### Model Selection

- **Primary Model**: `text-embedding-3-large` from OpenAI or equivalent high-quality model
- **Fallback Model**: `all-mpnet-base-v2` from sentence-transformers
- **Rationale**: Prioritizing embedding quality over processing speed, as this is a personal system with moderate data volume. Model can be easily replaced as technology evolves.

#### Embedding Strategy

- **Document Representations**:
  - Primary embeddings: Full chunk content embeddings
  - Auxiliary embeddings: Title-enhanced embeddings (title + content combined)
  - Metadata stored alongside vectors (not embedded): id, date, title, tags
- **Metadata Indexing**: Implement explicit filtering on date, title, and tags
- **Rationale**: This hybrid approach enhances retrieval quality by allowing both semantic search and metadata filtering. Title-enhanced embeddings improve relevance for topic-based searches.

### 4. Change Detection & Synchronization

#### Tracking Mechanism

- **Primary Approach**: Git-based tracking using local git hooks
  - `post-commit` hook to flag changed files for processing
  - `post-merge` hook to handle pulls and fetches
- **Supplementary**: Simple tracking database for state management
- **Rationale**: Git hooks integrate seamlessly with the existing GitLab workflow while the tracking database manages processing state and handles failures.

#### Git Hook Implementation Considerations

- **Potential Issues**:
  - Hooks don't track deleted files automatically
  - Local hook setup required on each development machine
- **Solutions**:
  - Use `git diff` to identify deleted files
  - Include hook installation in setup documentation
  - Store minimal state in tracking database

### 5. Architecture Pattern

#### Adapted Producer-Consumer Pattern

- **Operational Scenarios**:

  - **Initial Bulk Embedding** (~4500 files): Full corpus processing with pause/resume capability
  - **Incremental Updates** (<10 files typically): Lightweight processing for daily changes

- **Components**:

  - **Document Processor** (Producer): Identifies files for processing, extracts content, generates work items
  - **Adaptive Work Queue**:
    - Persistent queue (SQLite-backed) for bulk operations enabling pause/resume
    - In-memory queue for small incremental changes
  - **Embedding Workers** (Consumers):
    - Auto-scaling worker pool based on workload size
    - Single worker for small batches, multiple for bulk operations
  - **Database Uploader**: Handles vector database API communication with retry capability

- **Processing Modes**:

  - **Bulk Processing Mode**:

    - Automatically activated when file count exceeds threshold
    - Persistent queue with progress tracking and checkpoints
    - Configurable batch sizes and maximum parallel workers
    - Ability to prioritize certain document types

  - **Incremental Mode**:
    - Used for daily operations with small file changes
    - Lightweight in-memory queue
    - Optional single-threaded processing for very small batches
    - Immediate processing option that bypasses complex queuing

- **Flow**:

  1. System detects operation mode based on file count or explicit flag
  2. Git hooks or manual command triggers the Document Processor
  3. Processor configures appropriate queue and worker count based on mode
  4. Files are processed through the queue system
  5. Database Uploader sends completed embeddings to cloud vector database

- **Rationale**:
  - This adaptive approach provides optimal performance for both initial embedding and daily updates
  - Reuses core processing logic while adapting resource allocation to workload
  - Enables pause/resume for long-running initial embedding
  - Simplifies processing for typical small daily changes

#### Pipeline Configuration

- **Configuration Management**: YAML configuration file with environment variable overrides
- **Tunable Parameters**:
  - Number of worker processes
  - Chunk size and overlap settings
  - Embedding model selection
  - Vector database connection details
  - Processing batch sizes

## Implementation

For the detailed implementation phases and plan, see the [PKM Chatbot Architecture](tech.projects.pkm-chatbot.architecture.md) document.
