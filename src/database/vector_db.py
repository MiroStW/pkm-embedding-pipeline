"""
Vector database uploader module for sending embeddings to Pinecone.
"""
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec, PodSpec

from src.models import EmbeddingModelFactory
from src.config import ConfigManager

logger = logging.getLogger(__name__)

class VectorDatabaseUploader:
    """
    Handles uploading embeddings to the vector database (Pinecone) with retry logic.
    """

    def __init__(self,
                 api_key: str,
                 environment: str,
                 index_name: str,
                 dimension: int = 1024,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 batch_size: int = 100):
        """
        Initialize the vector database uploader.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Vector dimension
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts (seconds)
            batch_size: Number of vectors to upload in a single batch
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.pc = None
        self.index = None

        # Initialize connection
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize connection to Pinecone."""
        try:
            # Initialize Pinecone with the latest API (v6.x)
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index '{self.index_name}' with dimension {self.dimension}")

                # Create the index with appropriate specs
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Free tier supported region
                    )
                )

                # Wait for index to be ready
                while self.index_name not in self.pc.list_indexes().names():
                    logger.info("Waiting for index to be created...")
                    time.sleep(1)

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index '{self.index_name}'")

        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            raise

    def _with_retry(self, operation: callable, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function

        Returns:
            Result from the operation

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {str(e)}")

                if attempt < self.max_retries:
                    # Calculate backoff delay with jitter
                    delay = self.retry_delay * (1.5 ** (attempt - 1))
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

                    # Recreate connection if needed
                    if self.index is None:
                        self._init_connection()

        # If we get here, all retries failed
        logger.error(f"All retry attempts failed: {str(last_exception)}")
        raise last_exception

    def upload_vectors(self, vectors: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Upload vectors to Pinecone.

        Args:
            vectors: List of vector dictionaries, each with:
                - id: Vector ID
                - values: Vector values (embedding)
                - metadata: Dictionary of metadata

        Returns:
            Tuple of (success_count, error_count)
        """
        if not vectors:
            logger.warning("No vectors provided for upload")
            return 0, 0

        total_vectors = len(vectors)
        success_count = 0
        error_count = 0

        # Process in batches
        for i in range(0, total_vectors, self.batch_size):
            batch = vectors[i:i + self.batch_size]
            batch_size = len(batch)

            try:
                # Upload batch with retry
                self._with_retry(self.index.upsert, vectors=batch)
                success_count += batch_size
                logger.info(f"Uploaded batch of {batch_size} vectors ({i+batch_size}/{total_vectors})")
            except Exception as e:
                error_count += batch_size
                logger.error(f"Failed to upload batch ({i}/{total_vectors}): {str(e)}")

        return success_count, error_count

    def upload_document_chunks(self,
                               document_id: str,
                               chunks: List[Dict[str, Any]],
                               embeddings: List[List[float]],
                               title_enhanced_embeddings: Optional[List[List[float]]] = None) -> Tuple[int, int]:
        """
        Upload document chunks with their embeddings to Pinecone.

        Args:
            document_id: The unique ID of the document
            chunks: List of document chunks
            embeddings: List of embeddings corresponding to chunks
            title_enhanced_embeddings: Optional list of title-enhanced embeddings

        Returns:
            Tuple of (success_count, error_count)
        """
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
            return 0, len(chunks)

        if title_enhanced_embeddings and len(chunks) != len(title_enhanced_embeddings):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and title-enhanced embeddings ({len(title_enhanced_embeddings)})")
            return 0, len(chunks)

        vectors = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a stable, unique ID for each chunk
            chunk_id = f"{document_id}_{i}"

            # Prepare metadata from chunk
            metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk.get("content", ""),
                "section_title": chunk.get("metadata", {}).get("section_title", ""),
                "section_level": chunk.get("metadata", {}).get("section_level", 0)
            }

            # Add any additional metadata from the chunk
            for key, value in chunk.get("metadata", {}).items():
                if key not in metadata and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value

            # Create the vector record
            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            }

            vectors.append(vector)

            # If we have title-enhanced embeddings, create a separate vector with "_enhanced" suffix
            if title_enhanced_embeddings:
                enhanced_vector = {
                    "id": f"{chunk_id}_enhanced",
                    "values": title_enhanced_embeddings[i],
                    "metadata": {**metadata, "embedding_type": "title_enhanced"}
                }
                vectors.append(enhanced_vector)

        return self.upload_vectors(vectors)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.

        Args:
            document_id: The unique ID of the document

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Using filter metadata to delete all vectors for this document
            if self.index is None:
                self._init_connection()

            # For Serverless and Starter indexes, we need to get the IDs first
            # as metadata filtering in delete operations is not supported
            try:
                # First query to get all vector IDs for this document
                results = self._with_retry(
                    self.index.query,
                    vector=[0.1] * self.dimension,  # Dummy vector for metadata query
                    top_k=10000,  # Set high to get all vectors
                    filter={"document_id": {"$eq": document_id}},
                    include_metadata=False
                )

                # Extract IDs from results
                ids_to_delete = [match.get('id') for match in results.get('matches', [])]

                if ids_to_delete:
                    logger.info(f"Deleting {len(ids_to_delete)} vectors for document {document_id}")
                    # Delete vectors by ID
                    self._with_retry(
                        self.index.delete,
                        ids=ids_to_delete
                    )
                else:
                    logger.info(f"No vectors found for document {document_id}")

                return True

            except Exception as e:
                if "not support deleting with metadata filtering" in str(e):
                    # Alternative approach: delete using ID prefix
                    # This relies on our ID naming convention: document_id_chunk_index
                    logger.info(f"Falling back to ID prefix-based deletion for document {document_id}")

                    # Delete vectors with prefix matching (this is a specialized method)
                    self._with_retry(
                        self.index.delete,
                        ids=f"{document_id}_*"  # Using wildcard pattern if supported
                    )
                    return True
                else:
                    # Re-raise if it's a different error
                    raise e

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dictionary with statistics
        """
        try:
            stats = self._with_retry(self.index.describe_index_stats)
            return stats
        except Exception as e:
            logger.error(f"Failed to get vector database stats: {str(e)}")
            return {}

    def query_test(self, vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Run a test query on the vector database.

        Args:
            vector: Query vector
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        try:
            results = self._with_retry(
                self.index.query,
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )

            return results.matches
        except Exception as e:
            logger.error(f"Error querying vector database: {str(e)}")
            return []

    async def index_document(self, document_result: Dict[str, Any]) -> bool:
        """
        Index a document in the vector database.
        Used by the pipeline orchestrator to process document results.

        Args:
            document_result: Document processing result containing metadata and chunks

        Returns:
            True if successful, False otherwise
        """
        try:
            if document_result.get('status') != 'success':
                logger.error(f"Cannot index document with status: {document_result.get('status')}")
                return False

            document_id = document_result.get('metadata', {}).get('id')
            if not document_id:
                logger.error("Document has no ID, cannot index")
                return False

            chunks = document_result.get('chunks', [])
            if not chunks:
                logger.warning(f"Document {document_id} has no chunks to index")
                return False

            # Get embedding config
            config_manager = ConfigManager()
            embedding_config = config_manager.get_embedding_config()
            logger.debug(f"Using embedding config: {embedding_config}")

            logger.info(f"Creating embedding model for document {document_id}")
            try:
                # Create embedding model
                embedding_model = await EmbeddingModelFactory.create_model(embedding_config)
                logger.debug("Successfully created embedding model")
            except Exception as e:
                logger.error(f"Failed to create embedding model: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False

            # Generate content embeddings
            content_texts = [chunk.get('content', '') for chunk in chunks]
            logger.info(f"Generating {len(content_texts)} regular embeddings for document {document_id}")

            try:
                embeddings = await embedding_model.generate_embeddings(content_texts)
                logger.info(f"Generated {len(embeddings)} regular embeddings for document {document_id}")
            except Exception as e:
                logger.error(f"Error generating embeddings for document {document_id}: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Error generating embeddings for document {document_id}: {str(e)}")

            # Check if title-enhanced embeddings are enabled
            title_enhanced_embeddings = None
            if embedding_config.get('enable_title_enhanced', True):
                logger.info(f"Title-enhanced embeddings are enabled with weight {embedding_config.get('title_weight', 0.3)}")
                # Generate title-enhanced embeddings
                title_weight = embedding_config.get('title_weight', 0.3)

                # Prepare batch input for title-enhanced embeddings
                enhanced_batch = []
                for i, chunk in enumerate(chunks):
                    title = chunk.get('metadata', {}).get('section_title', '') or document_result.get('metadata', {}).get('title', '')
                    content = chunk.get('content', '')

                    if title and content:
                        enhanced_batch.append({
                            'title': title,
                            'content': content,
                            'index': i
                        })

                # Generate title-enhanced embeddings in batch if we have any
                if enhanced_batch:
                    logger.info(f"Generating {len(enhanced_batch)} title-enhanced embeddings for document {document_id}")

                    # Process the batch more efficiently
                    title_enhanced_embeddings = [None] * len(chunks)

                    # Process in smaller batches to avoid memory issues (optional)
                    batch_size = 10  # Adjust based on available memory
                    for i in range(0, len(enhanced_batch), batch_size):
                        batch_slice = enhanced_batch[i:i + batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1}/{(len(enhanced_batch) + batch_size - 1)//batch_size} of title-enhanced embeddings")

                        # Generate embeddings for this batch slice
                        for item in batch_slice:
                            try:
                                enhanced = await embedding_model.generate_title_enhanced_embedding(
                                    title=item['title'],
                                    content=item['content'],
                                    title_weight=title_weight
                                )
                                title_enhanced_embeddings[item['index']] = enhanced
                            except Exception as e:
                                logger.error(f"Error generating enhanced embedding for chunk {item['index']}: {str(e)}")
                                # Use regular embedding as fallback
                                if item['index'] < len(embeddings):
                                    title_enhanced_embeddings[item['index']] = embeddings[item['index']]

                    # Fill in any missing title-enhanced embeddings with regular embeddings
                    missing_count = sum(1 for x in title_enhanced_embeddings if x is None)
                    if missing_count > 0:
                        logger.info(f"Filling in {missing_count} missing title-enhanced embeddings with regular embeddings")

                    for i in range(len(chunks)):
                        if title_enhanced_embeddings[i] is None:
                            title_enhanced_embeddings[i] = embeddings[i]
                else:
                    logger.info(f"No titles found for document {document_id}, using regular embeddings")
                    # No titles available, use regular embeddings
                    title_enhanced_embeddings = embeddings
            else:
                logger.info(f"Title-enhanced embeddings are disabled for document {document_id}")

            # Upload to vector database
            logger.info(f"Uploading document {document_id} with {len(embeddings)} regular embeddings and {len(title_enhanced_embeddings) if title_enhanced_embeddings else 0} title-enhanced embeddings")
            try:
                success_count, error_count = self.upload_document_chunks(
                    document_id=document_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    title_enhanced_embeddings=title_enhanced_embeddings
                )

                if error_count > 0:
                    logger.warning(f"Indexed document {document_id} with {error_count} errors")

                return success_count > 0
            except Exception as e:
                logger.error(f"Error uploading document chunks: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False

        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False