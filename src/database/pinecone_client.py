"""
Pinecone integration for the embedding pipeline.
Provides specialized functionality for managing vectors in Pinecone.
"""
import logging
import time
from pinecone import Pinecone, ServerlessSpec, PodSpec
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import traceback
import re
import json
import datetime

from src.database.vector_db import VectorDatabaseUploader

logger = logging.getLogger(__name__)

class PineconeClient(VectorDatabaseUploader):
    """
    Enhanced Pinecone client implementation for the embedding pipeline.
    Extends VectorDatabaseUploader with additional Pinecone-specific functionality.
    """

    def __init__(self,
                 api_key: str,
                 environment: str,
                 index_name: str,
                 dimension: int = 1024,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 batch_size: int = 100,
                 serverless: bool = False,
                 cloud_provider: str = "aws",
                 region: str = "us-east-1"):
        """
        Initialize the Pinecone client.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Vector dimension
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts (seconds)
            batch_size: Number of vectors to upload in a single batch
            serverless: Whether to use serverless deployment (default: False)
            cloud_provider: Cloud provider for serverless (aws, gcp, azure)
            region: Region for serverless deployment
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.serverless = serverless
        self.cloud_provider = cloud_provider
        self.region = region
        self.pc = None
        self.index = None

        # Initialize connection
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize connection to Pinecone with enhanced options."""
        try:
            # Initialize Pinecone client with the latest API (v6.x)
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index '{self.index_name}' with dimension {self.dimension}")

                # DEBUG: Print the index name right before creating
                print(f"DEBUG [_init_connection]: Attempting to create index with name: '{self.index_name}'")

                # Create the index with appropriate specs
                if self.serverless:
                    # Use serverless spec
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=self.cloud_provider,
                            region=self.region
                        )
                    )
                else:
                    # Use pod-based spec
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=PodSpec(
                            environment=self.environment,
                            pod_type="p1.x1"  # Use appropriate pod type based on needs
                        )
                    )

                # Wait for index to be ready
                while self.index_name not in self.pc.list_indexes().names():
                    logger.info("Waiting for index to be created...")
                    time.sleep(1)

                logger.info(f"Index '{self.index_name}' created successfully")

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index '{self.index_name}'")

        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def check_connection(self) -> bool:
        """
        Check if connection to Pinecone is valid.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            if self.index is None:
                self._init_connection()

            # Simple operation to check connection
            self.index.describe_index_stats()
            return True
        except Exception as e:
            logger.warning(f"Connection to Pinecone is invalid: {str(e)}")
            return False

    def query_vectors(self,
                     query_vector: List[float],
                     top_k: int = 5,
                     filter: Optional[Dict[str, Any]] = None,
                     include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Query vectors from Pinecone based on similarity.

        Args:
            query_vector: Vector to query against
            top_k: Number of results to return
            filter: Optional metadata filters
            include_metadata: Whether to include metadata in response

        Returns:
            List of matching vectors with scores and metadata
        """
        try:
            if self.index is None:
                self._init_connection()

            query_result = self._with_retry(
                self.index.query,
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter
            )

            return query_result.get('matches', [])
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return []

    def query_text(self,
                  text: str,
                  embedding_model,
                  top_k: int = 5,
                  filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query vectors using text input that gets converted to an embedding.

        Args:
            text: Text to query with
            embedding_model: Model to use for generating embeddings
            top_k: Number of results to return
            filter: Optional metadata filters

        Returns:
            List of matching vectors with scores and metadata
        """
        try:
            # Generate embedding for the text
            embedding = embedding_model.encode(text)

            # Query using the embedding
            return self.query_vectors(
                query_vector=embedding,
                top_k=top_k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Error querying with text: {str(e)}")
            return []

    def sync_document(self,
                     document_id: str,
                     new_chunks: List[Dict[str, Any]],
                     embeddings: List[List[float]],
                     title_enhanced_embeddings: Optional[List[List[float]]] = None) -> bool:
        """
        Synchronize a document by deleting old vectors and uploading new ones.

        Args:
            document_id: The unique ID of the document
            new_chunks: List of new document chunks
            embeddings: List of embeddings corresponding to chunks
            title_enhanced_embeddings: Optional title-enhanced embeddings

        Returns:
            True if synchronization was successful, False otherwise
        """
        try:
            # First delete the existing document vectors
            if not self.delete_document(document_id):
                logger.warning(f"Failed to delete existing vectors for document {document_id}")
                # Continue anyway to upload new vectors

            # Then upload the new vectors
            success_count, error_count = self.upload_document_chunks(
                document_id=document_id,
                chunks=new_chunks,
                embeddings=embeddings,
                title_enhanced_embeddings=title_enhanced_embeddings
            )

            # Check if all uploads succeeded
            if error_count > 0:
                logger.warning(f"Errors during synchronization of document {document_id}: {error_count} errors")
                return False

            logger.info(f"Successfully synchronized document {document_id} with {success_count} vectors")
            return True

        except Exception as e:
            logger.error(f"Error synchronizing document {document_id}: {str(e)}")
            return False

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the index.

        Returns:
            Total vector count or 0 if error
        """
        try:
            if self.index is None:
                self._init_connection()

            stats = self.index.describe_index_stats()
            return stats.get('total_vector_count', 0)
        except Exception as e:
            logger.error(f"Error getting vector count: {str(e)}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        try:
            if self.index is None:
                self._init_connection()

            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index statistics: {str(e)}")
            return {"error": str(e)}

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

    def get_document_ids(self) -> List[str]:
        """
        Get all unique document IDs in the index.
        Note: This is an expensive operation that performs a full scan.

        Returns:
            List of document IDs
        """
        try:
            # This requires a metadata query which might be limited in size
            # A more efficient approach would be to store document IDs elsewhere
            if self.index is None:
                self._init_connection()

            # This is a simplified implementation; in practice, you'd need pagination
            # since this operation could time out for large indexes
            query_result = self._with_retry(
                self.index.query,
                vector=[0] * self.dimension,  # Dummy vector
                top_k=10000,  # Adjust based on expected document count
                include_metadata=True
            )

            # Extract unique document IDs from metadata
            document_ids = set()
            for match in query_result.get('matches', []):
                doc_id = match.get('metadata', {}).get('document_id')
                if doc_id:
                    document_ids.add(doc_id)

            return list(document_ids)

        except Exception as e:
            logger.error(f"Error retrieving document IDs: {str(e)}")
            return []

    def verify_document_indexed(self, document_id: str) -> bool:
        """
        Verify that a document is properly indexed in Pinecone.

        Args:
            document_id: The document ID to check

        Returns:
            True if document is indexed, False otherwise
        """
        try:
            if self.index is None:
                self._init_connection()

            # Query for vectors with this document_id
            query_result = self._with_retry(
                self.index.query,
                vector=[0] * self.dimension,  # Dummy vector
                top_k=1,
                include_metadata=True,
                filter={"document_id": document_id}
            )

            # Check if we got any matches
            return len(query_result.get('matches', [])) > 0

        except Exception as e:
            logger.error(f"Error verifying document {document_id}: {str(e)}")
            return False

    def bulk_delete(self, filter: Dict[str, Any]) -> bool:
        """
        Delete multiple vectors based on a filter.

        Note: For free tier, this uses client-side filtering since metadata filtering
        is not supported in the free tier. This may be inefficient for large indexes.

        Args:
            filter: Metadata filter to identify vectors to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if self.index is None:
                self._init_connection()

            # For free tier, we need to query and filter client-side
            dummy_vector = [0.1] * self.dimension

            # Query for all vectors
            results = self._with_retry(
                self.index.query,
                vector=dummy_vector,
                top_k=1000,  # Adjust based on expected index size
                include_metadata=True
            )

            # Filter based on metadata
            vector_ids = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                # Check if all filter key-value pairs are in metadata
                if all(metadata.get(k) == v for k, v in filter.items()):
                    vector_ids.append(match.get('id'))

            if not vector_ids:
                logger.warning(f"No vectors found matching filter: {filter}")
                return False

            logger.info(f"Deleting {len(vector_ids)} vectors matching filter: {filter}")
            self._with_retry(self.index.delete, ids=vector_ids)

            logger.info(f"Bulk deleted vectors with filter: {filter}")
            return True

        except Exception as e:
            logger.error(f"Error during bulk deletion: {str(e)}")
            return False

    def generate_vector_id(self, document_id: str, chunk_index: int, is_enhanced: bool = False) -> str:
        """
        Generate a stable, unique ID for a vector.

        Args:
            document_id: Document ID
            chunk_index: Chunk index within document
            is_enhanced: Whether this is a title-enhanced vector

        Returns:
            Unique vector ID
        """
        if is_enhanced:
            return f"{document_id}_{chunk_index}_enhanced"
        return f"{document_id}_{chunk_index}"

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors associated with a document.

        Args:
            document_id: The unique ID of the document

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
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

    async def index_document(self, document_result: Dict[str, Any]) -> bool:
        """
        Index a document in Pinecone.
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

            # Look for 'document_id' in metadata
            document_id = document_result.get('metadata', {}).get('document_id')
            if not document_id:
                logger.error("Document result metadata has no 'document_id', cannot index")
                # Log the metadata for debugging
                logger.debug(f"Metadata received: {document_result.get('metadata')}")
                return False

            logger.info(f"Processing document with ID: {document_id}") # Log the ID being used

            chunks = document_result.get('chunks', [])
            if not chunks:
                logger.warning(f"Document {document_id} has no chunks to index")
                return False

            # Get embedding config
            from src.config import ConfigManager
            config_manager = ConfigManager()
            embedding_config = config_manager.get_embedding_config()
            logger.debug(f"Using embedding config: {embedding_config}")

            logger.info(f"Creating embedding model for document {document_id}")
            try:
                # Create embedding model
                from src.models import EmbeddingModelFactory
                embedding_model = await EmbeddingModelFactory.create_model(embedding_config)
                logger.debug("Successfully created embedding model")
            except Exception as e:
                logger.error(f"Failed to create embedding model: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False

            # Generate content embeddings
            content_texts = [chunk.get('content', '') for chunk in chunks]
            logger.info(f"Generating {len(content_texts)} regular embeddings for document {document_id}")
            print(f"DEBUG [index_document]: About to generate {len(content_texts)} embeddings for {document_id}") # DEBUG PRINT

            try:
                embeddings = await embedding_model.generate_embeddings(content_texts)
                # Explicitly check if embeddings generation returned None (indicating an error)
                if embeddings is None:
                    logger.error(f"Embedding generation failed for document {document_id}, received None.")
                    print(f"DEBUG [index_document]: Embedding generation returned None for {document_id}, returning False") # DEBUG PRINT
                    return False

                logger.info(f"Generated {len(embeddings)} regular embeddings for document {document_id}")
                print(f"DEBUG [index_document]: Successfully generated {len(embeddings)} embeddings for {document_id}") # DEBUG PRINT
                # Check if embeddings list is empty or contains invalid items (e.g., empty lists if model returned them despite the fix)
                if not embeddings or not all(embeddings):
                    logger.error(f"Generated empty or invalid embeddings for document {document_id}")
                    print(f"DEBUG [index_document]: Generated empty/invalid embeddings for {document_id}, returning False") # DEBUG PRINT
                    return False

            except Exception as e:
                logger.error(f"Error generating embeddings for document {document_id}: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                print(f"DEBUG [index_document]: Exception during embedding generation for {document_id}: {e}") # DEBUG PRINT
                return False

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
                # Add a print after title enhanced generation completes or is skipped
                print(f"DEBUG [index_document]: Finished title-enhanced embedding step for {document_id}") # DEBUG PRINT
            else:
                logger.info(f"Title-enhanced embeddings are disabled for document {document_id}")
                print(f"DEBUG [index_document]: Title-enhanced embeddings disabled for {document_id}") # DEBUG PRINT

            # Ensure we fall back correctly if title enhancement was enabled but failed
            if embedding_config.get('enable_title_enhanced', True) and title_enhanced_embeddings is None:
                logger.warning(f"Title enhanced was enabled but result is None for {document_id}, falling back to regular embeddings.")
                print(f"DEBUG [index_document]: Falling back title_enhanced to regular embeddings for {document_id}") # DEBUG PRINT
                title_enhanced_embeddings = embeddings

            # Upload to Pinecone
            logger.info(f"Uploading document {document_id} with {len(embeddings)} regular embeddings and {len(title_enhanced_embeddings) if title_enhanced_embeddings else 0} title-enhanced embeddings")
            print(f"DEBUG [index_document]: About to call upload_document_chunks for document {document_id}") # DEBUG PRINT
            try:
                success_count, error_count = await self.upload_document_chunks(
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

    async def upload_document_chunks(self,
                               document_id: str,
                               chunks: List[Dict[str, Any]],
                               embeddings: List[List[float]],
                               title_enhanced_embeddings: Optional[List[List[float]]] = None) -> Tuple[int, int]:
        """
        Upload document chunks to Pinecone.

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

        if title_enhanced_embeddings and len(title_enhanced_embeddings) != len(chunks):
            logger.error(f"Mismatch between chunks ({len(chunks)}) and title-enhanced embeddings ({len(title_enhanced_embeddings)})")
            return 0, len(chunks)

        # PRINT THE RECEIVED document_id
        print(f"DEBUG [upload_document_chunks]: Received document_id: '{document_id}'")

        # Prepare vectors for upload
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Get raw file name for metadata
            raw_file_name = chunk.get("metadata", {}).get("file_name", "unknown_file")

            # Create unique vector ID using the main document_id and chunk index
            # Ensure document_id itself is safe (should be, if generated correctly)
            # Let's assume document_id is already clean (e.g., UUID, hash)
            vector_id = f"doc-{document_id}-chunk-{i}"

            # Prepare metadata - ensure raw file_name is included for reference
            metadata = {
                "document_id": document_id, # Main document ID
                "chunk_index": i,
                "content": chunk.get("content", ""),
                # Include original metadata, ensuring file_name is present
                **chunk.get("metadata", {})
            }
            if "file_name" not in metadata:
                 metadata["file_name"] = raw_file_name # Store raw name for filtering

            # --- Sanitize metadata for Pinecone compatibility ---
            sanitized_metadata = {}
            for key, value in metadata.items():
                # 1. Sanitize the key (replace space with _, remove other invalid chars)
                sanitized_key = key.replace(' ', '_')
                sanitized_key = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized_key)

                # Ensure key is not empty after sanitization
                if not sanitized_key:
                    logger.warning(f"Skipping metadata field with original key '{key}' because key became empty after sanitization (doc {document_id})")
                    continue

                # 2. Skip if the value is None
                if value is None:
                    logger.debug(f"Skipping metadata field '{sanitized_key}' because its value is None (doc {document_id})")
                    continue

                # 3. Sanitize the value based on allowed types
                if isinstance(value, (datetime.date, datetime.datetime)):
                    sanitized_value = value.isoformat()
                elif isinstance(value, (str, int, float, bool)):
                    sanitized_value = value # Keep allowed scalar types
                elif isinstance(value, list):
                    # Ensure list contains only strings (as per error message)
                    string_list = []
                    valid_list = True
                    for item in value:
                        if isinstance(item, str):
                            string_list.append(item)
                        else:
                            # Convert non-strings to strings
                            logger.warning(f"Converting non-string item {type(item)} in list for key '{sanitized_key}' to string (doc {document_id})")
                            string_list.append(str(item))
                    sanitized_value = string_list
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings as they might not be supported directly
                    logger.warning(f"Converting dict metadata type to JSON string for key '{sanitized_key}' (doc {document_id})")
                    try:
                        sanitized_value = json.dumps(value)
                    except Exception:
                        logger.error(f"Could not convert dict to JSON string for key '{sanitized_key}', converting to basic string.")
                        sanitized_value = str(value) # Fallback
                else:
                    # Convert any other unsupported types to string as a fallback
                    logger.warning(f"Converting unsupported metadata type {type(value)} to string for key '{sanitized_key}' (doc {document_id})")
                    sanitized_value = str(value)

                # Add the sanitized key and value
                sanitized_metadata[sanitized_key] = sanitized_value
            # ----------------------------------------------------

            # Add regular embedding vector with sanitized metadata
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": sanitized_metadata
            })

            # Add title-enhanced embedding if available
            if title_enhanced_embeddings:
                enhanced_id = f"doc-{document_id}-chunk-{i}-enhanced" # Consistent naming
                # Use the already sanitized metadata, just add the flag
                enhanced_metadata = {**sanitized_metadata, "is_title_enhanced": True}

                vectors.append({
                    "id": enhanced_id,
                    "values": title_enhanced_embeddings[i],
                    "metadata": enhanced_metadata
                })

        # Upload vectors in batches
        success_count = 0
        error_count = 0

        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            logger.debug(f"Uploading batch {i//self.batch_size + 1}/{(len(vectors) + self.batch_size - 1)//self.batch_size} for document {document_id}")
            print(f"DEBUG: Preparing to upload batch {i//self.batch_size + 1} for doc {document_id}. Batch size: {len(batch)}")

            # PRINT THE BATCH CONTENT JUST BEFORE UPSERT
            # Use json.dumps for cleaner multi-line printing, handle potential non-serializable data gracefully
            try:
                # Create a copy of the batch to modify for printing
                batch_copy_for_print = []
                for vector_data in batch:
                    vector_copy = vector_data.copy()
                    if 'values' in vector_copy:
                        vector_copy['values'] = f"<embedding vector of dim {len(vector_copy['values']) if isinstance(vector_copy['values'], list) else 'unknown'}>"
                    batch_copy_for_print.append(vector_copy)

                batch_json = json.dumps(batch_copy_for_print, indent=2, default=lambda o: '<not serializable>')
                print(f"DEBUG [upload_document_chunks]: Batch content before upsert (batch {i//self.batch_size + 1}):\n{batch_json}")
            except Exception as json_e:
                print(f"DEBUG [upload_document_chunks]: Could not serialize/modify batch for printing: {json_e}")
                # Fallback to printing limited raw info if modification fails
                print(f"DEBUG [upload_document_chunks]: Batch content (raw, limited): {[v.get('id') for v in batch]}")

            try:
                # Upload batch with retries using asyncio.to_thread for the sync call
                for attempt in range(self.max_retries):
                    try:
                        print(f"DEBUG: Attempting upsert (Attempt {attempt + 1}/{self.max_retries}) for batch {i//self.batch_size + 1}")
                        # Run the synchronous upsert in a thread
                        upsert_response = await asyncio.to_thread(
                            self.index.upsert,
                            vectors=batch
                        )
                        print(f"DEBUG: Upsert successful for batch {i//self.batch_size + 1}. Response: {upsert_response}")
                        success_count += upsert_response.upserted_count if hasattr(upsert_response, 'upserted_count') else len(batch) # Use response count if available
                        break # Exit retry loop on success
                    except Exception as e:
                        print(f"DEBUG: Upsert FAILED (Attempt {attempt + 1}) for batch {i//self.batch_size + 1}. Error: {str(e)}")
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Upload attempt {attempt + 1} failed for batch {i//self.batch_size + 1}: {str(e)}, retrying...")
                            # Use asyncio.sleep in async function
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                        else:
                            # Log final failure and raise to be caught by the outer try/except
                            logger.error(f"Final upload attempt {attempt + 1} failed for batch {i//self.batch_size + 1}: {str(e)}")
                            raise e # Re-raise the exception after final retry failure

            except Exception as e:
                print(f"DEBUG: FINAL FAILURE for batch {i//self.batch_size + 1} after retries. Error: {str(e)}")
                logger.error(f"Failed to upload batch {i//self.batch_size + 1} for document {document_id}: {str(e)}")
                error_count += len(batch) # Increment error count for the entire failed batch

        print(f"DEBUG: Finished uploading chunks for document {document_id}. Success: {success_count}, Errors: {error_count}")
        logger.info(f"Finished uploading {success_count} vectors for document {document_id} ({error_count} errors).")
        return success_count, error_count