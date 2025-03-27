"""
Vector database uploader module for sending embeddings to Pinecone.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, PodSpec

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
            self.pc = Pinecone(api_key=self.api_key, environment=self.environment)

            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating Pinecone index '{self.index_name}' with dimension {self.dimension}")

                # Create the index
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
                while not self.index_name in self.pc.list_indexes().names():
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
            # Delete vectors with the document ID prefix
            self._with_retry(self.index.delete, filter={"document_id": document_id})
            logger.info(f"Deleted all vectors for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors for document {document_id}: {str(e)}")
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

    def index_document(self, document_result: Dict[str, Any]) -> bool:
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

            # For now, we're mocking the embeddings generation
            # In a real implementation, this would use a model to generate embeddings
            mock_embeddings = [[0.1] * self.dimension for _ in range(len(chunks))]

            # Upload to vector database
            success_count, error_count = self.upload_document_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=mock_embeddings
            )

            if error_count > 0:
                logger.warning(f"Indexed document {document_id} with {error_count} errors")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False