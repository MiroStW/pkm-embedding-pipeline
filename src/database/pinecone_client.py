"""
Pinecone integration for the embedding pipeline.
Provides specialized functionality for managing vectors in Pinecone.
"""
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from pinecone import Pinecone, ServerlessSpec, PodSpec

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
                 region: str = "us-west-2"):
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

    def delete_document(self, document_id: str) -> bool:
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