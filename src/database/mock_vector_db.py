"""
Mock vector database for benchmarking.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class MockVectorDatabaseUploader:
    """
    Mock implementation of a vector database uploader for benchmarking.
    Mimics the interface of VectorDatabaseUploader but doesn't actually connect to any external service.
    """

    def __init__(self, **kwargs):
        """
        Initialize the mock vector database uploader.
        Accepts any kwargs for compatibility with the real uploader.
        """
        self.uploaded_vectors = 0
        self.uploaded_documents = set()
        self.deleted_documents = set()
        self.dimension = kwargs.get("dimension", 512)
        logger.info("Initialized MockVectorDatabaseUploader")

    def upload_vectors(self, vectors: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Mock upload vectors.

        Args:
            vectors: List of vector dictionaries

        Returns:
            Tuple of (success_count, error_count)
        """
        if not vectors:
            return 0, 0

        count = len(vectors)
        self.uploaded_vectors += count
        logger.info(f"[MOCK] Uploaded {count} vectors")

        return count, 0

    def upload_document_chunks(self,
                               document_id: str,
                               chunks: List[Dict[str, Any]],
                               embeddings: List[List[float]],
                               title_enhanced_embeddings: Optional[List[List[float]]] = None) -> Tuple[int, int]:
        """
        Mock upload document chunks.

        Args:
            document_id: The unique ID of the document
            chunks: List of document chunks
            embeddings: List of embeddings corresponding to chunks
            title_enhanced_embeddings: Optional list of title-enhanced embeddings

        Returns:
            Tuple of (success_count, error_count)
        """
        if len(chunks) != len(embeddings):
            return 0, len(chunks)

        # Calculate total vectors (including enhanced if present)
        total_vectors = len(chunks)
        if title_enhanced_embeddings:
            total_vectors *= 2

        self.uploaded_vectors += total_vectors
        self.uploaded_documents.add(document_id)

        logger.info(f"[MOCK] Uploaded {total_vectors} vectors for document {document_id}")
        return total_vectors, 0

    def delete_document(self, document_id: str) -> bool:
        """
        Mock delete document vectors.

        Args:
            document_id: The unique ID of the document

        Returns:
            True (always successful in mock)
        """
        self.deleted_documents.add(document_id)
        logger.info(f"[MOCK] Deleted vectors for document {document_id}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get mock statistics.

        Returns:
            Dictionary with mock statistics
        """
        return {
            "dimension": self.dimension,
            "total_vector_count": self.uploaded_vectors,
            "namespaces": {},
            "documents": {
                "uploaded": len(self.uploaded_documents),
                "deleted": len(self.deleted_documents)
            }
        }

    def index_document(self, document_result: Dict[str, Any]) -> bool:
        """
        Mock implementation of index_document method.
        Used by the pipeline orchestrator to process document results.

        Args:
            document_result: Document processing result containing metadata and chunks

        Returns:
            True if successful, False otherwise
        """
        try:
            if document_result.get('status') != 'success':
                logger.debug(f"[MOCK] Cannot index document with status: {document_result.get('status')}")
                return False

            document_id = document_result.get('metadata', {}).get('id')
            if not document_id:
                logger.debug("[MOCK] Document has no ID, cannot index")
                return False

            chunks = document_result.get('chunks', [])
            if not chunks:
                logger.debug(f"[MOCK] Document {document_id} has no chunks to index")
                return False

            # Generate mock embeddings
            mock_embeddings = [[0.1] * self.dimension for _ in range(len(chunks))]

            # Upload to mock database
            success_count, error_count = self.upload_document_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=mock_embeddings
            )

            logger.info(f"[MOCK] Successfully indexed document {document_id} with {len(chunks)} chunks")
            return success_count > 0

        except Exception as e:
            logger.error(f"[MOCK] Error indexing document: {str(e)}")
            return False