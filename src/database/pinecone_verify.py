"""
Utility script to verify and test Pinecone integration.
Provides functions for verifying synchronization and testing queries.
"""
import logging
import argparse
import sys
import uuid
import time
from typing import Dict, Any, List, Optional

from pinecone import Pinecone, ServerlessSpec

from src.config import ConfigManager
from src.database.vector_db_factory import create_vector_db_uploader
from src.database.pinecone_client import PineconeClient
from src.models.embedding_factory import EmbeddingModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeVerification:
    """Utility class for verifying Pinecone integration with v6.x SDK."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.db_config = self.config_manager.get_database_config()
        self.embedding_config = self.config_manager.get_embedding_config()

        # Create vector database client
        self.client = create_vector_db_uploader(self.db_config)
        if not self.client:
            raise ValueError("Failed to create vector database client")

        # Check if client is a PineconeClient
        if not isinstance(self.client, PineconeClient):
            raise ValueError("Vector database client is not a PineconeClient")

        # Create embedding model
        self.embedding_factory = EmbeddingModelFactory(**self.embedding_config)
        self.embedding_model = self.embedding_factory.create_model()

    def check_connection(self) -> bool:
        """
        Check connection to Pinecone.

        Returns:
            True if connection is successful
        """
        logger.info("Checking connection to Pinecone...")
        if self.client.check_connection():
            logger.info("✅ Connection successful")
            return True
        else:
            logger.error("❌ Connection failed")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from Pinecone index.

        Returns:
            Dictionary with statistics
        """
        logger.info("Getting index statistics...")
        stats = self.client.get_stats()
        logger.info(f"Total vectors: {stats.get('total_vector_count', 0)}")
        return stats

    def test_index_document(self, content: str, title: str = "Test Document") -> Optional[str]:
        """
        Test indexing a document with sample content.

        Args:
            content: Document content to index
            title: Document title

        Returns:
            Document ID if successful, None otherwise
        """
        logger.info(f"Testing document indexing with content: '{content[:50]}...'")

        # Generate a unique document ID
        document_id = f"test_{uuid.uuid4().hex[:8]}"

        # Create a test document structure
        document = {
            "metadata": {
                "id": document_id,
                "title": title,
                "created": int(time.time() * 1000)
            },
            "chunks": [
                {
                    "content": content,
                    "metadata": {
                        "section_title": title,
                        "section_level": 1
                    }
                }
            ],
            "status": "success"
        }

        # Generate an embedding for the content
        embedding = self.embedding_model.encode(content)

        # Upload to Pinecone
        success_count, error_count = self.client.upload_document_chunks(
            document_id=document_id,
            chunks=document["chunks"],
            embeddings=[embedding]
        )

        if success_count > 0:
            logger.info(f"✅ Document indexed successfully with ID: {document_id}")
            return document_id
        else:
            logger.error("❌ Failed to index document")
            return None

    def test_query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Test querying the index with a text query.

        Args:
            query_text: Text to search for
            top_k: Number of results to return

        Returns:
            List of matching results
        """
        logger.info(f"Testing query: '{query_text}'")

        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text)

        # Query Pinecone
        results = self.client.query_vectors(
            query_vector=query_embedding,
            top_k=top_k
        )

        # Log results
        logger.info(f"Found {len(results)} matches")
        for i, result in enumerate(results):
            score = result.get('score', 0)
            text = result.get('metadata', {}).get('text', '')[:100]
            doc_id = result.get('metadata', {}).get('document_id', '')
            logger.info(f"  {i+1}. Score: {score:.4f}, Doc: {doc_id}, Text: '{text}...'")

        return results

    def test_delete(self, document_id: str) -> bool:
        """
        Test deleting a document.

        Args:
            document_id: Document ID to delete

        Returns:
            True if deletion was successful
        """
        logger.info(f"Testing deletion of document: {document_id}")

        # Delete the document
        if self.client.delete_document(document_id):
            logger.info(f"✅ Document {document_id} deleted successfully")
            return True
        else:
            logger.error(f"❌ Failed to delete document {document_id}")
            return False

    def test_sync(self, document_id: str, new_content: str) -> bool:
        """
        Test synchronizing a document with new content.

        Args:
            document_id: Document ID to sync
            new_content: New content for the document

        Returns:
            True if sync was successful
        """
        logger.info(f"Testing synchronization of document: {document_id}")

        # Create updated chunks
        new_chunks = [
            {
                "content": new_content,
                "metadata": {
                    "section_title": "Updated Test Document",
                    "section_level": 1,
                    "updated": int(time.time() * 1000)
                }
            }
        ]

        # Generate embeddings
        new_embedding = self.embedding_model.encode(new_content)

        # Sync the document
        if self.client.sync_document(
            document_id=document_id,
            new_chunks=new_chunks,
            embeddings=[new_embedding]
        ):
            logger.info(f"✅ Document {document_id} synchronized successfully")
            return True
        else:
            logger.error(f"❌ Failed to synchronize document {document_id}")
            return False

    def run_all_tests(self) -> None:
        """Run a complete test suite for Pinecone integration."""
        logger.info("Starting Pinecone integration verification...")

        # Check connection
        if not self.check_connection():
            logger.error("Cannot proceed with tests due to connection failure")
            return

        # Get stats
        stats = self.get_stats()

        # Test indexing
        test_content = "This is a test document for Pinecone integration verification. It contains some text that will be used for embedding and retrieval testing."
        document_id = self.test_index_document(test_content)
        if not document_id:
            logger.error("Cannot proceed with tests due to indexing failure")
            return

        # Give Pinecone a moment to index
        logger.info("Waiting for indexing to complete...")
        time.sleep(2)

        # Test query
        results = self.test_query("test document for verification")

        # Test sync
        new_content = "This is an updated test document with new content. The document has been modified to test synchronization capabilities."
        sync_success = self.test_sync(document_id, new_content)

        # Test query again after sync
        if sync_success:
            logger.info("Testing query after synchronization...")
            time.sleep(2)  # Give time for index to update
            results_after_sync = self.test_query("updated test document")

        # Test deletion
        delete_success = self.test_delete(document_id)

        # Final stats
        logger.info("Final index statistics after tests:")
        final_stats = self.get_stats()

        # Summary
        logger.info("=== Test Summary ===")
        logger.info(f"Connection: {'✅ Success' if self.check_connection() else '❌ Failed'}")
        logger.info(f"Indexing: {'✅ Success' if document_id else '❌ Failed'}")
        logger.info(f"Querying: {'✅ Success' if results else '❌ Failed'}")
        logger.info(f"Synchronization: {'✅ Success' if sync_success else '❌ Failed'}")
        logger.info(f"Deletion: {'✅ Success' if delete_success else '❌ Failed'}")


def main():
    """Main function to run verification from command line."""
    parser = argparse.ArgumentParser(description="Verify Pinecone integration")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--test", choices=["all", "connection", "index", "query", "sync", "delete"],
                      default="all", help="Test to run")
    parser.add_argument("--document-id", help="Document ID for delete or sync tests")
    parser.add_argument("--content", default="Test content for Pinecone verification",
                      help="Content to use for testing")

    args = parser.parse_args()

    try:
        verifier = PineconeVerification(args.config)

        if args.test == "all":
            verifier.run_all_tests()
        elif args.test == "connection":
            verifier.check_connection()
            verifier.get_stats()
        elif args.test == "index":
            verifier.test_index_document(args.content)
        elif args.test == "query":
            verifier.test_query(args.content)
        elif args.test == "delete":
            if not args.document_id:
                logger.error("Document ID is required for delete test")
                sys.exit(1)
            verifier.test_delete(args.document_id)
        elif args.test == "sync":
            if not args.document_id:
                logger.error("Document ID is required for sync test")
                sys.exit(1)
            verifier.test_sync(args.document_id, args.content)

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        sys.exit(1)

    logger.info("Verification completed")


if __name__ == "__main__":
    main()